from pathlib import Path
import copy
import os
import logging

from tqdm import tqdm
import torch
import sentencepiece as spm
import k2
from icefall.utils import add_sos
from icefall.lexicon import Lexicon
from icefall.utils import write_error_stats_with_timestamps

from decode import get_parser, decode_dataset
from train import get_params, get_transducer_model
from keyword_spotting import get_model
from asr_datamodule import AsrDataModule


FUSE_BN = False
AVG_PATH = None #"exp/en/qat/a8w4/e200-avg64-a8w4.pt"
WEIGHT_ONLY = True
if WEIGHT_ONLY:
    from int8_w8_outscaling import insert_observers, quantize_model
else:
    # from int8_inoutscaling import insert_observers, quantize_model
    from int8_a8w8 import insert_observers, quantize_model


def main(params, sp):
    device = "cuda"
    dtype = torch.float32
    dtype_act = torch.int8
    dtype_weight = torch.int4
    dtype_scale = torch.float32

    # Prepare model
    model = get_transducer_model(params)
    get_model(params, model, device, sp, params.exp_dir / f"epoch-{params.epoch}-avg-{params.avg}.pt")
    model.eval()
    model.encoder.remove_weight_reparameterizations(fuse_bn=FUSE_BN, ema=False)
    model.decoder.remove_weight_reparameterizations()
    model.joiner.remove_weight_reparameterizations()
    model.to(device=device, dtype=dtype)

    # Prepare data
    datamodule = AsrDataModule(params)
    train_dl = datamodule.get_train_dataloader(params, sp)
    test_dl_dict = datamodule.get_test_dataloader_dict(params)
    if "fast_beam_search" in params.decoding_method:
        if params.decoding_method == "fast_beam_search_nbest_LG":
            lexicon = Lexicon(params.lang_dir)
            word_table = lexicon.word_table
            lg_filename = params.lang_dir / "LG.pt"
            logging.info(f"Loading {lg_filename}")
            decoding_graph = k2.Fsa.from_dict(
                torch.load(lg_filename, map_location=device)
            )
            decoding_graph.scores *= params.ngram_lm_scale
        else:
            word_table = None
            decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None
        word_table = None

    # Test
    # print("Testing model")
    # with open(os.devnull, "w") as fnull:
    #     with torch.no_grad():
    #         with torch.amp.autocast("cuda", enabled=True):
    #             for test_set, test_dl in test_dl_dict.items():
    #                 results_dict = decode_dataset(
    #                     dl=test_dl,
    #                     params=params,
    #                     model=model,
    #                     sp=sp,
    #                     word_table=word_table,
    #                     decoding_graph=decoding_graph,
    #                     dtype=dtype,
    #                 )
    #                 for key, results in results_dict.items():
    #                     print(f"\t{test_set}-{key}")
    #                     results = sorted(results)
    #                     wer = write_error_stats_with_timestamps(
    #                         fnull, f"{test_set}-{key}", results, enable_log=True
    #                     )
    # exit()

    # Insert observers
    act_observer = "minmax"
    act_scale_observer = "minmax"
    for module in [
        model.encoder,
        model.decoder,
        model.joiner,
    ]:
        insert_observers(
            module,
            dtype_act,
            dtype_weight,
            dtype_scale,
            act_observer,
            act_scale_observer=act_scale_observer,
            alpha=0.5
        )

    if WEIGHT_ONLY:
        # Quantize the model
        quantize_model(model, "quantized")
    else:
        for mode in ["quantized"]:
        # for mode in ["quantized_wonly"]:
        # for mode in ["scaled", "quantized"]:
            for idx, batch in tqdm(
                enumerate(train_dl, start=1),
                desc=f"Calibrating for mode=`{mode}`"
            ):
                x = batch["inputs"].to(dtype=dtype, device=device)
                x_lens = batch["supervisions"]["num_frames"].to(device)
                texts = batch["supervisions"][params.cutset_text]
                y = sp.encode(texts, out_type=int)
                y = k2.RaggedTensor(y).to(device)

                # Forward
                with torch.amp.autocast("cuda", enabled=True):
                    with torch.no_grad():
                        encoder_out, x_lens, _ = model.encoder(x, x_lens)
                        row_splits = y.shape.row_splits(1)
                        y_lens = row_splits[1:] - row_splits[:-1]
                        blank_id = params.blank_id
                        sos_y = add_sos(y, sos_id=blank_id)
                        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
                        decoder_out = model.decoder(sos_y_padded)
                        y_padded = y.pad(mode="constant", padding_value=0).to(torch.int64)
                        boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
                        boundary[:, 2] = y_lens
                        boundary[:, 3] = x_lens
                        lm = model.simple_lm_proj(decoder_out)
                        am = model.simple_am_proj(encoder_out)
                        _, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                            lm=lm.float(),
                            am=am.float(),
                            symbols=y_padded,
                            termination_symbol=blank_id,
                            lm_only_scale=params.lm_scale,
                            am_only_scale=params.am_scale,
                            boundary=boundary,
                            reduction="none",
                            delay_penalty=0.0,
                            return_grad=True,
                        )
                        ranges = k2.get_rnnt_prune_ranges(
                            px_grad=px_grad,
                            py_grad=py_grad,
                            boundary=boundary,
                            s_range=params.prune_range,
                        )
                        am_pruned, lm_pruned = k2.do_rnnt_pruning(
                            am=model.joiner.encoder_proj(encoder_out),
                            lm=model.joiner.decoder_proj(decoder_out),
                            ranges=ranges,
                        )
                        logits = model.joiner(am_pruned, lm_pruned, project_input=False)

                # Check if logits are finite
                is_finite = torch.isfinite(logits)
                if not torch.all(is_finite):
                    print(
                        "logits are not finite!\n"
                        f"logits: {logits}"
                    )
                # print(logits.shape)
                if idx >= 1:
                    # for module in [model.encoder, model.decoder, model.joiner]:
                    #     for submodule in module.modules():
                    #         if hasattr(submodule, "set_mode"):
                    #             submodule.set_mode(mode)
                    quantize_model(model, mode)
                    break

    # Test
    print("Testing quantized model")
    with open(os.devnull, "w") as fnull:
        with torch.no_grad():
            for test_set, test_dl in test_dl_dict.items():
                results_dict = decode_dataset(
                    dl=test_dl,
                    params=params,
                    model=model,
                    sp=sp,
                    word_table=word_table,
                    decoding_graph=decoding_graph,
                    dtype=dtype,
                )
                for key, results in results_dict.items():
                    print(f"\t{test_set}-{key}")
                    results = sorted(results)
                    wer = write_error_stats_with_timestamps(
                        fnull, f"{test_set}-{key}", results, enable_log=True
                    )

    if AVG_PATH is not None:
        breakpoint()
        torch.save(
            {'model': model.state_dict()},
            AVG_PATH
        )
    exit()

    # Quantize model (custom impl)
    for module in [model.encoder, model.decoder, model.joiner]:
        for submodule in module.modules():
            if hasattr(submodule, "set_int_mode"):
                submodule.set_int_mode()

    # Test
    print("Testing int8 model")
    with open(os.devnull, "w") as fnull:
        for test_set, test_dl in test_dl_dict.items():
            results_dict = decode_dataset(
                dl=test_dl,
                params=params,
                model=model,
                sp=sp,
                word_table=word_table,
                decoding_graph=decoding_graph,
                dtype=dtype,
            )
            for key, results in results_dict.items():
                print(f"\t{test_set}-{key}")
                results = sorted(results)
                cer = write_error_stats_with_timestamps(
                    fnull, f"{test_set}-{key}", results, enable_log=True
                )


if __name__ == "__main__":
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args([
        # Model configs
        '--channels', '256',
        '--channels-expansion', '1024',
        '--dilations-version', '11',
        '--kernel-size', '8',
        '--encoder-dropout', '0.075',
        '--encoder-activation', 'ReLU',
        '--encoder-se-activation', 'ReLU',
        '--encoder-norm', 'BatchNorm',
        '--decoder-dim', '256',
        '--joiner-dim', '256',
        '--scale-limit', '2.0',
        '--ema-gamma', '0.97',
        '--chunksize', '8',

        # Checkpoints and data
        '--exp-dir', 'exp/en/l11_chunk8_g.97_wlimit.3',
        '--epoch', '200',
        '--avg', '64',
        '--use-averaged-model', 'True',
        '--update-bn', 'False',
        '--data-libri-train', 'True',
        '--data-libri-test-clean', 'True',
        '--data-libri-test-other', 'True',
        '--manifest-dir', 'data/en/fbank',
        '--bpe-model', str(Path(__file__).parent / "../data/en/lang_bpe_500/bpe.model"),
        # '--cutset-text', 'custom.ipa_filtered',
        '--decoding-method', 'fast_beam_search',
        '--max-duration', '600',
    ])
    args.exp_dir = Path(args.exp_dir)
    params = get_params()
    params.update(vars(args))

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO
    )
    main(params, sp)
