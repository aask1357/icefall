#!/usr/bin/env python3

import time
import argparse
import copy
import logging
import warnings
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import k2
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from icefall import diagnostics
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    update_averaged_model,
    average_checkpoints_with_averaged_model,
    load_checkpoint,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    display_and_save_batch,
    setup_logger,
    str2bool,
    parse_hyp_and_timestamp,
)
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed, LOG_EPSILON

from asr_datamodule import AsrDataModule
from decoder import Decoder
from joiner import Joiner
from partition_params import update_param_groups
from encoder import Encoder
from model import Transducer
from optim import Eden, Eve, LRScheduler, ExponentialWarmupLR, LinearWarmupLR, CosineWarmupLR
from plot_params import plot_params
from beam_search import fast_beam_search_one_best
from custom_utils import MetricsTracker, write_error_stats
from train import add_model_arguments


LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, LRScheduler]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=40,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="Eve",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay-projection",
        type=float,
        default=1e-5,
        help="Not Used. Ignore it.",
    )

    parser.add_argument(
        "--initial-lr",
        type=float,
        default=0.003,
        help="""The initial learning rate. This value should not need to be
        changed.""",
    )

    parser.add_argument(
        "--scheduler-name",
        type=str,
        default="Eden",
        choices=["Eden", "ExponentialWarmupLR", "LinearWarmupLR", "CosineWarmupLR"],
    )

    parser.add_argument(
        "--lr-warmup-iterations",
        type=int,
        default=3000,
        help="Number of iterations for warmup.",
    )

    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.98,
        help="Number of iterations for warmup.",
    )

    parser.add_argument(
        "--lr-eta-min",
        type=float,
        default=1.0e-4,
        help="Number of iterations for warmup.",
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate decreases.
        We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=10,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network) part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=100,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--delay-penalty",
        type=float,
        default=0.0,
        help="""A constant value used to penalize symbol delay,
        to encourage streaming models to emit symbols earlier.
        See https://github.com/k2-fsa/k2/issues/955 and
        https://arxiv.org/pdf/2211.00490.pdf for more details.""",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=20.0,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        """,
    )

    parser.add_argument(
        "--max-contexts",   # beam size (https://www.danielpovey.com/files/2023_icassp_fast_parallel_transducer_decoding.pdf
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--compute-cer",
        type=str2bool,
        default=False,
        help="Whether to compute CER instead of WER.",
    )

    parser.add_argument(
        "--ft-path-start",
        type=str
    )

    parser.add_argument(
        "--ft-path-end",
        type=str
    )
    
    parser.add_argument(
        "--model-warmup-step",
        type=int,
        default=0,
        help="Number of steps for model warmup.",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - warm_step: The warm_step for Noam optimizer.
    """
    params = AttributeDict(
        {
            "frame_shift_ms": 10.0,
            "batch_idx_train": 0,
            "log_interval": 100,
            "reset_interval": 200,
            "valid_interval": 500,  # For the 100h subset, use 800
            # parameters for conformer
            "feature_dim": 80,
            # "subsampling_factor": 4,
            # parameters for Noam
            "env_info": get_env_info(),
            "is_pnnx": False,
        }
    )

    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    if params.dilations_version == 1:
        dilations = [1, 2, 4, 1, 2, 4]
    elif params.dilations_version == 2:
        dilations = [1 for _ in range(16)]
    elif params.dilations_version == 3:
        dilations = [1 for _ in range(12)]
    elif params.dilations_version == 4:
        dilations = [1 for _ in range(21)]
    elif params.dilations_version == 5:
        dilations = [1 for _ in range(48)]
    else:
        dilations = [1 for _ in range(params.dilations_version)]
    encoder = Encoder(
        num_features=params.feature_dim,
        subsampling_factor=params.subsampling_factor,
        channels=params.channels,
        channels_expansion=params.channels_expansion,
        kernel_size=params.kernel_size,
        dilations=dilations,
        output_channels=params.encoder_dim,
        dropout=params.encoder_dropout,
        activation=params.encoder_activation,
        norm=params.encoder_norm,
        se_activation=params.encoder_se_activation,
        act_bal=params.act_bal,
        zero_init_residual=params.zero_init_residual,
        se_gate=params.se_gate,
        gamma=params.ema_gamma,
        use_cache=params.use_cache,
        mean=params.encoder_mean,
        std=params.encoder_std,
        chunksize=params.chunksize,
        scale_limit=params.scale_limit,
        skip=params.skip,
        n_bits_act=params.n_bits_act,
        n_bits_weight=params.n_bits_weight,
        eps=params.eps,
        decay=params.quantizer_decay,
        quantile=params.quantizer_quantile,
        learnable_gamma=params.quantizer_learnable_gamma,
        gamma_min=params.quantizer_gamma_min,
        gamma_max=params.quantizer_gamma_max,
        weight_quantizer_mode=params.weight_quantizer_mode,
        redundant_bias=True,
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
        n_bits_act=params.n_bits_act,
        n_bits_weight=params.n_bits_weight,
        eps=params.eps,
        decay=params.quantizer_decay,
        quantile=params.quantizer_quantile,
        learnable_gamma=params.quantizer_learnable_gamma,
        gamma_min=params.quantizer_gamma_min,
        gamma_max=params.quantizer_gamma_max,
        weight_quantizer_mode=params.weight_quantizer_mode,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
        n_bits_act=params.n_bits_act,
        n_bits_weight=params.n_bits_weight,
        eps=params.eps,
        decay=params.quantizer_decay,
        quantile=params.quantizer_quantile,
        learnable_gamma=params.quantizer_learnable_gamma,
        gamma_min=params.quantizer_gamma_min,
        gamma_max=params.quantizer_gamma_max,
        weight_quantizer_mode=params.weight_quantizer_mode,
    )
    return joiner


def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    rank: int = 0,
) -> Tuple[Optional[Dict[str, Any]], Optional[nn.Module]]:
    if params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        if params.ft_path_start is not None:
            # Load pretrained checkpoint -> rescale weight
            # sd = torch.load(
            #     params.ft_path, weights_only=False,
            #     map_location="cpu"
            # )
            sd = average_checkpoints_with_averaged_model(
                filename_start=params.ft_path_start,
                filename_end=params.ft_path_end,
                device="cpu",
            )
            new_sd = {}
            for k, v in sd.items():
                if k == "encoder.proj.weight":
                    # [512, 512, 1] -> [512, 256, 1]
                    v = v[:, :v.size(1)//2, :].contiguous()
                new_sd[k] = v
            model.load_state_dict(new_sd, strict=False)
            logging.info(
                f"Loaded pretrained model from {params.ft_path_start}"
                f" to {params.ft_path_end}"
            )

        model_avg: Optional[nn.Module] = None
        if rank == 0:
            model_avg = copy.deepcopy(model)
        return None, model_avg

    assert filename.is_file(), f"{filename} does not exist!"

    # Rescale weight -> load checkpoint
    for module in model.modules():
        if hasattr(module, "rescale_weight"):
            module.rescale_weight()

    model_avg: Optional[nn.Module] = None
    if rank == 0:
        model_avg = copy.deepcopy(model)

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params, model_avg


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
    warmup: float = 2.0,
    rank: int = 0,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute RNN-T loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Conformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
     warmup: a floating point value which increases throughout training;
        values >= 1.0 are fully warmed up and have all modules present.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)
    texts = batch["supervisions"][params.cutset_text]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y).to(device)

    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss = model(
            x=feature,
            x_lens=feature_lens,
            y=y,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
            warmup=warmup,
            reduction="none",
            delay_penalty=params.delay_penalty if warmup >= 2.0 else 0,
        )
        simple_loss_is_finite = torch.isfinite(simple_loss)
        pruned_loss_is_finite = torch.isfinite(pruned_loss)
        is_finite = simple_loss_is_finite & pruned_loss_is_finite
        if not torch.all(is_finite):
            logging.info(
                "Not all losses are finite!\n"
                f"simple_loss: {simple_loss}\n"
                f"pruned_loss: {pruned_loss}"
            )
            if rank == 0:
                display_and_save_batch(batch, params=params, sp=sp)
                torch.save(model.state_dict(), params.exp_dir / f"epoch-{params.cur_epoch}-failed.pt")
            simple_loss = simple_loss[simple_loss_is_finite]
            pruned_loss = pruned_loss[pruned_loss_is_finite]

            # If either all simple_loss or pruned_loss is inf or nan,
            # we stop the training process by raising an exception
            if torch.all(~simple_loss_is_finite) or torch.all(~pruned_loss_is_finite):
                raise ValueError(
                    "There are too many utterances in this batch "
                    "leading to inf or nan losses."
                )

        simple_loss = simple_loss.sum()
        pruned_loss = pruned_loss.sum()
        pruned_loss_scale = (
            0.0 if warmup < 1.0 else (0.1 if warmup > 1.0 and warmup < 2.0 else 1.0)
        )
        loss = params.simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # info["frames"] is an approximate number for two reasons:
        # (1) The acutal subsampling factor is ((lens - 1) // 2 - 1) // 2
        # (2) If some utterances in the batch lead to inf/nan loss, they
        #     are filtered out.
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # `utt_duration` and `utt_pad_proportion` would be normalized by `utterances`  # noqa
    info["utterances"] = feature.size(0)
    # averaged input duration in frames over utterances
    info["utt_duration"] = feature_lens.sum().item()
    # averaged padding proportion over utterances
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )

    # Note: We use reduction=sum while computing the loss.
    info["simple_loss"] = simple_loss.detach().cpu().item()
    info["pruned_loss"] = pruned_loss.detach().cpu().item()

    return loss, info


@torch.no_grad()
def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    decoding_graph: Optional[k2.Fsa] = None,
    rank: int = 0,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    id_ref_hyp_list = []
    _model = model.module if isinstance(model, DDP) else model
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    for batch_idx, batch in enumerate(valid_dl):
        feature = batch["inputs"]
        assert feature.ndim == 3

        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)
        texts = batch["supervisions"][params.cutset_text]
        y = sp.encode(texts, out_type=int)
        y = k2.RaggedTensor(y).to(device)
        feature = feature.to(device)

        num_tail_padded_frames = 36
        feature = torch.nn.functional.pad(
            feature,
            (0, 0, 0, num_tail_padded_frames),
            mode="constant",
            value=LOG_EPSILON,
        )
        feature_lens += num_tail_padded_frames
        with torch.amp.autocast("cuda", enabled=params.use_fp16):
            encoder_out, feature_lens, _ = _model.encoder(feature, feature_lens, warmup=2.0)

            # Calculate WER
            res = fast_beam_search_one_best(
                model=_model,
                decoding_graph=decoding_graph,
                encoder_out=encoder_out,
                encoder_out_lens=feature_lens,
                beam=params.beam,
                max_contexts=params.max_contexts,
                max_states=params.max_states,
                return_timestamps=True,
            )
            hyps, timestamps = parse_hyp_and_timestamp(
                res=res,
                sp=sp,
                subsampling_factor=params.subsampling_factor,
                frame_shift_ms=params.frame_shift_ms,
            )
            cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = ref_text.split()
                id_ref_hyp_list.append((cut_id, ref_words, hyp_words))

            # Calculate loss
            padding = num_tail_padded_frames // params.subsampling_factor
            feature = encoder_out[:, :-padding, :]  # remove the tail padding for loss calculation
            feature_lens -= padding
            simple_loss, pruned_loss = model(
                x=feature,
                x_lens=feature_lens,
                y=y,
                prune_range=params.prune_range,
                am_scale=params.am_scale,
                lm_scale=params.lm_scale,
                warmup=1.0,
                reduction="none",
                delay_penalty=params.delay_penalty,
                is_encoder_processed=True,
            )
            simple_loss_is_finite = torch.isfinite(simple_loss)
            pruned_loss_is_finite = torch.isfinite(pruned_loss)
            is_finite = simple_loss_is_finite & pruned_loss_is_finite
            if not torch.all(is_finite):
                logging.info(
                    "Not all losses are finite!\n"
                    f"simple_loss: {simple_loss}\n"
                    f"pruned_loss: {pruned_loss}"
                )
                if rank == 0:
                    display_and_save_batch(batch, params=params, sp=sp)
                    
                simple_loss = simple_loss[simple_loss_is_finite]
                pruned_loss = pruned_loss[pruned_loss_is_finite]

                # If either all simple_loss or pruned_loss is inf or nan,
                # we stop the training process by raising an exception
                if torch.all(~simple_loss_is_finite) or torch.all(~pruned_loss_is_finite):
                    raise ValueError(
                        "There are too many utterances in this batch "
                        "leading to inf or nan losses."
                    )

            simple_loss = simple_loss.sum()
            pruned_loss = pruned_loss.sum()
            info = MetricsTracker()
            info["simple_loss"] = simple_loss.cpu().item()
            info["pruned_loss"] = pruned_loss.cpu().item()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info["frames"] = (feature_lens // params.subsampling_factor).sum().item()
        tot_loss = tot_loss + info

    ref_len, tot_errs = write_error_stats(
        id_ref_hyp_list,
        compute_CER=params.compute_cer
    )
    tot_loss["ref_len"] = ref_len
    tot_loss["tot_errs"] = tot_errs
    if world_size > 1:
        tot_loss.reduce(device)
    wer = tot_loss.pop("tot_errs") / tot_loss.pop("ref_len") * 100
    wer = round(wer, 2)
    if params.compute_cer:
        tot_loss["CER"] = wer
    else:
        tot_loss["WER"] = wer
    print(tot_loss)

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    sp: spm.SentencePieceProcessor,
    train_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    rank: int = 0,
) -> None:
    model.train()

    tot_loss = MetricsTracker()
    st = time.perf_counter()
    for batch_idx, batch in enumerate(train_dl, start=0):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])
        print(
            f"\rbatch load time: {time.perf_counter() - st:.2f} sec",
            end="",
            flush=True
        )

        try:
            with torch.amp.autocast("cuda", enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                    warmup=(params.batch_idx_train / (params.model_warmup_step + 0.1)),
                    rank=rank,
                )
            # summary stats
            tot_loss = tot_loss + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()

            scheduler.step_batch(params.batch_idx_train)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except:  # noqa
            if rank == 0:
                display_and_save_batch(batch, params=params, sp=sp)
            raise

        if params.print_diagnostics and batch_idx == 30:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if batch_idx % params.log_interval == 0 and not params.print_diagnostics:
            cur_lr = scheduler.get_last_lr()[0]
            logging.info(
                f"\rEpoch {params.cur_epoch}, "
                f"batch {batch_idx}, "
                f"pruned_loss: {tot_loss['pruned_loss'] / tot_loss['frames']:.2f}, "
                f"simple_loss: {tot_loss['simple_loss'] / tot_loss['frames']:.2f}, "
                f"batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}"
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )
                if scaler.is_enabled():
                    tb_writer.add_scalar(
                        "train/scale", scaler.get_scale(), params.batch_idx_train
                    )

        st = time.perf_counter()

    tot_loss.reduce(loss.device)
    if tb_writer is not None:
        tot_loss.write_summary(tb_writer, "train/", params.cur_epoch)


def valid_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl_dict: Dict[str, torch.utils.data.DataLoader],
    world_size: int = 1,
    tb_writer: Optional[SummaryWriter] = None,
    rank: int = 0,
    decoding_graph: Optional[k2.Fsa] = None,
) -> None:
    model.eval()

    for name, valid_dl in valid_dl_dict.items():
        logging.info(f"Computing validation loss for {name}")
        valid_info = compute_validation_loss(
            params=params,
            model=model,
            sp=sp,
            valid_dl=valid_dl,
            world_size=world_size,
            decoding_graph=decoding_graph,
            rank=rank,
        )
        logging.info(f"Epoch {params.cur_epoch}, {name}: {valid_info}")
        if tb_writer is not None:
            valid_info.write_summary(
                tb_writer, f"valid/{name}_", params.cur_epoch
            )
            model_params_to_plot = {}
            plot_params(model_params_to_plot, model)
            for key, value in model_params_to_plot.items():
                tb_writer.add_histogram(key, value, params.cur_epoch)


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))
    if params.full_libri is False:
        params.valid_interval = 800

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.start_epoch > 0, params.start_epoch
    checkpoints, model_avg = load_checkpoint_if_available(
        params=params, model=model, rank=rank
    )
    model.to(device)

    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank])

    nowd = dict(
        regex_list = [
            "ema\\.weight",
            "encoder\\.cnn\\.\\d+\\.scale",
            "\\.quantizer_(act|weight)\\.gamma$",
        ],
        weight_decay = 0.0,
        limit = None,
        scale_max = None,
    )
    model_params = update_param_groups(model, [nowd])

    if params.optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model_params,
            lr=params.initial_lr,
            weight_decay=params.weight_decay,
        )
    elif params.optimizer_name == "Eve":
        optimizer = Eve(model_params, lr=params.initial_lr)
    else:
        raise ValueError(f"Unsupported optimizer: {params.optimizer_name}")

    if params.scheduler_name == "Eden":
        scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)
    elif params.scheduler_name == "ExponentialWarmupLR":
        scheduler = ExponentialWarmupLR(optimizer, params.lr_warmup_iterations, params.lr_gamma)
    elif params.scheduler_name == "LinearWarmupLR":
        scheduler = LinearWarmupLR(optimizer, params.lr_warmup_iterations, params.num_epochs, params.lr_eta_min)
    elif params.scheduler_name == "CosineWarmupLR":
        scheduler = CosineWarmupLR(optimizer, params.lr_warmup_iterations, params.num_epochs, params.lr_eta_min)
    else:
        raise ValueError(f"Unsupported scheduler: {params.scheduler_name}")

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        diagnostic = diagnostics.attach_diagnostics(model)

    datamodule = AsrDataModule(args)
    train_dl = datamodule.get_train_dataloader(params, sp, checkpoints)

    valid_dl_dict = datamodule.get_valid_dataloader_dict(params)
    decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)

    if checkpoints is None:
        params.cur_epoch = 0
        # use_fp16 = params.use_fp16
        # params.use_fp16 = False
        valid_one_epoch(
            params=params,
            model=model,
            sp=sp,
            valid_dl_dict=valid_dl_dict,
            world_size=world_size,
            tb_writer=tb_writer,
            rank=rank,
            decoding_graph=decoding_graph,
        )
        # params.use_fp16 = use_fp16

    scaler = GradScaler("cuda", enabled=params.use_fp16)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    if not params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            sp=sp,
            params=params,
            warmup=2.0,
            scaler=scaler,
        )

    if checkpoints and "batch_idx_train" in checkpoints:
        params.batch_idx_train = checkpoints["batch_idx_train"]

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        valid_one_epoch(
            params=params,
            model=model,
            sp=sp,
            valid_dl_dict=valid_dl_dict,
            world_size=world_size,
            tb_writer=tb_writer,
            rank=rank,
            decoding_graph=decoding_graph,
        )

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    sp: spm.SentencePieceProcessor,
    params: AttributeDict,
    warmup: float,
    scaler: GradScaler,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        st = time.perf_counter()
        batch = train_dl.dataset[cuts]
        logging.info(
            f"batch load time: {time.perf_counter() - st:.2f} sec"
        )
        try:
            with torch.amp.autocast("cuda", enabled=params.use_fp16):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                    warmup=warmup,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            raise


def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
