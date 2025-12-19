#!/usr/bin/env python3
from pathlib import Path
import wave
import typing as tp
import math
import random
from collections import defaultdict
import copy
import time

import matplotlib.pyplot as plt
import sklearn.metrics
from tqdm import tqdm
import sentencepiece as spm
import numpy as np
import torch
import torch.utils.data
import k2

from asr_datamodule import AsrDataModule
from decode import get_parser, decode_one_batch
from train import get_params, get_transducer_model
from update_bn import update_bn
from beam_search import fast_beam_search_one_best
from local.custom_fbank import CustomFbank, CustomFbankConfig

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    add_sos,
    AttributeDict,
    DecodingResults,
    parse_hyp_and_timestamp,
    setup_logger,
    store_transcripts_and_timestamps,
    str2bool,
    write_error_stats_with_timestamps,
)

LOG_EPS = math.log(1e-10)

WORDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]

UNKNOWN_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
    
    # added at v0.02
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
]
ALL_WORDS = WORDS + UNKNOWN_WORDS
SILENCE = "_silence_"   # background noise
UNKNOWN = "_unknown_"   # words that don't belong to anywhere
LABELS = ALL_WORDS + [SILENCE, UNKNOWN]

MERGE = "LOG_ADD"   # "MAX" or "LOG_ADD"

class SpeechCommandsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir: str = "/home/shahn/Datasets/SpeechCommands", 
        mode: str = "train",
    ):
        r''' return label \in WORDS + [SILENCE, UNKNOWN]'''
        super().__init__()
        self.base_dir = Path(base_dir)
        
        if mode == "train":
            filelist = "train_list.txt"
        elif mode == "valid":
            filelist = "validation_list.txt"
        elif mode == "test":
            filelist = "testing_list.txt"
        else:
            raise ValueError(f"Invalid mode {mode}")
        
        with open(self.base_dir / filelist, "r") as f:
            self.filelist = f.read().splitlines()
    
    def __len__(self) -> int:
        return len(self.filelist)
    
    def __getitem__(self, idx: int) -> tp.Tuple[np.ndarray, str, str]:
        file = Path(self.filelist[idx])
        
        # load audio
        wav_read = wave.open(str(self.base_dir / file))
        buffer = wav_read.readframes(16_000)
        audio = np.frombuffer(buffer, dtype=np.int16, count=len(buffer)//2, offset=0)
        audio = audio.astype(np.float32) / 32768.0
        if len(audio) < 16_000:
            audio = np.pad(audio, (0, 16_000-len(audio)))
        
        # get label
        word = str(file.parent)
        if word in WORDS:
            label = word
        elif word == SILENCE:
            label = SILENCE
        else:
            label = UNKNOWN
        
        return audio, label, str(self.base_dir / file)


class SpeechCommandsCustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir: str = "/home/shahn/Datasets/SpeechCommands", 
    ):
        r''' return label \in ALL_WORDS + [SILENCE, UNKNOWN] '''
        super().__init__()
        self.base_dir = Path(base_dir)
        
        self.filelist: tp.List[Path] = []
        for file in Path(base_dir).rglob("*.wav"):
            relative_path = file.relative_to(base_dir)
            if not str(relative_path).startswith("_background_noise_"):
                self.filelist.append(relative_path)
    
    def __len__(self) -> int:
        return len(self.filelist)
    
    def __getitem__(self, idx: int) -> tp.Tuple[np.ndarray, str]:
        file = self.filelist[idx]
        
        # load audio
        wav_read = wave.open(str(self.base_dir / file))
        buffer = wav_read.readframes(16_000)
        audio = np.frombuffer(buffer, dtype=np.int16, count=len(buffer)//2, offset=0)
        audio = audio.astype(np.float32) / 32768.0
        if len(audio) < 16_000:
            audio = np.pad(audio, (0, 16_000-len(audio)))
        
        # get label
        label = str(file.parent)
        
        return audio, label.upper()


def get_model(params, model, device, args, sp, path=None):
    if path is not None and Path(path).exists():
        state_dict = torch.load(path, map_location=device)
        model.to(device)
        model.load_state_dict(state_dict["model"])
        model.eval()
        return

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device)) # type: ignore
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)  # type: ignore
            print(f"load checkpoint {params.epoch}")
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            print(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            print(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            print(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
    model.to(device)
    model.eval()
    
    if ((params.avg >= 1 and params.use_averaged_model) or params.avg > 1) \
        and params.update_bn and params.encoder_norm == "BatchNorm":
        update_bn(model.encoder, args, sp)


def rnnt_loss(model, specs, T_spec, blank_id, y):
    encoder_out, encoder_out_lens, _ = model.encoder(x=specs, x_lens=T_spec)
    encoder_out = model.joiner.encoder_proj(encoder_out)    # [B, T, C]
    
    sos_y = add_sos(y, sos_id=blank_id)

    # sos_y_padded: [B, S + 1], start with SOS.
    sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

    # decoder_out: [B, S + 1, decoder_dim]
    decoder_out = model.decoder(sos_y_padded)
    decoder_out = model.joiner.decoder_proj(decoder_out)    # [B, S+1, C]
    
    logits = model.joiner(
        encoder_out.unsqueeze(2),
        decoder_out.unsqueeze(1),
        project_input=False
    )       # [B, T, S+1, C]
    y_padded = y.pad(mode="constant", padding_value=0)
    y_padded = y_padded.to(torch.int64) # [B, S]

    logprob_list = []
    for i in range(specs.size(0)):
        S = len(y[i])
        nll = k2.rnnt_loss(
            logits[i:i+1, :, :S+1, :],
            y_padded[i:i+1, :S],
            blank_id,
            reduction="none"
        )
        logprob_list.append(-nll.item())
    return logprob_list


def compute_eer(label, pred, positive_label=1, key="delete_it"):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, pos_label=positive_label, drop_intermediate=True)
    fnr = 1 - tpr
    plt.clf()
    plt.plot(threshold, fpr, label='false positive rate')
    plt.plot(threshold, fnr, label='false negative rate')

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    plt.hlines(eer, min(pred), max(pred), linestyles='dashed')
    plt.legend()
    
    yticks = [eer]
    for y in [0., 0.2, 0.4, 0.6, 0.8, 1.0]:
        if abs(y - eer) >= 0.05:
            yticks.append(y)
    plt.yticks(yticks)
    plt.savefig(f"ema/roc/{key}.png")
    return eer_threshold, eer


def choice(word):
    while True:
        false_word = random.choice(ALL_WORDS).upper()
        if false_word != word:
            return false_word


class Node:
    def __init__(self, id: int, label: int, boost: float = 1.0,
                 sum_boost: float = 1.0, threshold: float = 0.2) -> None:
        self.id = id    # -1: root node
        self.next_nodes: tp.Dict[int, "Node"] = {}
        self.boost = boost
        self.sum_boost = sum_boost
        self.threshold = threshold
        self.is_end = False
        self.label = label
    
    # def append(self, node: "Node") -> None:
    #     self.next_nodes[node.id] = node
    
    def add_next_node_by_id(self, id: int, label: int, boost: float, threshold: float) -> "Node":
        """If next node with the given id already exists, return that node.
        Otherwise, create a new node, append, and return."""
        if id in self.next_nodes:
            return self.next_nodes[id]
        next_node = Node(id, label, boost, self.sum_boost + boost, threshold)
        self.next_nodes[id] = next_node
        return next_node


def print_tree(
    tree: Node,
    prefix: str = "",
    sp: tp.Optional[spm.SentencePieceProcessor] = None
) -> None:
    if sp is None:
        print(f"{tree.id:<3d} ", end="")
        prefix = f"{prefix}{' ' * (3 + 1)}"
    else:
        piece = "<root>"
        if tree.id >= 0:
            piece = f"{sp.IdToPiece(tree.id)}({tree.id})"
        print(f"{piece} ", end="")
        prefix = f"{prefix}{' ' * (len(piece) + 1)}"
    for idx, node in enumerate(tree.next_nodes.values(), start=1):
        if idx == 1:
            if len(tree.next_nodes) == 1:
                print("── ", end="")
                subgraph_prefix = f"{prefix}   "
            else:
                print("┬─ ", end="")
                subgraph_prefix = f"{prefix}│  "
        elif idx == len(tree.next_nodes):
            print(f"{prefix}└─ ", end="")
            subgraph_prefix = f"{prefix}   "
        else:
            print(f"{prefix}├─ ", end="")
            subgraph_prefix = f"{prefix}│  "
        print_tree(node, subgraph_prefix, sp)
    
    if len(tree.next_nodes) == 0:
        print(f"\n", end="")


def generate_phrase_tree(
    phrases: tp.List[str],
    sp: spm.SentencePieceProcessor,
    boost: float,
    threshold: float,
    blank_id: int = 0,
    eos_id: int = 500,
) -> Node:
    id_lists = sp.Encode([phrase.upper() for phrase in phrases], out_type=int)
    start_node = Node(-1, -1, 0.0, 0.0, 0.0)
    for label, id_list in enumerate(id_lists):
        tree = start_node
        for next_id in id_list:
            tree = tree.add_next_node_by_id(next_id, label=label, boost=boost,
                                            threshold=threshold)
        tree.is_end = True
    return start_node


class Item:
    def __init__(
        self,
        context_size: int,
        root_node: Node,
        sp: spm.SentencePieceProcessor
    ):
        self.context_size = context_size
        self.sum_logp = 0.  # sum of log_p along all the tokens, including blank
        self.node = root_node   # initialized to root node
        self._root_node = root_node
        self.ys: tp.List[int] = [-1] * (context_size - 1) + [sp.PieceToId("<blk>")]
        self.sum_p_ys = 0.  # sum(probs of tokens except blank)
        self.sp = sp        # for debugging
    
    def forward_one_step(self, token: int, logp: float):
        boost = 0.0
        if token in self.node.next_nodes:
            self.node = self.node.next_nodes[token]
            self.ys.append(token)
            self.sum_p_ys += math.exp(logp)
            boost = self.node.boost
            if self.node.is_end:
                boost += self.node.boost * len(self.ys[2:])
        elif token != self.sp.PieceToId("<blk>") and token != self.sp.PieceToId("<unk>"):
            # reset
            boost = -self.node.sum_boost
            self.node = self._root_node
            self.ys = self.ys[:self.context_size]
            self.sum_p_ys = 0.
        self.sum_logp += logp + boost
        return boost
    
    def copy(self) -> "Item":
        item = Item(self.context_size, self._root_node, self.sp)
        item.sum_logp = self.sum_logp
        item.node = self.node
        item.sum_p_ys = self.sum_p_ys
        item.ys = copy.deepcopy(self.ys)
        return item
    
    def __repr__(self):
        node = self.node
        y = "".join([self.sp.IdToPiece(x) for x in self.ys[self.context_size:]])
        return f"{y}({self.sum_logp:.2e})"


class Table:
    def __init__(self, item: tp.Optional[Item] = None):
        self.table: tp.List[Item] = []
        if item is not None:
            self.table.append(item)
    
    def __len__(self):
        return len(self.table)
    
    def __repr__(self):
        return f"Table {repr(self.table)}"
    
    def append(self, item: Item):
        found = False
        for x in self.table:
            if x.node == item.node:
                found = True
                x.sum_p_ys = max(x.sum_p_ys, item.sum_p_ys)
                
                if MERGE == "MAX":
                    x.sum_logp = max(x.sum_logp, item.sum_logp)
                elif MERGE == "LOG_ADD":
                    if x.sum_logp < item.sum_logp:
                        p1, p2 = item.sum_logp, x.sum_logp
                    else:
                        p1, p2 = x.sum_logp, item.sum_logp
                    x.sum_logp = p1 + math.log1p(math.exp(p2-p1))
                break
        if not found:
            self.table.append(item)
    
    def max(self):
        if not self.table:
            raise RuntimeError("Trying to call max() from an empty table")
        idx, max_logp = 0, self.table[0].sum_logp
        for i in range(1, len(self.table)):
            if max_logp < self.table[i].sum_logp:
                max_logp = self.table[i].sum_logp
                idx = i
        return self.table[idx]


def calculate_total_logp(model, enc_out, blank_id, sp, label) -> float:
    y = sp.Encode(list(label), out_type=int)
    y = k2.RaggedTensor(y).to(enc_out.device)
    sos_y = add_sos(y, sos_id=blank_id)

    # sos_y_padded: [B, S + 1], start with SOS.
    sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

    # decoder_out: [B, S + 1, decoder_dim]
    decoder_out = model.decoder(sos_y_padded)
    decoder_out = model.joiner.decoder_proj(decoder_out)    # [B, S+1, C]
    
    logits = model.joiner(
        enc_out.unsqueeze(2),
        decoder_out.unsqueeze(1),
        project_input=False
    )       # [B, T, S+1, C]
    y_padded = y.pad(mode="constant", padding_value=0)
    y_padded = y_padded.to(torch.int64) # [B, S]

    i = 0   # assume a batch size of 1
    S = len(y[i])
    nll = k2.rnnt_loss(
        logits[i:i+1, :, :S+1, :],
        y_padded[i:i+1, :S],
        blank_id,
        reduction="none"
    )
    return -nll.item()


def beam_search_in_vocabulary(
    model, specs, T_spec, blank_id, eos_id,
    tree: Node, sp: spm.SentencePieceProcessor, beam_size=2, DEBUG=False,
):
    '''Args:
        model: torch.nn.Module composed of an encoder/decoder/joiner 
        specs: spectrogram input tensor [Batch, Time, Freq]
        T_spec: int64 tensor of size [B] where T_spec[b] = length of specs[b]
        blank_id: int
        tree: vocabulary tree
    '''
    unk_id = sp.PieceToId("<unk>")
    batch_size = specs.size(0)
    encoder_out, _, _ = model.encoder(x=specs, x_lens=T_spec)
    encoder_out = model.joiner.encoder_proj(encoder_out)    # [B, T, C]
    context_size: int = model.decoder.context_size
    logp_list, seq_list = [], []
    correct_num = 0
    candidates: tp.List[Table] = []
    for batch in range(batch_size):
        table = Table(Item(context_size, tree, sp))
        candidates.append(table)
    out_label = -torch.ones(batch_size, dtype=torch.int16)
    mean_probs = torch.zeros(batch_size, dtype=torch.float32)
    for time in range(encoder_out.size(1)):
        enc_out = encoder_out[:, time:time+1, :]                # [B, 1, C]
        dec_in = []
        num_per_batch = []
        enc_out_idx = []
        logp_prev = []
        for batch in range(batch_size):
            _dec_in = []
            for item in candidates[batch].table:
                _dec_in.append(item.ys[-context_size:])   
                logp_prev.append(item.sum_logp)
            # _dec_in: [num, context_size] where 1 <= num <= beam_size
            num = len(_dec_in)
            num_per_batch.append(num)
            dec_in.extend(_dec_in)
            enc_out_idx.extend([batch] * num)
        enc_out_idx = torch.tensor(enc_out_idx, dtype=torch.int64, device=specs.device)
        logp_prev = torch.tensor(logp_prev, dtype=torch.float32, device=specs.device)   #[total_size]
        enc_out = torch.index_select(enc_out, dim=0, index=enc_out_idx) # [total_size, 1, C]
        dec_in = torch.tensor(dec_in, dtype=torch.int64, device=specs.device)
        dec_out = model.decoder(dec_in, need_pad=False)    # [total_size, 1, C]
        dec_out = model.joiner.decoder_proj(dec_out)
        logits = model.joiner(enc_out, dec_out, project_input=False)
        logp = logits.log_softmax(dim=-1).squeeze(1)            # [total_size, C]
        logp_total = logp + logp_prev.unsqueeze(1)

        VOCAB_SIZE = logp.size(1)
        start = 0
        for batch in range(batch_size):
            table = Table()
            end = start + num_per_batch[batch]
            # logp_b = []
            # for item in candidates[batch].table:
            #     for
            logp_b = logp_total[start:end].view(-1)    # [num*VOCAB_SIZE]
            logp_b = torch.topk(logp_b, beam_size, dim=0)
            num_beam = 0
            for idx in logp_b.indices:
                idx = idx.item()
                n = idx // VOCAB_SIZE
                token = idx % VOCAB_SIZE
                item = candidates[batch].table[n].copy()
                if DEBUG:
                    print(f"{num_beam}(", end="")
                    for y in item.ys[2:]:
                        print(f"{sp.IdToPiece(y)},", end="")
                boost = item.forward_one_step(token, logp[start+n, token].item())
                if DEBUG:
                    print(f"{sp.IdToPiece(token)}){logp_prev[n]:.2f},{logp[start+n, token]:.2f},{boost:.2f}|", end="")
                table.append(item)
                num_beam += 1
            max_candidate = table.max()
            if max_candidate.node.is_end and out_label[batch] < 0:
                p_mean = max_candidate.sum_p_ys / len(max_candidate.ys[2:])
                if DEBUG:
                    print(
                        f"matched: {sp.DecodeIds(max_candidate.ys[2:])} / {max_candidate.node.label}"
                        f" ({p_mean:.2f} / {max_candidate.node.threshold})"
                    )
                if p_mean >= max_candidate.node.threshold:
                    out_label[batch] = max_candidate.node.label
                    table = Table(Item(context_size, tree, sp))
                else:
                    if DEBUG:
                        print("Less Than Threshold")
                        # breakpoint()
                    out_label[batch] = -2
                mean_probs[batch] = p_mean
            start = end
            candidates[batch] = table
    return out_label, mean_probs


@torch.no_grad()
def beam():
    parser = get_parser()
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Set True for debugging",
    )
    parser.add_argument(
        "--boost",
        type=float,
        default=1.0,
        help="Boost logprob",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Threshold for matching",
    )
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    DEBUG = args.debug

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    print(f"beam size: {args.beam_size}, boost: {args.boost}, threshold: {args.threshold}, Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.LoadFromFile(params.bpe_model)
    blank_id = sp.PieceToId("<blk>")
    eos_id = sp.PieceToId("<sos/eos>")
    
    fbank = CustomFbank(CustomFbankConfig(num_mel_bins=80)).to(device)
    
    tree = generate_phrase_tree(
        # ["bay me one", "bay me two", "bay you", "bay him", "fuck", "fu"],
        # ["forever", "lovely child", "light up", "lovely dog"],
        WORDS,
        sp,
        args.boost,
        args.threshold,
        blank_id,
        eos_id,
    )
    print_tree(tree, sp=sp)
    
    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.PieceToId("<blk>")
    params.unk_id = sp.PieceToId("<unk>")
    params.vocab_size = sp.GetPieceSize()
    
    model = get_transducer_model(params)
    get_model(params, model, device, args, sp)
    breakpoint()
    
    dataset = SpeechCommandsDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    
    candidates, probs = [], []
    correct_total, num_total = 0, 0
    false_positive, false_negative = 0, 0
    st = time.time()
    len_dl = len(dataloader)
    for step, data in enumerate(dataloader, start=1):
        # if DEBUG and step > 1:
        #     break
        batch, label, file = data
        batch = batch.to(device)            # [B, T_wav]
        specs = fbank.extract_batch(batch)  # [B, T_spec, 80]
        label = [WORDS.index(l) if l in WORDS else -1 for l in label]
        label = torch.tensor(label, dtype=torch.int16)
        
        # for debugging, select only 1 batch
        # if DEBUG:
        #     idx = torch.nonzero(label != -1)[0, 0].item()
        #     specs = specs[idx:idx+1]
        #     label = label[idx:idx+1]
        #     print(f"Label: {WORDS[label.item()]} / {label.item()}")

        B, T, _ = specs.shape
        T_spec = torch.tensor([T], dtype=torch.int64, device=device).repeat(B)
        
        out_label_raw, mean_probs = beam_search_in_vocabulary(
            model, specs, T_spec, blank_id, eos_id, tree, sp,
            beam_size=args.beam_size, DEBUG=args.debug)
        out_label = torch.where(out_label_raw < 0, -1, out_label_raw)
        correct_total += (torch.where(out_label<0, -1, out_label) == label).sum().item()
        false_positive += torch.logical_and(out_label != label, out_label >= 0).sum().item()
        false_negative += torch.logical_and(out_label != label, out_label < 0).sum().item()
        num_total += batch.size(0)
        et = time.time()
        print(
            f"\r[{step}/{len_dl}] "
            f"Acc: {correct_total} / {num_total} ({correct_total / num_total * 100:.1f}%)  "
            f"FalsePos: {false_positive / num_total * 100:.1f}%  "
            f"FalseNeg: {false_negative / num_total * 100:.1f}%  "
            f"Time: {(et-st)/step:.1f} / {et-st:.1f} / {(et - st) / step * (len_dl - step):.1f} sec",
            end="",
            flush=True
        )
        # breakpoint()
    print("")


if __name__=="__main__":
    beam()
    # main()
