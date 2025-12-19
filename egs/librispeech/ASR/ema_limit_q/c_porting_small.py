import matplotlib.pyplot as plt
from pathlib import Path
import math

import numpy as np
import torch
from torch.nn import functional as F
import librosa
import sentencepiece as spm
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import add_sos
from local.custom_fbank import CustomFbank, CustomFbankConfig

from asr_datamodule import AsrDataModule
from decode import get_parser
from train import get_params, get_transducer_model
from keyword_spotting import get_model
from update_bn import update_bn
from chip_fp16 import Q, q, FP16

print(f"fp16: {FP16}")
device = torch.device("cuda")
parser = get_parser()
AsrDataModule.add_arguments(parser)
args = parser.parse_args()
args.exp_dir = Path(args.exp_dir)

EXP = "ja/ema_limit1_m_katakana"

args.exp_dir = Path(f"/home/shahn/Documents/icefall/egs/librispeech/ASR/exp/{EXP}")
args.channels = 256
args.channels_expansion = 1024
args.encoder_dim = 512
args.decoder_dim = 256
args.joiner_dim = 256
args.dilations_version = 11
args.avg = 30
args.epoch = 48
args.use_averaged_model = True
args.update_bn = True
args.data_reazonspeech_medium_train = True
args.max_duration = 600
args.bpe_model = "/home/shahn/Documents/icefall_github/egs/reazonspeech/ASR/data/lang_bpe_500_katakana/bpe.model"
BLANK_PENALTY = 4.0

params = get_params()
params.update(vars(args))

sp = spm.SentencePieceProcessor()
sp.LoadFromFile(params.bpe_model)
params.blank_id = sp.PieceToId("<blk>")
params.unk_id = sp.PieceToId("<unk>")
params.vocab_size = sp.GetPieceSize()

model = get_transducer_model(params)
avg_path = str(args.exp_dir / f"epoch-{args.epoch}-avg-{args.avg}-bp-{BLANK_PENALTY:.1f}.pt")
get_model(params, model, device, args, sp, avg_path)
# total_params = 0
# for n, p in model.named_parameters():
#     if n.startswith("simple"):
#         continue
#     total_params += p.numel()
#     # print(n, p.shape)
# print(f"total params: {total_params}")
torch.save(
    {
        'model': model.state_dict(),
        'description': (
            f"exp=exp/{EXP}, epoch={args.epoch}, avg={args.avg}, use-averaged-model=True, "
            "se-gate=tanh, ema-gamma=0.93"
        ),
    }, avg_path
)

x, _ = librosa.load("/home/shahn/Datasets/jsut_ver1.1/basic5000/wav/BASIC5000_4936.wav", sr=16_000)

hop = 160
window = 400
dtype = torch.float16 if FP16 else torch.float32
fbank = CustomFbank(CustomFbankConfig(num_mel_bins=80, snip_edges=True)).to(device=device)

# pad x to multiple of 64*hop
x = torch.from_numpy(x).to(device)
chunksize = 64 * hop
padding = (x.size(0) + chunksize - 1) // chunksize * chunksize - x.size(0)
x = q(F.pad(x, (0, padding)))
with open("input.buf", "wb") as f:
    f.write(np.ascontiguousarray(x.cpu()).tobytes())
    print(f"input shape: {x.shape}")

# extract fbank using fbank module
x = F.pad(x, (400 - hop, 0))
spec_fbank = fbank.extract_batch(x.view(1, -1))

# extract fbank on-the-fly
spec_len = (x.size(0) - window + hop) // hop
y = x.as_strided(size=(spec_len, window), stride=(hop, 1))

# remove dc
row_means = q(torch.mean(y, dim=1).unsqueeze(1))
y = q(y - row_means)

# pre emphasis
y_prev = F.pad(y.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)  # size (67, 401)
y = q(y - 0.97 * y_prev[:, :-1])

# window + stft
y = q(F.linear(y, fbank.ft_weight, None))   # [T, 512]

# absolute magnitude
y = q(y.view(spec_len, 2, 256).abs().sum(dim=1))  # [T, 256]

# mel
y = q(F.linear(y, fbank.mel_fbank))  # [T, 80]

# logmel
y = y.log().mul(2).clamp_min(-15.9453).unsqueeze(0)  # [B, T, C]

y = (y + 6.375938187300722) * 0.22963919954179052
with open("output.buf", "wb") as f:
    f.write(np.ascontiguousarray(y.cpu().numpy()).tobytes())

cache_file = open("cache.buf", "wb")
model.encoder.remove_weight_reparameterizations()
# Encoder
with torch.no_grad():
    z = torch.zeros(400 - hop + hop*4, device=device)
    z = fbank.extract_batch(z.view(1, -1))[0].unsqueeze(0)  # [1, 4, 80]
    z = z.transpose(1, 2)                   # [1, 80, 4]
    z = (z + 6.375938187300722) * 0.22963919954179052
    cache = q(model.encoder.conv_pre.conv[0](z)).squeeze(0)
    print(f'	load("cache_conv_pre", {", ".join(list(str(s) for s in cache.shape))});')
    cache_file.write(np.ascontiguousarray(cache.cpu().numpy()).tobytes())
    for idx, block in enumerate(model.encoder.cnn):
        cache = q(block.initial_cache.clone()).squeeze(0)
        if idx == 0:
            print(f'		load(enc + "depthwise", {", ".join(list(str(s) for s in cache.shape))});')
        cache_file.write(np.ascontiguousarray(cache.cpu().numpy()).tobytes())
    z, *_ = model.encoder(x=y, x_lens=torch.tensor([y.size(1)], device=device, dtype=torch.int64))

with torch.no_grad():
    z2, *_ = model.encoder(x=y, x_lens=torch.tensor([y.size(1)], device=device, dtype=torch.int64))

with open("param.buf", 'wb') as f:
    def fwrite(param: torch.Tensor, name: str):
        f.write(np.ascontiguousarray(q(param.detach()).cpu().numpy()).tobytes())
        print(f'	load("{name}", {", ".join(list(str(s) for s in param.shape))});')
    def fwrite_enc(param, name):
        f.write(np.ascontiguousarray(q(param.detach()).cpu().numpy()).tobytes())
        if idx == 0:
            print(f'		load(enc + "{name}", {", ".join(list(str(s) for s in param.shape))});')
    fwrite(fbank.mel_fbank, "mel_filterbank")
    fwrite(model.encoder.conv_pre.conv[0].weight, "enc.conv_pre.0.weight")
    fwrite(model.encoder.conv_pre.conv[1].weight.squeeze(1).transpose(0, 1), "enc.conv_pre.1.weight")
    fwrite(model.encoder.conv_pre.conv[1].bias, "enc.conv_pre.1.bias")
    for idx, block in enumerate(model.encoder.cnn):
        CH = 1024
        N = 2
        assert CH % N == 0
        CH_PER_N = CH // N
        for i_b in range(N):
            fwrite_enc(block.pointwise1.weight[i_b*CH_PER_N:(i_b+1)*CH_PER_N, :, :], f"{i_b}.pointwise1.weight")
            fwrite_enc(block.pointwise1.bias[i_b*CH_PER_N:(i_b+1)*CH_PER_N], f"{i_b}.pointwise1.bias")
            fwrite_enc(block.depthwise.weight[i_b*CH_PER_N:(i_b+1)*CH_PER_N, :, :], f"{i_b}.depthwise.weight")
            fwrite_enc(block.depthwise.bias[i_b*CH_PER_N:(i_b+1)*CH_PER_N], f"{i_b}.depthwise.bias")
            fwrite_enc(block.pointwise2.weight[:, i_b*CH_PER_N:(i_b+1)*CH_PER_N, :], f"{i_b}.pointwise2.weight")
        fwrite_enc(block.pointwise2.bias, f"{i_b}.pointwise2.bias")
        for n, p in block.se.named_parameters():
            fwrite_enc(p, f"se.{n}")
        fwrite_enc(block.scale, f"scale")
    w1 = model.encoder.proj.weight.data[:, :256, 0]
    w2 = model.encoder.proj.weight.data[:, 256:, 0]
    w3 = model.joiner.encoder_proj.get_weight()
    b3 = model.joiner.encoder_proj.get_bias()
    w_x = w3 @ w1
    w_x_in = w3 @ w2
    fwrite(w_x, "enc.conv_post_block.weight")
    fwrite(w_x_in, "enc.conv_post_skip.weight")
    fwrite(b3, "enc.conv_post.bias")
    
    w = model.decoder.embedding.weight.data
    scale = model.decoder.embedding.scale.exp().data
    w = w * scale               # [VOCAB_SIZE, C]
    fwrite(w, "dec.embedding.weight")
    w1 = model.decoder.conv.get_weight().data
    fwrite(w1, "dec.conv1.weight")
    w2 = model.joiner.decoder_proj.get_weight().data
    fwrite(w2, "dec.conv2.weight")
    fwrite(model.joiner.decoder_proj.get_bias(), "dec.conv2.bias")
    fwrite(model.joiner.output_linear.get_weight(), "dec.conv3.weight")
    bias = model.joiner.output_linear.get_bias()
    bias[0] -= BLANK_PENALTY
    fwrite(bias, "dec.conv3.bias")

cache_file.close()