model="ema_norm"
exp="en/norm_chunk8_g.97_dur2400"
avg=64
CUDA_VISIBLE_DEVICES=0 ${model}/decode.py \
    --epoch 190 \
    --iter 0 \
    --avg $avg \
    --exp-dir exp/$exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --use-averaged-model True \
    --update-bn True \
    --encoder-norm BatchNorm \
    --channels 256 \
    --channels-expansion 1024 \
    --dilations-version 11 \
    --kernel-size 8 \
    --encoder-activation ReLU \
    --encoder-se-activation ReLU \
    --se-gate tanh \
    --ema-gamma 0.93 \
    --chunksize 16 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --zero-init-residual True \
    --act-bal True \
    --whitener True \
    --bpe-model data/en/lang_bpe_500/bpe.model \
    --manifest-dir data/en/fbank \
    --cutset-text text \
    --on-the-fly-feats False