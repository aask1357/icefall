model="ema_limit"
exp="en/jsilu_bpemax4_dj512_chunk4_nooutskip"
avg=64
CUDA_VISIBLE_DEVICES=7 ${model}/decode.py \
    --epoch 200 \
    --iter 0 \
    --avg $avg \
    --exp-dir exp/$exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --use-averaged-model True \
    --update-bn False \
    --encoder-norm BatchNorm \
    --channels 256 \
    --channels-expansion 1024 \
    --dilations-version 11 \
    --kernel-size 8 \
    --encoder-activation ReLU \
    --encoder-se-activation ReLU \
    --skip residual \
    --zero-init-residual True \
    --out-skip False \
    --se-gate tanh \
    --ema-gamma 0.97 \
    --ema-r-activation sigmoid \
    --chunksize 4 \
    --encoder-dim 512 \
    --decoder-dim 512 \
    --joiner-dim 512 \
    --encoder-dropout 0.075 \
    --act-bal True \
    --whitener True \
    --joiner-activation SiLU \
    --bpe-model data/en/bpe_max4/bpe.model \
    --manifest-dir data/en/fbank \
    --cutset-text text \
    --on-the-fly-feats False