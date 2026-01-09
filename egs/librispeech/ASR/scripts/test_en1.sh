model="ema_limit"
exp="en/default_2"
avg=1
CUDA_VISIBLE_DEVICES=4 ${model}/decode.py \
    --epoch 200 \
    --iter 0 \
    --avg $avg \
    --exp-dir exp/$exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --use-averaged-model False \
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
    --se-gate tanh \
    --ema-gamma 0.97 \
    --ema-r-activation sigmoid \
    --chunksize 8 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --act-bal True \
    --whitener True \
    --joiner-activation Tanh \
    --decoder-activation ReLU \
    --bpe-model data/en/lang_bpe_500/bpe.model \
    --manifest-dir data/en/fbank \
    --cutset-text text \
    --on-the-fly-feats False