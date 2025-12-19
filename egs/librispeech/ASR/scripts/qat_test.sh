model="ema_limit"
exp="en/qat/a8w4"
avg=1


CUDA_VISIBLE_DEVICES=0 python ${model}/qat_decode.py \
    --epoch 14 \
    --iter 0 \
    --avg $avg \
    --exp-dir exp/$exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --use-averaged-model False \
    --encoder-norm BatchNorm \
    --channels 256 \
    --channels-expansion 1024 \
    --dilations-version 11 \
    --kernel-size 8 \
    --encoder-activation ReLU \
    --encoder-se-activation ReLU \
    --zero-init-residual True \
    --se-gate tanh \
    --ema-gamma 0.97 \
    --chunksize 8 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --act-bal False \
    --whitener False \
    --n-bits-act 8 \
    --n-bits-weight 4 \
    --data-libri-train True \
    --data-libri-test-clean True \
    --data-libri-test-other True \
    --enable-musan True \
    --enable-spec-aug True \
    --bpe-model data/en/lang_bpe_500/bpe.model \
    --manifest-dir data/en/fbank \
    --cutset-text text
