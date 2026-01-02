model="ema_limit_customquant"
exp="en/scratch/a8w4_omni_emamax_lrlin5e-3_nowlimit_actbal"
avg=1
CUDA_VISIBLE_DEVICES=4 ${model}/decode.py \
    --epoch 100 \
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
    --skip residual-zeroinit \
    --se-gate tanh \
    --ema-gamma 0.97 \
    --chunksize 8 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --act-bal True \
    --eps 1.0e-5 \
    --n-bits-act 8 \
    --n-bits-weight 4 \
    --quantizer-mode omni_emamax \
    --quantizer-gamma 0.95 \
    --bpe-model data/en/lang_bpe_500/bpe.model \
    --manifest-dir data/en/fbank \
    --cutset-text text \
    --on-the-fly-feats False