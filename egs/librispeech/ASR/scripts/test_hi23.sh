model="ema_q_lss"
exp="en/scratch/a8w4_lss_glr0.1"
avg=64
CUDA_VISIBLE_DEVICES=6 ${model}/decode.py \
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
    --skip residual-zeroinit \
    --se-gate tanh \
    --ema-gamma 0.97 \
    --chunksize 8 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --act-bal True \
    --whitener True \
    --eps 1.0e-2 \
    --n-bits-act 8 \
    --n-bits-weight 4 \
    --weight-quantizer-mode scale \
    --quantizer-gamma-lr-ratio 0.1 \
    --bpe-model data/en/lang_bpe_500/bpe.model \
    --manifest-dir data/en/fbank \
    --cutset-text text \
    --on-the-fly-feats False