model="ema_norm"
exp="en/norm_chunk8_g.97_dur2400"
avg=64
CUDA_VISIBLE_DEVICES=0 ${model}/decode.py \
    --epoch 191 \
    --iter 0 \
    --avg $avg \
    --exp-dir exp/$exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --use-averaged-model True \
    --encoder-norm BatchNorm \
    --channels 256 \
    --channels-expansion 1024 \
    --dilations-version 11 \
    --kernel-size 8 \
    --encoder-activation ReLU \
    --encoder-se-activation ReLU \
    --skip residual \
    --se-gate tanh \
    --ema-gamma 0.97 \
    --chunksize 8 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --clamp-method None \
    --conv-pre-norm False \
    --conv-post-norm False \
    --bpe-model data/en/lang_bpe_500/bpe.model \
    --manifest-dir data/en/fbank \
    --cutset-text text \
    --on-the-fly-feats False \
    --update-bn True