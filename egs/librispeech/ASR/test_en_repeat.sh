model="ema_repeat"
exp="en/b4r10_ch960_actlimit1_wlimit0.3"
avg=1
CUDA_VISIBLE_DEVICES=4 ${model}/decode.py \
    --epoch 47 \
    --iter 0 \
    --avg $avg \
    --exp-dir exp/$exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --use-averaged-model True \
    --update-bn True \
    --encoder-norm BatchNorm \
    --channels 256 \
    --channels-expansion 960 \
    --num-blocks 4 \
    --repeat 10 \
    --kernel-size 8 \
    --shared-norm False \
    --encoder-activation PReLU \
    --encoder-se-activation PReLU \
    --se-gate tanh \
    --ema-gamma 0.93 \
    --chunksize 8 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --zero-init-residual True \
    --act-bal False \
    --whitener False \
    --bpe-model /home/shahn/Documents/icefall/egs/librispeech/ASR/data/en/lang_bpe_500/bpe.model \
    --manifest-dir data/en/fbank \
    --cutset-text text \
    --on-the-fly-feats False
    # --subsampling-factor 4 \
    # --scaled-conv True \
    # --act-bal True \
    # --whitener False \
    # --weight-norm True \
    # --whitener True \
    # --encoder-mean -6.883708542616444 \
    # --encoder-std 4.677519844325524