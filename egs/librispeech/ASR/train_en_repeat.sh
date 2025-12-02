model="ema_repeat"
exp="en/b4r10_ch960_actlimit1_wlimit0.3_prenorm"

CUDA_VISIBLE_DEVICES=0,1,2,3 ${model}/train.py \
    --world-size 4 \
    --num-epochs 200 \
    --start-epoch 1 \
    --exp-dir exp/$exp \
    --full-libri 1 \
    --max-duration 1800 \
    --master-port 54320 \
    --use-fp16 True \
    --encoder-norm SyncBatchNorm \
    --channels 256 \
    --channels-expansion 960 \
    --num-blocks 4 \
    --repeat 10 \
    --kernel-size 8 \
    --shared-norm False \
    --block-pre-norm True \
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
    --data-libri-train True \
    --data-libri-dev-clean True \
    --data-libri-dev-other True \
    --data-ksponspeech-train False \
    --data-ksponspeech-dev False \
    --data-zeroth-train False \
    --data-zeroth-test False \
    --data-command-nor-train False \
    --data-freetalk-nor-train False \
    --on-the-fly-feats False \
    --bpe-model /home/shahn/Documents/icefall/egs/librispeech/ASR/data/en/lang_bpe_500/bpe.model \
    --manifest-dir data/en/fbank \
    --cutset-text text \
    --num-workers 2 \
    --simple-loss-scale 0.5 \
    --optimizer-name Eve \
    --weight-decay 0.001 \
    --weight-limit 0.3 \
    --min-utt-duration 1.0 \
    --max-utt-duration 20.0
    # --subsampling-factor 4 \
    # --whitener False \
    # --weight-norm True \
    # --dec-residual False \
    # --subsampling-factor 4 \
    # --encoder-mean -6.883708542616444 \
    # --encoder-std 4.677519844325524 \
    # --iir-r-max 0.95 \
    # --iir-lr-ratio 1.0 \
    # --iir-bidirectional False \
    # --iir-filtfilt True \
    # --iir-unpad-delay False
