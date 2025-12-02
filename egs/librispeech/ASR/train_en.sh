model="ema_limit"
exp="en/l11_chunk8_g.97_wlimit.3"

CUDA_VISIBLE_DEVICES=0,1,2,3 ${model}/train.py \
    --world-size 4 \
    --num-epochs 200 \
    --start-epoch 1 \
    --exp-dir exp/$exp \
    --full-libri 1 \
    --max-duration 2400 \
    --master-port 54320 \
    --use-fp16 True \
    --encoder-norm SyncBatchNorm \
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
    --act-bal True \
    --whitener True \
    --scale-limit 2.0 \
    --weight-limit 0.3 \
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
    --bpe-model data/en/lang_bpe_500/bpe.model \
    --manifest-dir data/en/fbank \
    --cutset-text text \
    --num-workers 2 \
    --simple-loss-scale 0.5 \
    --optimizer-name Eve \
    --weight-decay 0.001 \
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
