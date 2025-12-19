model="ema_limit"
exp="en/l11_chunk8_g.97_wlimit.3_bypass0init"

CUDA_VISIBLE_DEVICES=4,5,6,7 ${model}/ft.py \
    --world-size 4 \
    --num-epochs 200 \
    --start-epoch 98 \
    --ft-path exp/ko/ipa_do_sl0.5/epoch-97-avg-32.pt \
    --exp-dir exp/$exp \
    --full-libri 1 \
    --max-duration 2400 \
    --master-port 54324 \
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
    --ema-gamma 0.93 \
    --chunksize 16 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --data-libri-train False \
    --data-libri-dev-clean False \
    --data-libri-dev-other False \
    --data-ksponspeech-enhanced-train True \
    --data-ksponspeech-enhanced-dev True \
    --data-zeroth-test True \
    --enable-musan False \
    --enable-spec-aug False \
    --manifest-dir data/ko/fbank \
    --bpe-model data/ko/lang_bpe_500_ipa_filtered/bpe.model \
    --cutset-text custom.ipa_filtered \
    --num-workers 2 \
    --simple-loss-scale 0.5
    # --subsampling-factor 4
    # --optimizer AdamP \
    # --weight-decay 0.01 \
    # --weight-decay-projection 0.00001 \
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
