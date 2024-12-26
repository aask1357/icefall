model="ema_noscale"
exp="korean/vocab500_ipa3_l11_customfbank_adamp_wd1e-2"

CUDA_VISIBLE_DEVICES=0,1,2,3 ${model}/train.py \
    --world-size 4 \
    --num-epochs 200 \
    --start-epoch 1 \
    --exp-dir ${model}/$exp \
    --full-libri 1 \
    --max-duration 2432 \
    --master-port 54321 \
    --use-fp16 True \
    --encoder-norm SyncBatchNorm \
    --channels 256 \
    --channels-expansion 1024 \
    --dilations-version 11 \
    --kernel-size 8 \
    --encoder-activation ReLU \
    --encoder-se-activation ReLU \
    --act-bal False \
    --whitener False \
    --zero-init-residual True \
    --se-gate tanh \
    --ema-gamma 0.93 \
    --chunksize 16 \
    --scale-limit 1.0 \
    --weight-norm True \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --optimizer-name AdamP \
    --weight-decay 0.01 \
    --weight-decay-projection 1.0e-6 \
    --data-libri-train False \
    --data-libri-dev-clean False \
    --data-libri-dev-other False \
    --data-ksponspeech-train True \
    --data-ksponspeech-dev True \
    --data-zeroth-train False \
    --data-zeroth-test True \
    --data-command-nor-train False \
    --data-freetalk-nor-train True \
    --on-the-fly-feats True \
    --bpe-model /home/shahn/Documents/icefall_github/egs/ksponspeech/ASR/data/lang_bpe_500_ipa_max3/bpe.model \
    --cutset-text custom.ipa_filtered \
    --num-workers 24 \
    --simple-loss-scale 0.5
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
