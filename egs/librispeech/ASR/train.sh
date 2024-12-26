model="ema"
exp="eng/l11_noemawd"

CUDA_VISIBLE_DEVICES=0,1,2,3 ${model}/train.py \
    --world-size 4 \
    --num-epochs 200 \
    --start-epoch 1 \
    --exp-dir ${model}/$exp \
    --full-libri 1 \
    --max-duration 2400 \
    --master-port 54321 \
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
    --data-libri-train True \
    --data-libri-dev-clean True \
    --data-libri-dev-other True \
    --data-ksponspeech-train False \
    --data-ksponspeech-dev False \
    --data-zeroth-train False \
    --data-zeroth-test False \
    --data-command-nor-train False \
    --data-freetalk-nor-train False \
    --on-the-fly-feats True \
    --bpe-model /home/shahn/Documents/icefall/egs/librispeech/ASR/data/lang_bpe_500/bpe.model \
    --cutset-text text \
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
