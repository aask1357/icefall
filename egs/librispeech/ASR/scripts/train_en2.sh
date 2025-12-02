model="ema_norm"
exp="en/norm_bypass1_chunk8_ipa3"

CUDA_VISIBLE_DEVICES=4,5 ${model}/train.py \
    --world-size 2 \
    --num-epochs 200 \
    --start-epoch 1 \
    --exp-dir exp/$exp \
    --full-libri 1 \
    --max-duration 4000 \
    --master-port 54324 \
    --use-fp16 True \
    --encoder-norm SyncBatchNorm \
    --channels 256 \
    --channels-expansion 1024 \
    --dilations-version 11 \
    --kernel-size 8 \
    --encoder-activation ReLU \
    --encoder-se-activation ReLU \
    --skip bypass-oneinit \
    --se-gate tanh \
    --ema-gamma 0.93 \
    --chunksize 8 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --clamp-method None \
    --conv-pre-norm True \
    --conv-post-norm False \
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
    # --weight-limit 0.3 \
