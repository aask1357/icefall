model="ema"
exp="small"

CUDA_VISIBLE_DEVICES=0,1,2 ${model}/train.py \
    --world-size 3 \
    --num-epochs 200 \
    --start-epoch 6 \
    --exp-dir ${model}/$exp \
    --full-libri 1 \
    --max-duration 1800 \
    --master-port 54321 \
    --use-fp16 True \
    --encoder-norm SyncBatchNorm \
    --channels 256 \
    --channels-expansion 1024 \
    --dilations-version 12 \
    --kernel-size 8 \
    --encoder-activation ReLU \
    --encoder-se-activation ReLU \
    --scaled-conv True \
    --act-bal True \
    --conv1d-subsampling-version 2 \
    --zero-init-residual True \
    --se-gate tanh \
    --ema-gamma 0.93 \
    --chunksize 16 \
    --encoder-dim 256 \
    --decoder-dim 256 \
    --joiner-dim 256
    # --whitener True \
    # --encoder-dropout 0.0 \
    # --simple-loss-scale 0.2
    # --subsampling-factor 4 \
    # --encoder-mean -6.883708542616444 \
    # --encoder-std 4.677519844325524 \
    # --iir-r-max 0.95 \
    # --iir-lr-ratio 1.0 \
    # --iir-bidirectional False \
    # --iir-filtfilt True \
    # --iir-unpad-delay False
