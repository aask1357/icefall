model="ema"
exp="c256_e1024_l12_d256"
avg=32
CUDA_VISIBLE_DEVICES=3 ${model}/decode.py \
    --epoch 200 \
    --iter 0 \
    --avg $avg \
    --exp-dir ${model}/$exp \
    --max-duration 2400 \
    --decoding-method fast_beam_search \
    --use-averaged-model True \
    --update-bn True \
    --encoder-norm BatchNorm \
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
    --joiner-dim 256