model="ema"
exp="kws_abs"
avg=64
CUDA_VISIBLE_DEVICES=3 ${model}/decode.py \
    --epoch 120 \
    --iter 0 \
    --avg $avg \
    --exp-dir ${model}/$exp \
    --max-duration 2400 \
    --decoding-method fast_beam_search \
    --use-averaged-model True \
    --update-bn True \
    --encoder-norm BatchNorm \
    --channels 384 \
    --channels-expansion 1536 \
    --dilations-version 22 \
    --kernel-size 8 \
    --encoder-activation ReLU \
    --encoder-se-activation ReLU \
    --scaled-conv True \
    --act-bal True \
    --conv1d-subsampling-version 2 \
    --zero-init-residual True \
    --se-gate tanh \
    --ema-gamma 0.93