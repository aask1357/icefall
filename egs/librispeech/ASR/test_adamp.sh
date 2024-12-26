model="ema_cba"
exp="cba_adamp_wn_nogwd_wd0"
avg=8
CUDA_VISIBLE_DEVICES=0 ${model}/decode.py \
    --epoch 200 \
    --iter 0 \
    --avg $avg \
    --exp-dir ${model}/$exp \
    --max-duration 2000 \
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
    --scaled-conv False \
    --act-bal True \
    --zero-init-residual True \
    --se-gate tanh \
    --ema-gamma 0.93 \
    --chunksize 16 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --weight-norm True