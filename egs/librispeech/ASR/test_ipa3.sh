model="ema_noscale"
exp="korean/vocab500_ipa3_l11_customfbank_adamp_wd1e-2"
avg=1
CUDA_VISIBLE_DEVICES=0 ${model}/decode_korean_cer.py \
    --epoch 200 \
    --iter 0 \
    --avg $avg \
    --exp-dir ${model}/$exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --use-averaged-model False \
    --update-bn True \
    --encoder-norm BatchNorm \
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
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --bpe-model /home/shahn/Documents/icefall_github/egs/ksponspeech/ASR/data/lang_bpe_500_ipa_max3/bpe.model \
    --cutset-text custom.ipa_filtered \
    --on-the-fly-feats True
    # --scaled-conv True \
    # --subsampling-factor 4 \
    # --whitener False \
    # --weight-norm True \
    # --whitener True \
    # --encoder-mean -6.883708542616444 \
    # --encoder-std 4.677519844325524