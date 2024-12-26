model="ema"
exp="korean/vocab500_ipa_do_sl0.5"
avg=60
CUDA_VISIBLE_DEVICES=1 ${model}/decode_korean_cer.py \
    --epoch 600 \
    --iter 0 \
    --avg $avg \
    --exp-dir ${model}/$exp \
    --max-duration 200 \
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
    --zero-init-residual True \
    --se-gate tanh \
    --ema-gamma 0.93 \
    --chunksize 16 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --bpe-model /home/shahn/Documents/icefall_github/egs/ksponspeech/ASR/data/lang_bpe_500_ipa_filtered/bpe.model \
    --cutset-text custom.ipa_filtered \
    --on-the-fly-feats True
    # --subsampling-factor 4 \
    # --whitener False \
    # --weight-norm True \
    # --whitener True \
    # --encoder-mean -6.883708542616444 \
    # --encoder-std 4.677519844325524