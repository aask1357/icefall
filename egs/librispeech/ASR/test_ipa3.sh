model="ema_cba_new_2"
exp="korean/${model}_nonormprepost_eve_wd1e-3"
avg=32
CUDA_VISIBLE_DEVICES=2 ${model}/decode_cer.py \
    --epoch 120 \
    --iter 0 \
    --avg $avg \
    --exp-dir exp/$exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --use-averaged-model True \
    --update-bn True \
    --encoder-norm BatchNorm \
    --channels 256 \
    --channels-expansion 1024 \
    --dilations-version 11 \
    --kernel-size 8 \
    --encoder-activation ReLU \
    --encoder-se-activation ReLU \
    --act-bal True \
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
    --logit-no-bias False \
    --data-libri-train False \
    --data-ksponspeech-train True \
    --data-freetalk-nor-train True \
    --bpe-model /home/shahn/Documents/icefall_github/egs/ksponspeech/ASR/data/lang_bpe_500_ipa_max3/bpe.model \
    --cutset-text custom.ipa_filtered \
    --on-the-fly-feats True
    # --scaled-conv True
    # --subsampling-factor 4 \
    # --whitener False \
    # --weight-norm True \
    # --whitener True \
    # --encoder-mean -6.883708542616444 \
    # --encoder-std 4.677519844325524