model="ema"
exp=exp/ko/ipa_do_sl0.5
avg=32
CUDA_VISIBLE_DEVICES=4 ${model}/decode_korean_cer.py \
    --epoch 97 \
    --iter 0 \
    --avg $avg \
    --exp-dir ${exp} \
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
    --zero-init-residual True \
    --se-gate tanh \
    --ema-gamma 0.93 \
    --chunksize 16 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --data-ksponspeech-train True \
    --data-ksponspeech-eval-clean True \
    --data-ksponspeech-eval-other True \
    --data-zeroth-test True \
    --on-the-fly-feats True \
    --bpe-model data/ko/lang_bpe_500_ipa_filtered/bpe.model \
    --manifest-dir data/ko/fbank \
    --cutset-text custom.ipa_filtered
