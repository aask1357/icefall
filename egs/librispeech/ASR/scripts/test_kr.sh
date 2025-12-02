model="ema"
exp="ko/kspon_noisy_ft_from_avg"
avg=12
start_weight=12
CUDA_VISIBLE_DEVICES=6 ${model}/decode_korean_cer.py \
    --epoch 108 \
    --iter 0 \
    --avg $avg \
    --start-weight $start_weight \
    --exp-dir exp/$exp \
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
    --zero-init-residual True \
    --se-gate tanh \
    --ema-gamma 0.93 \
    --chunksize 16 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --data-libri-train False \
    --data-ksponspeech-enhanced-train True \
    --data-ksponspeech-eval-clean True \
    --data-ksponspeech-eval-other True \
    --data-ksponspeech-enhanced-eval-clean True \
    --data-ksponspeech-enhanced-eval-other True \
    --enable-musan False \
    --enable-spec-aug True \
    --data-zeroth-test True \
    --on-the-fly-feats True \
    --bpe-model data/ko/lang_bpe_500_ipa_filtered/bpe.model \
    --manifest-dir data/ko/fbank \
    --cutset-text custom.ipa_filtered
