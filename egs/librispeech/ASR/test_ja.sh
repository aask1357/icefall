model="ema_limit"
exp="ja/${model}1_m_katakana"
avg=20
CUDA_VISIBLE_DEVICES=0 ${model}/decode_cer.py \
    --epoch 40 \
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
    --zero-init-residual True \
    --se-gate tanh \
    --ema-gamma 0.93 \
    --chunksize 16 \
    --encoder-dim 512 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --encoder-dropout 0.075 \
    --data-reazonspeech-medium-train True \
    --data-reazonspeech-medium-test True \
    --data-jsut-basic5000-sudachi True \
    --bpe-model /home/shahn/Documents/icefall_github/egs/reazonspeech/ASR/data/lang_bpe_500_katakana/bpe.model \
    --cutset-text custom.katakana \
    --blank-penalty 4.0
    # --subsampling-factor 4 \
    # --scaled-conv True \
    # --act-bal True \
    # --whitener False \
    # --weight-norm True \
    # --whitener True \
    # --encoder-mean -6.883708542616444 \
    # --encoder-std 4.677519844325524