# https://github.com/google/sentencepiece/blob/master/doc/options.md
lang_dir=data/lang_bpe_500_ipa_max3
force=1
mkdir -p $lang_dir
if [ $force ] || [ ! -e $lang_dir/bpe.model ]; then
    python local/train_bpe_model.py --lang-dir $lang_dir
fi