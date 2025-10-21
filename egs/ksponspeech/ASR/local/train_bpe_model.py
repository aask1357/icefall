#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# You can install sentencepiece via:
#
#  pip install sentencepiece
#
# Due to an issue reported in
# https://github.com/google/sentencepiece/pull/642#issuecomment-857972030
#
# Please install a version >=0.1.96

import argparse
import shutil
from pathlib import Path
from typing import Dict
from icefall.utils import str2bool

import sentencepiece as spm


def get_args():
    # ksponspeech -> normalization=identity add_dummy_prefix=False
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        The generated bpe.model is saved to this directory.
        """,
    )

    parser.add_argument(
        "--transcript",
        type=str,
        default='/home/shahn/Documents/icefall_github/egs/ksponspeech/ASR/data/lang_bpe_500_ipa_filtered/transcript_words_ipa.txt',
        help="Training transcript.",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Vocabulary size for BPE training",
    )
    
    parser.add_argument(
        "--normalization",
        type=str,
        # default="nmt_nfkc", # nmt_nfkc | nfkc | nmt_nfkc_cf | nfkc_cf | identity
        default="identity",
        help="Normalization to be used for training BPE model",
    )
    
    parser.add_argument(
        "--add-dummy-prefix",
        type=str2bool,
        # default=True,
        default=False,
        help="Whether to add dummy whitespace at the beginning of text",
    )
    
    parser.add_argument(
        "--max-sentencepiece-length",
        type=int,
        # default=16,
        default=3,
        help="Maximum length of sentence piece",
    )
    
    parser.add_argument(
        "--num-sub-iterations",
        type=int,
        # default=2,
        default=5,
        help="Number of number of EM sub-iterations",
    )

    return parser.parse_args()


def generate_tokens(lang_dir: Path):
    """
    Generate the tokens.txt from a bpe model.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(lang_dir / "bpe.model"))
    token2id: Dict[str, int] = {sp.id_to_piece(i): i for i in range(sp.vocab_size())}
    with open(lang_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for sym, i in token2id.items():
            f.write(f"{sym} {i}\n")


def main():
    args = get_args()
    vocab_size = args.vocab_size
    lang_dir = Path(args.lang_dir)

    model_type = "unigram"

    model_prefix = f"{lang_dir}/{model_type}_{vocab_size}"
    train_text = args.transcript
    character_coverage = 1.0
    input_sentence_size = 100000000

    user_defined_symbols = [
        "<blk>",
        "<sos/eos>",
        # "tɕ",   # 초성 ㅈ
        # "dʑ",   # 초성 ㅈ
        # "tʃh",  # 초성 ㅊ
        # "kh",   # 초성 ㅋ
        # "th",   # 초성 ㅌ
        # "ph",   # 초성 ㅍ
    ]
    unk_id = 2
    # Note: unk_id is fixed to 2.
    # If you change it, you should also change other
    # places that are using it.

    model_file = Path(model_prefix + ".model")
    if not model_file.is_file():
        spm.SentencePieceTrainer.train(
            input=train_text,
            vocab_size=vocab_size,
            model_type=model_type,
            model_prefix=model_prefix,
            input_sentence_size=input_sentence_size,
            character_coverage=character_coverage,
            user_defined_symbols=user_defined_symbols,
            normalization_rule_name=args.normalization,
            add_dummy_prefix=args.add_dummy_prefix,
            num_sub_iterations=args.num_sub_iterations,
            max_sentencepiece_length=args.max_sentencepiece_length,
            unk_id=unk_id,
            bos_id=-1,
            eos_id=-1,
        )
    else:
        print(f"{model_file} exists - skipping")
        return

    shutil.copyfile(model_file, f"{lang_dir}/bpe.model")

    generate_tokens(lang_dir)


if __name__ == "__main__":
    main()
