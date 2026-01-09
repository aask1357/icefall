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
from icefall.utils import str2bool

import sentencepiece as spm


def get_args():
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
        help="Training transcript.",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Vocabulary size for BPE training",
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="bpe",
        help="bpe or unigram",
    )
    
    parser.add_argument(
        "--add-dummy-prefix",
        type=str2bool,
        default=True,
        help="Whether to add dummy whitespace at the beginning of text",
    )
    
    parser.add_argument(
        "--max-sentencepiece-length",
        type=int,
        default=4,
        help="Maximum length of sentence piece",
    )
    
    parser.add_argument(
        "--num-sub-iterations",
        type=int,
        default=2,
        help="Number of number of EM sub-iterations",
    )
    
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Whether to overwrite the existing model",
    )

    return parser.parse_args()


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)

    model_prefix = f"{lang_dir}/{args.model_type}_{args.vocab_size}"
    character_coverage = 1.0
    input_sentence_size = 100000000

    user_defined_symbols = ["<blk>", "<sos/eos>"]
    unk_id = len(user_defined_symbols)
    # Note: unk_id is fixed to 2.
    # If you change it, you should also change other
    # places that are using it.

    model_file = Path(model_prefix + ".model")
    if args.force or not model_file.is_file():
        spm.SentencePieceTrainer.train(
            input=args.transcript,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            model_prefix=model_prefix,
            input_sentence_size=input_sentence_size,
            character_coverage=character_coverage,
            user_defined_symbols=user_defined_symbols,
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


if __name__ == "__main__":
    main()
