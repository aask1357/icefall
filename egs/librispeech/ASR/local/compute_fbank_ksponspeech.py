#!/usr/bin/env python3
# Copyright    2024   (Author: SeungHyun Lee, Contacts: whsqkaak@naver.com)
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


import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
import numpy as np
from filter_cuts import filter_cuts
from lhotse import CutSet, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor, str2bool
from custom_fbank import CustomFbank, CustomFbankConfig

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to the bpe.model. If not None, we will remove short and
        long utterances before extracting features""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="""Dataset parts to compute fbank. If None, we will use all""",
    )

    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=True,
        help="""Perturb speed with factor 0.9 and 1.1 on train subset.""",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="""Path of data directory""",
    )

    return parser.parse_args()


def compute_fbank_speechtools(
    bpe_model: Optional[str] = None,
    dataset: Optional[str] = None,
    perturb_speed: Optional[bool] = False,
    data_dir: Optional[str] = "data",
):
    # src_dir = Path(data_dir) / "manifests"
    src_dir = Path("/home/shahn/Documents/icefall/egs/librispeech/ASR/data/manifests")
    output_dir = Path(data_dir) / "fbank"
    num_jobs = min(4, os.cpu_count())
    num_mel_bins = 80

    if bpe_model:
        logging.info(f"Loading {bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    if dataset is None:
        dataset_parts = (
            "train",
            "dev",
            "eval_clean",
            "eval_other",
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    prefix = "ksponspeech"
    suffix = "jsonl.gz"
    logging.info(f"Read manifests...")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=f"{prefix}_",
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    if torch.cuda.is_available():
        # Use cuda for fbank compute
        device = "cuda"
    else:
        device = "cpu"
    logging.info(f"Device: {device}")

    extractor = CustomFbank(CustomFbankConfig(num_mel_bins=num_mel_bins)).to("cuda")

    with get_executor() as ex:  # Initialize the executor only once.
        logging.info(f"Executor: {ex}")
        for partition, m in manifests.items():
            cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
            if (output_dir / cuts_filename).is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )

            # Filter duration
            cut_set = cut_set.filter(
                lambda x: x.duration >= 0.9 and x.sampling_rate == 16000
            )

            if "train" in partition:
                if bpe_model:
                    cut_set = filter_cuts(cut_set, sp)
                if perturb_speed:
                    logging.info(f"Doing speed perturb")
                    cut_set = (
                        cut_set
                        + cut_set.perturb_speed(0.9)
                        + cut_set.perturb_speed(1.1)
                    )
            logging.info(f"Compute & Store features...")
            if device == "cuda":
                cut_set = cut_set.compute_and_store_features_batch(
                    extractor=extractor,
                    storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                    num_workers=0,
                    storage_type=LilcomChunkyWriter,
                )
            else:
                cut_set = cut_set.compute_and_store_features(
                    extractor=extractor,
                    storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                    # when an executor is specified, make more partitions
                    num_jobs=num_jobs if ex is None else 80,
                    executor=ex,
                    storage_type=LilcomChunkyWriter,
                )
            cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_fbank_speechtools(
        bpe_model=args.bpe_model,
        dataset=args.dataset,
        perturb_speed=args.perturb_speed,
        data_dir=args.data_dir,
    )
