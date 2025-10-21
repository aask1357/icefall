from pathlib import Path
import argparse
import os

import torch
from tqdm import tqdm
from lhotse import (
    CutSet,
    LilcomChunkyWriter,
)
from local.custom_fbank import CustomFbank, CustomFbankConfig


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f", "--fbank-dir",
        default="/home/shahn/Documents/icefall/egs/librispeech/ASR/data/fbank",
        type=Path
    )
    parser.add_argument(
        "-m", "--manifests-dir",
        default="/home/shahn/Documents/icefall/egs/librispeech/ASR/data/manifests",
        type=Path
    )
    parser.add_argument(
        "-s", "--subset",
        default="large",
        type=str
    )
    parser.add_argument(
        "-n","--num-workers",
        default=32,
        type=int
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=6000.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing files.'
    )
    return parser.parse_args()


def main():
    args = get_args()

    extractor = CustomFbank(
        CustomFbankConfig(
            num_mel_bins=80,
            device="cuda"
        )
    )

    partitions = ["dev", "test", "train"]
    for part in partitions:
        # Check if output exists
        output_path = args.fbank_dir / f"reazonspeech_{args.subset}_cuts_{part}.jsonl.gz"
        if not args.force and output_path.exists():
            print(
                f"[{args.subset}] Previous fbank computed for ReazonSpeech-{args.subset} found. "
                f"If you want to overwrite, use --force option."
            )
            continue

        # Load cuts
        print(f"[{part}] Loading cuts.")
        cuts_path = args.manifests_dir / f"reazonspeech_{args.subset}_cuts_{part}"
        if part == "train":
            cutset = CutSet.from_file(f"{cuts_path}_raw.jsonl.gz")
        else:
            cutset = CutSet.from_file(f"{cuts_path}.jsonl.gz")

        # Compute and store features
        num_workers = 1
        if part == "train":
            num_workers = min(args.num_workers, os.cpu_count())
        print(f"[{part}] ", end="", flush=True)
        cutset = cutset.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=(args.fbank_dir / f"reazonspeech_{args.subset}_feats_{part}").as_posix(),
            storage_type=LilcomChunkyWriter,
            num_workers=num_workers,
            batch_duration=args.batch_duration,
            overwrite=True,
        )
        print(f"[{part}] Saving to {output_path}.")
        cutset.to_file(output_path)

if __name__ == "__main__":
    main()
