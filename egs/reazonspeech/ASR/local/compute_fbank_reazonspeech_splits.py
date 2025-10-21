from pathlib import Path
import argparse
import os
import glob
import sys

import torch
from tqdm import tqdm
from lhotse import (
    CutSet,
    LilcomChunkyWriter,
)
from local.custom_fbank import CustomFbank, CustomFbankConfig

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
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
        "-s", "--subset",
        default="large",
        type=str
    )
    parser.add_argument(
        "-n","--num-workers",
        default=32,
        type=int,
        help="Number of dataloading workers used for reading the audio.",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=600.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )
    return parser.parse_args()


def clear_current_line():
    # cols = array('h', ioctl(sys.stdout.fileno(), TIOCGWINSZ, '\0' * 8))[1]
    # print("\r" + " " * cols, flush=True, end="")
    sys.stdout.write("\033[2K") # clear the whole line
    print("\r", end="")         # move the cursor to the beginning of the line


def main():
    args = get_args()
    path_split = args.fbank_dir / f"reazonspeech_{args.subset}_split_train"
    extractor = CustomFbank(
        CustomFbankConfig(
            num_mel_bins=80,
            device="cuda"
        )
    )

    device = torch.device("cuda", 0)
    num_splits = len([x for x in glob.glob(f"{path_split}/*_raw*.jsonl.gz")])

    for i in range(num_splits):
        idx = f"{i:0>8d}"
        clear_current_line()
        print(f"\rProcessing {i+1}/{num_splits}.", end="", flush=True)

        cuts_path = path_split / f"reazonspeech_{args.subset}_cuts_train.{idx}.jsonl.gz"
        if cuts_path.is_file():
            print(f"\r{cuts_path} exists - skipping.", end="", flush=True)
            continue

        raw_cuts_path = path_split / f"reazonspeech_{args.subset}_cuts_train_raw.{idx}.jsonl.gz"

        print(f"\nLoading {raw_cuts_path}.", end="", flush=True)
        cut_set = CutSet.from_file(raw_cuts_path)

        clear_current_line()
        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=path_split / f"reazonspeech_{args.subset}_train_feats_{idx}",
            num_workers=args.num_workers,
            batch_duration=args.batch_duration,
            overwrite=True,
        )

        print("About to split cuts into smaller chunks.", end="", flush=True)
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        clear_current_line()
        print(f"\rSaving to {cuts_path}.", end="", flush=True)
        cut_set.to_file(cuts_path)
    print("\r", flush=True)

if __name__ == "__main__":
    main()
