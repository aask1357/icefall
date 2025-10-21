from pathlib import Path
import argparse
from collections import defaultdict
import warnings
import math
import re
import time
import multiprocessing as mp

import soundfile as sf
from tqdm import tqdm
import numpy as np
from lhotse import RecordingSet, Recording, SupervisionSegment, SupervisionSet, CutSet
from lhotse.recipes.utils import read_manifests_if_cached
import pykakasi

from filter_hiragana import filter_hiragana, HIRAGANA


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--data-dir", type=Path)
    parser.add_argument("-t", "--transcript-dir", type=Path)
    return parser.parse_args()


def main():
    args = get_args()

    dataset_parts = (
        "dev",
        "test",
        "train",
    )

    prefix = "reazonspeech"
    suffix = "jsonl.gz"
    print(f"Read manifests...")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=args.data_dir / "manifests",
        prefix=f"{prefix}_",
        suffix=suffix,
    )

    transcript = ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kakasi = pykakasi.kakasi()
        kakasi.setMode("K", "H")
        kakasi.setMode("H", None)
        kakasi.setMode("J", "H")
        converter = kakasi.getConverter()
        
        for partition, manifest in manifests.items():
            num_processed = 0
            dataset_length = 0
            new_recordings = []
            new_supervisions = []
            
            recordings = manifest["recordings"]
            supervisions = manifest["supervisions"]
            pbar = tqdm(dynamic_ncols=True, total=len(recordings), desc=f"{partition}", leave=False)
            for rec, sup in zip(recordings, supervisions):
                pbar.update(1)
                text = re.sub("[［］〈〉・♪《》−．⁉“”◆）‼≪。、]", "", sup.text)
                output = converter.do(text)
                output, filtered = filter_hiragana(output, check=True)
                if not filtered:
                    num_processed += 1
                    new_recordings.append(rec)
                    sup.text = text
                    sup.custom = {"hiragana_filtered": output}
                    transcript += f"{output}\n"
                    new_supervisions.append(sup)
                    dataset_length += rec.duration
            pbar.close()
            print(
                f"[{partition}] {num_processed}/{len(recordings)} processed, "
                f"total duration: {dataset_length/60/60:.1f} hours"
            )
            rec = RecordingSet.from_recordings(new_recordings)
            sup = SupervisionSet.from_segments(new_supervisions)
            cutset = CutSet.from_manifests(recordings=rec, supervisions=sup)
            rec.to_file(args.data_dir / f"manifests/reazonspeech_recordings_{partition}.jsonl.gz")
            sup.to_file(args.data_dir / f"manifests/reazonspeech_supervisions_{partition}.jsonl.gz")
            cutset.to_file(args.data_dir / f"manifests/reazonspeech_cuts_{partition}.jsonl.gz")
            
            if partition == "train":
                args.transcript_dir.mkdir(parents=True, exist_ok=True)
                with open(args.transcript_dir / "transcript_words.txt", "w", encoding="utf-8") as f:
                    f.write(transcript)

if __name__ == "__main__":
    main()
