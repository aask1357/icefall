from pathlib import Path
import argparse
import warnings
import re
import time
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from lhotse import (
    RecordingSet,
    Recording,
    SupervisionSegment,
    SupervisionSet,
    CutSet,
    LilcomChunkyWriter,
)
import pykakasi
from sudachipy import Dictionary
import yaml

from normalize import normalize
from filter_katakana import filter_katakana
from filter_hiragana import filter_hiragana
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
        default="/home/shahn/Documents/icefall_github/egs/librispeech/ASR/data/manifests",
        type=Path
    )
    parser.add_argument(
        "-t", "--tedxjp-dir",
        default="/home/shahn/Datasets/TEDxJP-10K/v1.1",
        type=Path
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=1000.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing files.'
    )
    return parser.parse_args()


def get_time_string(seconds: float) -> str:
    seconds = int(seconds)
    second = seconds % 60
    minute = seconds // 60 % 60
    hour = seconds // 3600
    return f"{hour}:{minute:02d}:{second:02d}"


class TEDxJP:
    def __init__(self, data_dir: Path):
        self.sampling_rate = 48_000
        self.data_dir = data_dir
        self.fid_list = []
        self.dict_of_list = defaultdict(list)
        self.num_items = 0
        with open(data_dir / "text", encoding="utf-8") as f:
            for line in f.readlines():
                file, text = line.strip().split(" ")
                fid = file[:11]
                start = file[12:20]
                end = file[21:29]
                start = float(start) / 100
                duration = float(end) / 100 - start
                if fid not in self.fid_list:
                    self.fid_list.append(fid)
                self.dict_of_list[fid].append((text, start, duration))
                self.num_items += 1

    def __len__(self):
        return len(self.fid_list)
    
    def __getitem__(self, idx: int):
        fid = self.fid_list[idx]
        text_start_dur_list = self.dict_of_list[fid]
        path = self.data_dir / "wav" / f"{fid}.16k.wav"
        return path, text_start_dur_list


def main():
    args = get_args()
    print("Reading transcripts...", end=" ", flush=True)
    datasets = {
        "tedxjp-10k-v1.1": TEDxJP(args.tedxjp_dir)
    }
    print("Done.")

    sudachi = Dictionary(dict='full').create()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kakasi = pykakasi.kakasi()
        kakasi.setMode("K", None)
        kakasi.setMode("H", "K")
        kakasi.setMode("J", None)
        kakasi.setMode("a", None)
        kakasi.setMode("E", None)
        hira_to_kata = kakasi.getConverter()

        kakasi = pykakasi.kakasi()
        kakasi.setMode("K", "H")
        kakasi.setMode("H", None)
        kakasi.setMode("J", "H")
        to_hira = kakasi.getConverter()

        def encode_katakana(text):
            morpheme_list = sudachi.tokenize(text)
            katakana = ""
            for morpheme in morpheme_list:
                if morpheme.surface() in [" "]:
                    continue
                katakana = f"{katakana}{morpheme.reading_form()}"
            katakana = katakana.replace("ー", "").replace("・", "")
            katakana = hira_to_kata.do(katakana)
            katakana, filtered = filter_katakana(katakana, check=True)
            return katakana, filtered

        def encode_hiragana(text):
            hiragana = to_hira.do(text)
            hiragana = hiragana.replace(" ", "")
            hiragana, filtered = filter_hiragana(hiragana, check=True)
            return hiragana, filtered
        
        for dataset_name, dataset in datasets.items():
            if not args.force and (args.fbank_dir / f".{dataset_name}-fbank.done").exists():
                print(
                    f"Previous fbank computed for {dataset_name} found. "
                    f"If you want to overwrite, use --force option."
                )
                continue
            extractor = CustomFbank(
                CustomFbankConfig(
                    num_mel_bins=80,
                    sampling_rate_orig=dataset.sampling_rate,
                    device="cuda"
                )
            )
            num_processed = 0
            dataset_length = 0
            recordings = []
            supervisions = []
            idx_sup = 0
            pbar = tqdm(
                dynamic_ncols=True,
                desc=f"[{dataset_name}] Processing dataset",
                smoothing=0,
                total=dataset.num_items
            )
            for idx_rec, (path, text_start_end_list) in enumerate(dataset):
                rec = Recording.from_file(path, recording_id=idx_rec)
                recordings.append(rec)
                for text_orig, start, duration in text_start_end_list:
                    text = normalize(text_orig)
                    katakana, filtered_k = encode_katakana(text)
                    hiragana, filtered_h = encode_hiragana(text.replace("今日", "きょう"))
                    if filtered_k or filtered_h:
                        continue
                    sup = SupervisionSegment(
                        id=idx_sup,
                        recording_id=idx_rec,
                        text=text_orig,
                        start=start,
                        duration=duration,
                        custom=dict(
                            hiragana=hiragana,
                            katakana=katakana,
                        )
                    )
                    supervisions.append(sup)
                    num_processed += 1
                    dataset_length += duration
                    idx_sup += 1
                    pbar.update(1)
            pbar.close()
            print(
                f"[{dataset_name}] {num_processed}/{dataset.num_items} processed, "
                f"total duration: {get_time_string(dataset_length)}\n"
                f"[{dataset_name}] Saving recordings and supervisions."
            )
            rec = RecordingSet.from_recordings(recordings)
            sup = SupervisionSet.from_segments(supervisions)
            cutset = CutSet.from_manifests(recordings=rec, supervisions=sup)
            rec.to_file(args.manifests_dir / f"{dataset_name.replace("-", "_")}_recordings.jsonl.gz")
            sup.to_file(args.manifests_dir / f"{dataset_name.replace("-", "_")}_supervisions.jsonl.gz")

            print(f"[{dataset_name}] ", end="", flush=True)
            cutset = cutset.compute_and_store_features_batch(
                extractor=extractor,
                num_workers=1,
                batch_duration=args.batch_duration,
                storage_path=(args.fbank_dir / f"{dataset_name.replace("-", "_")}_feats").as_posix(),
                storage_type=LilcomChunkyWriter,
            )
            print(f"[{dataset_name}] Saving to {path}.")
            cutset.to_file(args.fbank_dir / f"{dataset_name.replace("-", "_")}_cuts.jsonl.gz")
        (args.fbank_dir / f".{dataset_name}-fbank.done").touch()

if __name__ == "__main__":
    main()
