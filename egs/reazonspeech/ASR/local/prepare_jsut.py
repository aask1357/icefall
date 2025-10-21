from pathlib import Path
import argparse
import warnings
import re
import time
import os

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
        "-j", "--jsut-dir",
        default="/home/shahn/Datasets/jsut_ver1.1/basic5000",
        type=Path
    )
    parser.add_argument(
        "-t", "--tedxjp-dir",
        default="/home/shahn/Datasets/TEDxJP-10k/v1.1",
        type=Path
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


class JSUT:
    def __init__(self, data_dir: Path):
        self.sampling_rate = 48_000
        self.data_dir = data_dir
        with open(data_dir / "basic5000.yaml", encoding="utf-8") as f:
            data = yaml.safe_load(f.read())
        self.wav_text_list = []
        for file, config in data.items():
            # self.wav_text_list.append((file, config["kana_level3"].replace("、", "")))
            self.wav_text_list.append((file, config["text_level0"]))

    def __len__(self):
        return len(self.wav_text_list)
    
    def __getitem__(self, idx: int):
        filename, text = self.wav_text_list[idx]
        path = self.data_dir / "wav" / f"{filename}.wav"
        return path, text


def main():
    args = get_args()
    print("Reading transcripts...", end=" ", flush=True)
    datasets = {
        # "jsut-basic5000": JSUT(args.jsut_dir)
        "jsut-basic5000-sudachi": JSUT(args.jsut_dir)
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
                )
            )
            num_processed = 0
            dataset_length = 0
            recordings = []
            supervisions = []

            for idx, (path, text_orig) in enumerate(tqdm(
                dataset,
                dynamic_ncols=True,
                desc=f"[{dataset_name}] Processing dataset",
                smoothing=0
            )):
                rec = Recording.from_file(path, recording_id=idx)
                text = normalize(text_orig)
                katakana, filtered_k = encode_katakana(text)
                hiragana, filtered_h = encode_hiragana(text.replace("今日", "きょう"))
                if filtered_k or filtered_h:
                    continue
                sup = SupervisionSegment(
                    id=idx,
                    recording_id=idx,
                    text=text_orig,
                    start=0,
                    duration=rec.duration,
                    custom=dict(
                        hiragana=hiragana,
                        katakana=katakana,
                    )
                )
                recordings.append(rec)
                supervisions.append(sup)
                num_processed += 1
                dataset_length += rec.duration
            print(
                f"[{dataset_name}] {num_processed}/{len(dataset)} processed, "
                f"total duration: {get_time_string(dataset_length)}\n"
                f"[{dataset_name}] Saving recordings and supervisions."
            )
            rec = RecordingSet.from_recordings(recordings)
            sup = SupervisionSet.from_segments(supervisions)
            cutset = CutSet.from_manifests(recordings=rec, supervisions=sup)
            rec.to_file(args.manifests_dir / f"{dataset_name.replace("-", "_")}_recordings.jsonl.gz")
            sup.to_file(args.manifests_dir / f"{dataset_name.replace("-", "_")}_supervisions.jsonl.gz")

            print(f"[{dataset_name}] ", end="", flush=True)
            cutset = cutset.compute_and_store_features(
                extractor=extractor,
                num_jobs=1,
                storage_path=(args.fbank_dir / f"{dataset_name.replace("-", "_")}_feats").as_posix(),
                storage_type=LilcomChunkyWriter,
            )
            fbank_path = args.fbank_dir / f"{dataset_name.replace('-', '_')}_cuts.jsonl.gz"
            print(f"[{dataset_name}] Saving to {fbank_path}.")
            cutset.to_file(fbank_path)
        (args.fbank_dir / f".{dataset_name}-fbank.done").touch()

if __name__ == "__main__":
    main()
