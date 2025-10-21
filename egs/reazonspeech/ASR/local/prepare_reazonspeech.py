from pathlib import Path
import argparse
import warnings
import re
import os

import torch
from tqdm import tqdm
from lhotse import (
    RecordingSet,
    Recording,
    SupervisionSegment,
    SupervisionSet,
    CutSet,
)
from icefall.utils import str2bool
import pykakasi
from sudachipy import Dictionary

from normalize import normalize
from filter_katakana import filter_katakana
from filter_hiragana import filter_hiragana


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--manifests-dir",
        default="/home/shahn/Documents/icefall/egs/librispeech/ASR/data/manifests",
        type=Path
    )
    parser.add_argument(
        "-r", "--reazonspeech-dir",
        default="/home/shahn/Datasets/reazonspeech-v2",
        type=Path
    )
    parser.add_argument(
        "-s", "--subset",
        default="large",
        type=str
    )
    parser.add_argument(
        "-l", "--lang-dir",
        default="data/lang_bpe_500_katakana",
        type=Path
    )
    parser.add_argument(
        "-f", "--force",
        action='store_true',
        help='Force overwrite existing files.'
    )
    parser.add_argument(
        "-p", "--perturb-speed",
        type=str2bool,
        default=True,
        help="Perturb speed with factor 0.9 and 1.1 on train subset."
    )
    return parser.parse_args()


class ReazonSpeech:
    def __init__(self, subset: str, data_dir: Path, mode: str):
        self.data_dir = data_dir
        with open(data_dir / "meta" / f"{subset}.tsv") as f:
            wav_text_list = [l.strip().split("	") for l in f.readlines()]
        
        if mode == "dev":
            self.wav_text_list = wav_text_list[:1000]
        elif mode == "test":
            self.wav_text_list = wav_text_list[1000:2000]
        elif mode == "train":
            self.wav_text_list = wav_text_list[2000:]
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __len__(self):
        return len(self.wav_text_list)

    def __getitem__(self, idx: int):
        filename, text = self.wav_text_list[idx]
        path = self.data_dir / "audio" / f"{filename}"
        return path, text


def get_time_string(seconds: float) -> str:
    seconds = int(seconds)
    second = seconds % 60
    minute = seconds // 60 % 60
    hour = seconds // 3600
    return f"{hour}:{minute:02d}:{second:02d}"


def main():
    args = get_args()

    if not args.force and (args.manifests_dir / f".reazonspeech-{args.subset}.done").exists():
        print(
            f"Previous manifests computed for ReazonSpeech-{args.subset} found. "
            f"If you want to overwrite, use --force option."
        )
        return

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

        for part in ["dev", "test", "train"]:
            print(f"[{part}] Reading transcripts.")
            dataset = ReazonSpeech(args.subset, args.reazonspeech_dir, part)

            num_processed = 0
            dataset_length = 0
            recordings = []
            supervisions = []
            transcript_k = ""
            transcript_h = ""


            for idx, (path, text_orig) in enumerate(tqdm(
                dataset, dynamic_ncols=True, desc=f"[{part}] Processing dataset", smoothing=0
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
                transcript_k += f"{katakana}\n"
                transcript_h += f"{hiragana}\n"
                num_processed += 1
                dataset_length += rec.duration
            print(
                f"[{part}] {num_processed}/{len(dataset)} processed, "
                f"total duration: {get_time_string(dataset_length)}."
            )
            rec = RecordingSet.from_recordings(recordings)
            sup = SupervisionSet.from_segments(supervisions)
            cutset = CutSet.from_manifests(recordings=rec, supervisions=sup)
            rec.to_file(args.manifests_dir / f"reazonspeech_{args.subset}_recordings_{part}.jsonl.gz")
            sup.to_file(args.manifests_dir / f"reazonspeech_{args.subset}_supervisions_{part}.jsonl.gz")
            if part == "train":
                path = args.manifests_dir / f"reazonspeech_{args.subset}_cuts_{part}_raw.jsonl.gz"
                if args.perturb_speed:
                    print(f"[train] Perturbing speed with factor 0.9 and 1.1.")
                    cutset = (
                        cutset + cutset.perturb_speed(0.9) + cutset.perturb_speed(1.1)
                    )
            else:
                path = args.manifests_dir / f"reazonspeech_{args.subset}_cuts_{part}.jsonl.gz"
            print(f"[{part}] Saving to {path}.")
            cutset.to_file(path)
            if part == "train":
                for phoneme, transcript in zip(["katakana", "hiragana"], [transcript_k, transcript_h]):
                    lang_dir = args.lang_dir.parent / f"lang_bpe_500_{phoneme}"
                    lang_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[{part}] Saving transcript to {lang_dir}.")
                    with open(lang_dir / "transcript_words.txt", "w", encoding="utf-8") as f:
                        f.write(transcript)
    (args.manifests_dir / f".reazonspeech-{args.subset}.done").touch()

if __name__ == "__main__":
    main()
