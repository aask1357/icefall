from pathlib import Path
import argparse
import warnings
import re
import json

import torch
from tqdm import tqdm
from lhotse import (
    RecordingSet,
    Recording,
    SupervisionSegment,
    SupervisionSet,
    CutSet,
)
from icefall.utils import str2bool, text_to_pinyin


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
        "-d", "--data-dir",
        default="/home/shahn/Datasets/aihub/002.라이브 스트리밍 영상 중국어 통번역 데이터/3.개방데이터/1.데이터",
        type=Path
    )
    parser.add_argument(
        "-l", "--lang-dir",
        default="data/lang_bpe_500",
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


class Dataset:
    def __init__(self, data_dir: Path, part: str):
        assert part in ["train", "valid"]
        self.part = part
        self.wav_path = []
        self.text = []

        if part == "train":
            self.base_dir = data_dir / "Training/01.원천데이터/2.중국어"
            label_dir = data_dir / "Training/02.라벨링데이터/TL/2.중국어"
            total_num = 0
            for file in label_dir.rglob(f"*.json"):
                total_num += 1
            for file in tqdm(
                label_dir.rglob(f"*.json"),
                total=total_num,
                smoothing=0.0, 
                dynamic_ncols=True,
                desc=f"[{part}] Reading transcripts.",
            ):
                with open(file, "r", encoding="utf-8") as f:
                    data = json.loads(f.read())
                path = file.relative_to(label_dir).with_suffix(".wav")
                text = data["tc_text"]
                self.wav_path.append(path)
                self.text.append(text)
                if len(self.wav_path) > 10000:
                    break
        elif part == "valid":
            print(f"[{part}] Reading transcripts.")
            self.base_dir = data_dir / "Validation/01.원천데이터/VS/2.중국어"
            label_dir = data_dir / "Validation/02.라벨링데이터/VL/2.중국어"
            for file in label_dir.rglob(f"*.json"):
                with open(file, "r", encoding="utf-8") as f:
                    data = json.loads(f.read())
                path = file.relative_to(label_dir).with_suffix(".wav")
                text = data["tc_text"]
                self.wav_path.append(path)
                self.text.append(text)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx: int):
        path = self.base_dir / self.wav_path[idx]
        text = self.text[idx]
        return path, text


def get_time_string(seconds: float) -> str:
    seconds = int(seconds)
    second = seconds % 60
    minute = seconds // 60 % 60
    hour = seconds // 3600
    return f"{hour}:{minute:02d}:{second:02d}"


symbols = " abcdefghijklmnopqrstuvwxyzāáǎàēéěèīíǐìōóǒòūúǔùüǘǚǜ"


def main():
    args = get_args()

    if not args.force and (args.manifests_dir / f".manifest.done").exists():
        print(
            f"Previous manifests computed for aihub found. "
            f"If you want to overwrite, use --force option."
        )
        return

    for part in ["train", "valid"]:
        dataset = Dataset(args.data_dir, part)

        num_processed = 0
        dataset_length = 0
        recordings = []
        supervisions = []
        transcript = ""

        for idx, (path, hanja) in enumerate(tqdm(
            dataset, dynamic_ncols=True, desc=f"[{part}] Processing dataset", smoothing=0
        )):
            rec = Recording.from_file(path, recording_id=idx)
            pinyin = " ".join(text_to_pinyin(hanja, mode="full_with_tone"))
            pinyin = re.sub("[?!.,、，。？！]", " ", pinyin)
            pinyin = re.sub("\s+", " ", pinyin)
            breakpoint()
            # Too noisy. We should train a model, transcript, compare, and select those with low CER.
            found = False
            for char in pinyin:
                if char not in symbols:
                    found = True
                    break
            # if found:
            #     print(f"Invalid hanja : {hanja}")
            #     print(f"Invalid pinyin: {pinyin}")
            #     print(f"Invalid text  : ", end="")
            #     for char in pinyin:
            #         if char not in symbols:
            #             print(char, end=", ")
            #     print()
            # continue
            if not found:
                sup = SupervisionSegment(
                    id=idx,
                    recording_id=idx,
                    text=hanja,
                    start=0,
                    duration=rec.duration,
                    custom=dict(
                        pinyin=pinyin,
                    )
                )
                recordings.append(rec)
                supervisions.append(sup)
                transcript += f"{pinyin}\n"
                num_processed += 1
                dataset_length += rec.duration
        print(
            f"[{part}] {num_processed}/{len(dataset)} processed, "
            f"total duration: {get_time_string(dataset_length)}."
        )
        rec = RecordingSet.from_recordings(recordings)
        sup = SupervisionSet.from_segments(supervisions)
        cutset = CutSet.from_manifests(recordings=rec, supervisions=sup)
        rec.to_file(args.manifests_dir / f"aihub_ch_recordings_{part}.jsonl.gz")
        sup.to_file(args.manifests_dir / f"aihub_ch_supervisions_{part}.jsonl.gz")
        path = args.manifests_dir / f"aihub_ch_cuts_{part}.jsonl.gz"
        if part == "train":
            if args.perturb_speed:
                print(f"[train] Perturbing speed with factor 0.9 and 1.1.")
                cutset = (
                    cutset + cutset.perturb_speed(0.9) + cutset.perturb_speed(1.1)
                )
        print(f"[{part}] Saving to {path}.")
        cutset.to_file(path)
        if part == "train":
            lang_dir = args.lang_dir.parent / f"lang_bpe_500_pinyin"
            lang_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{part}] Saving transcript to {lang_dir}.")
            with open(lang_dir / "transcript_words.txt", "w", encoding="utf-8") as f:
                f.write(transcript)
    (args.manifests_dir / f".aihub-ch.done").touch()

if __name__ == "__main__":
    main()
