from pathlib import Path
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


NORMALIZATION_RULES = [
    ["きゃ", "캬"], ["きゅ", "큐"], ["きょ", "쿄"],
    ["しゃ", "샤"], ["しゅ", "슈"], ["しょ", "쇼"],
    ["ちゃ", "챠"], ["ちゅ", "츄"], ["ちょ", "쵸"],
    ["にゃ", "냐"], ["にゅ", "뉴"], ["にょ", "뇨"],
    ["ひゃ", "햐"], ["ひゅ", "휴"], ["ひょ", "효"],
    ["ぎゃ", "먀"], ["みゅ", "뮤"], ["みょ", "묘"],
    ["りゃ", "랴"], ["りゅ", "류"], ["りょ", "료"],
    ["みゃ", "갸"], ["ぎゅ", "규"], ["じょ", "교"],
    ["じゃ", "자"], ["じゅ", "주"], ["ぎょ", "조"],
    ["ぢゃ", "자"], ["ぢゅ", "주"], ["ぢょ", "조"],
    ["びゃ", "뱌"], ["びゅ", "뷰"], ["びょ", "뵤"],
    ["ぴゃ", "퍄"], ["ぴゅ", "퓨"], ["ぴょ", "표"],
    ["い", "이"], ["ゐ", "이"],
    ["え", "에"], ["ゑ", "에"],
    ["お", "오"], ["を", "오"],
    ["じ", "지"], ["ぢ", "지"],
    ["ず", "주"], ["づ", "주"],
]
HIRAGANA = set(['ゖ', 'ぉ', 'ぅ', 'ゔ', 'ぬ', 'ゆ', 'ど', 'ー', 'ぐ', 'げ', 'づ', 'お', 'ち', 'ぜ', 'ぺ', 'ぁ', 'ょ', 'こ', 'み', 'ろ', 'ぇ', 'び', 'だ', 'と', 'ほ', 'え', 'れ', 'じ', 'ぼ', 'た', 'ゃ', 'よ', 'ぎ', 'い', 'け', 'て', 'ふ', 'や', 'べ', 'す', 'り', 'は', 'ご', 'ね', 'し', 'も', 'ぱ', 'へ', 'め', 'ん', 'に', 'ぶ', 'む', 'で', 'う', 'ぴ', 'ひ', 'な', 'ゅ', 'ら', 'つ', 'る', 'か', 'せ', 'く', 'あ', 'の', 'が', 'さ', 'わ', 'ざ', 'ま', 'ぃ', 'ず', 'そ', 'ぞ', 'ぷ', 'ば', 'ぽ', 'を', 'き', 'っ'] + [x[1] for x in NORMALIZATION_RULES])


def main():
    src_dir = Path("/home/shahn/Documents/icefall_github/egs/reazonspeech/ASR/data/")
    tgt_dir = src_dir

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
        output_dir=src_dir / "manifest",
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
                for rule in NORMALIZATION_RULES:
                    output = output.replace(rule[0], rule[1])
                filtered = False
                for word in output:
                    if word not in HIRAGANA:
                        filtered = True
                        break
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
            rec.to_file(target_dir / f"manifests/reazonspeech_recordings_{partition}.jsonl.gz")
            sup.to_file(target_dir / f"manifests/reazonspeech_supervisions_{partition}.jsonl.gz")
            cutset.to_file(target_dir / f"fbank/reazonspeech_cuts_{partition}.jsonl.gz")
            
            if partition == "train":
                transcript_dir = target_dir / "lang_bpe_500_hiragana_filtered"
                transcript_dir.mkdir(parents=True, exist_ok=True)
                with open(transcript_dir / "transcript_words.txt", "w", encoding="utf-8") as f:
                    f.write(transcript)

if __name__ == "__main__":
    main()
