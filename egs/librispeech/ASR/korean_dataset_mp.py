from pathlib import Path
from collections import defaultdict
import math
import re
import time
import multiprocessing as mp
import soundfile as sf
from tqdm import tqdm
import numpy as np
from lhotse import RecordingSet, Recording, SupervisionSegment, SupervisionSet, CutSet
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator


class TextProcessor:
    def __init__(self):
        self.dict_cho   = {0:u"ᄀ",  1:u"ᄁ",  2:u"ᄂ",  3:u"ᄃ",  4:u"ᄄ",  5:u"ᄅ",  6:u"ᄆ",  7:u"ᄇ",  8:u"ᄈ",  9:u"ᄉ",
            10:u"ᄊ", 11:u"ᄋ", 12:u"ᄌ", 13:u"ᄍ", 14:u"ᄎ", 15:u"ᄏ", 16:u"ᄐ", 17:u"ᄑ", 18:u"ᄒ"}
        self.dict_jung  = {0:u"ㅏ",  1:u"ㅐ",  2:u"ㅑ",  3:u"ㅒ",  4:u"ㅓ",  5:u"ㅔ",  6:u"ㅕ",  7:u"ㅖ",  8:u"ㅗ",  9:u"ㅘ",
            10:u"ㅙ", 11:u"ㅚ", 12:u"ㅛ", 13:u"ㅜ", 14:u"ㅝ", 15:u"ㅞ", 16:u"ㅟ", 17:u"ㅠ", 18:u"ㅡ", 19:u"ㅢ", 20:u"ㅣ"}
        self.dict_jong  = { 0:u" ",   1:u"ᆨ",  2:u"ᆩ",  3:u"ᆪ",  4:u"ᆫ",  5:u"ᆬ",  6:u"ᆭ",  7:u"ᆮ",  8:u"ᆯ",  9:u"ᆰ",  
            10:u"ᆱ", 11:u"ᆲ", 12:u"ᆳ", 13:u"ᆴ", 14:u"ᆵ", 15:u"ᆶ", 16:u"ᆷ", 17:u"ᆸ", 18:u"ᆹ", 19:u"ᆺ", 
            20:u"ᆻ", 21:u"ᆼ", 22:u"ᆽ", 23:u"ᆾ", 24:u"ᆿ", 25:u"ᇀ", 26:u"ᇁ", 27:u"ᇂ"}

    def __call__(self, text: str) -> list:
        cjj = ""
        prefix = ""
        for unicode in text:
            enc = unicode.encode()
            if len(enc) == 3:   # 한글 자모
                h___ = enc[0]-224
                _h__ = (enc[1]-128) // 4
                next_ = (enc[1]-128) % 4
                __h_ = (next_*64 + enc[2]-128) // 16
                ___h = (next_*64 + enc[2]-128) % 16
                hex = h___ * 4096 + _h__ * 256 + __h_ * 16 + ___h
    
                if hex < 44032:
                    raise Exception(f"Invalid text ({unicode}) ({text})")
                cho  = self.dict_cho[(hex - 44032) // 588]
                jung = self.dict_jung[((hex - 44032) % 588) // 28]
                jong  = self.dict_jong[((hex - 44032) % 588) % 28]
                if jong == u" ": cjj = f"{cjj}{prefix}{cho}{jung}"    # 종성 없는 경우
                else : cjj = f"{cjj}{prefix}{cho}{jung}{jong}"        # 종성 있는 경우
                prefix = ""
            else:   # 문장부호
                if unicode not in [" "]:   # 문장 부호도 아닌 경우 에러 출력
                    raise Exception(f"Invalid text ({unicode}) ({text})")
                prefix = " "
        return cjj


AIHUB_MAPPING = dict(
    command_kid="명령어 음성(소아, 유아)",
    command_nor="명령어 음성(일반남녀)",
    command_old="명령어 음성(노인남녀)",
    freetalk_kid="자유대화 음성(소아유아)",
    freetalk_nor="자유대화 음성(일반남녀)",
    freetalk_old="자유대화 음성(노인남녀)",
)


def make_rec_sup(file, text, text_cjj, dataset, _id, ipa_backend, ipa_separator):
    _id = str(_id)
    text_ipa = ipa_backend.phonemize([text], separator=ipa_separator, njobs=1)[0]
    if dataset == "ksponspeech":
        name = re.match("ksponspeech\/KsponSpeech_\d{2}(?:_train|_test)?\/(KsponSpeech_\d{4}\/KsponSpeech_\d{6}\.)", file).groups()[0]
        # path = Path("/home/shahn/Datasets/hinas2/aihub/ksponspeech/Ksponall_wav") / f"{name}wav"
        path = Path("/home/shahn/Datasets/aihub/ksponspeech/Ksponall_wav") / f"{name}wav"
        if not path.exists():
            path = Path("/home/shahn/Datasets/aihub/ksponspeech/ksponspeech_wav_omitted") / f"{name}wav"
        rec = Recording.from_file(path, recording_id=_id)
        sup = SupervisionSegment(
            id=_id,
            recording_id=_id,
            text=text,
            start=0,
            duration=rec.duration,
            custom=dict(
                cjj=text_cjj,
                ipa=text_ipa
            ),
        )
    elif dataset == "zeroth":
        path = Path("/home/shahn/Datasets") / f"{file}flac"
        rec = Recording.from_file(path, recording_id=_id)
        sup = SupervisionSegment(
            id=_id,
            recording_id=_id,
            text=text,
            start=0,
            duration=rec.duration,
            custom=dict(
                cjj=text_cjj,
                ipa=text_ipa
            ),
        )
    elif dataset == "zeroth_wav":
        parts = Path(file).parts
        path = Path("/home/shahn/Datasets/zeroth") / f"{parts[1]}_wav" / f"{parts[-1]}wav"
        rec = Recording.from_file(path, recording_id=_id)
        sup = SupervisionSegment(
            id=_id,
            recording_id=_id,
            text=text,
            start=0,
            duration=rec.duration,
            custom=dict(
                cjj=text_cjj,
                ipa=text_ipa
            ),
        )
    elif dataset == "freetalk_old":
        return None, None, None
    elif dataset in ["command_nor", "freetalk_nor"]:
        hi14_path = AIHUB_MAPPING[dataset]
        ext = "wav"
        path = Path(f"/home/shahn/Datasets/aihub/{hi14_path}/wav_16khz") / f"{file[len(dataset)+1:]}{ext}"
        rec = Recording.from_file(path, recording_id=_id)
        sup = SupervisionSegment(
            id=_id,
            recording_id=_id,
            text=text,
            start=0,
            duration=rec.duration,
            custom=dict(
                cjj=text_cjj,
                ipa=text_ipa
            ),
        )
    else:
        hi14_path = AIHUB_MAPPING[dataset]
        ext = "PCM" if dataset == "freetalk_old" else "wav"
        path = Path(f"/home/shahn/Datasets/hinas2/aihub/{hi14_path}") / f"{file[len(dataset)+1:]}{ext}"
        rec = Recording.from_file(path, recording_id=_id)
        sup = SupervisionSegment(
            id=_id,
            recording_id=_id,
            text=text,
            start=0,
            duration=rec.duration,
            custom=dict(
                cjj=text_cjj,
                ipa=text_ipa
            ),
        )
    return rec, sup, dataset


def main():
    tp = TextProcessor()
    ipa_backend = EspeakBackend('ko')
    ipa_separator = Separator(phone=None, word=None)

    with open("/home/shahn/Datasets/KoreanASR/train/wav.scp", "r") as f:
        filelist = f.readlines()
    with open("/home/shahn/Datasets/KoreanASR/train/text", "r") as f:
        textlist = f.readlines()
    assert len(filelist) == len(textlist)
    print("wav.scp and text loaded")

    recordings = defaultdict(list)
    supervisions = defaultdict(list)
    dataset_count = defaultdict(int)
    dataset_length = defaultdict(int)

    pbar = tqdm(dynamic_ncols=True, total=len(filelist))
    
    def append_results(results):
        rec, sup, dataset = results
        if dataset is not None:
            recordings[dataset].append(rec)
            supervisions[dataset].append(sup)
            dataset_length[dataset] += rec.duration
        pbar.update(1)

    def error_callback(e):
        print(f"error: {e}")
        pbar.update(1)

    jobs = []
    with mp.Pool(processes=32) as pool:
        for file_hslee, text_hslee in zip(filelist, textlist):
            file = re.search("(?<=\.\/db\/)[^\s]+\.(?=pcm|flac|wav|PCM)", file_hslee).group()
            
            dataset = file.split("/")[0]
            if dataset == "zeroth":
                dataset = "zeroth_wav"
            if dataset != "freetalk_nor":
                pbar.update(1)
                continue
            _id = dataset_count[dataset] + 1
            dataset_count[dataset] = _id
            
            try:
                text = re.match("[^\s]+ (.+)", text_hslee).groups()[0]
                text = re.sub("['\" .,?!…‘’“”<>\\`]+", " ", text).strip()
            except:
                tqdm.write(f"{text_hslee.rstrip()} {file}")
                continue
            try:
                text_cjj = tp(text)
            except:
                tqdm.write(f"{text} {file}")
            
            results = make_rec_sup(file, text, text_cjj, dataset, _id, ipa_backend, ipa_separator)
            append_results(results)
        #     job = pool.apply_async(
        #         make_rec_sup,
        #         args=(file, text, text_cjj, dataset, _id, ipa_backend, ipa_separator),
        #         callback=append_results,
        #         error_callback=error_callback,
        #     )
        #     jobs.append(job)
        # for job in jobs:
        #     job.wait()
    pbar.close()
    for dataset in dataset_count.keys():
        print(
            f"{dataset} - Total {dataset_count[dataset]} files, "
            f"{dataset_length[dataset]/60/60:.1f} hours"
        )
    for dataset in dataset_count.keys():
        print(f"\r{dataset}          ", end="", flush=True)
        rec = RecordingSet.from_recordings(recordings[dataset])
        sup = SupervisionSet.from_segments(supervisions[dataset])
        cutset = CutSet.from_manifests(recordings=rec, supervisions=sup)
        rec.to_file(f"data/manifests/{dataset}_recordings_train.jsonl.gz")
        sup.to_file(f"data/manifests/{dataset}_supervisions_train.jsonl.gz")
        cutset.to_file(f"data/fbank/{dataset}_cuts_train.jsonl.gz")
    print("\r", end="", flush=True)


if __name__ == "__main__":
    main()
