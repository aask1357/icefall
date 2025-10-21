from typing import Tuple


# 스테가나 (https://namu.wiki/w/%EC%8A%A4%ED%85%8C%EA%B0%80%EB%82%98)
NORMALIZATION_RULES = [
    ['キャ', '캬', 'kya'], ['キュ', '큐', 'kyu'], ['キョ', '쿄', 'kyo'],
    ['シャ', '샤', 'sha'], ['シュ', '슈', 'shu'], ['ショ', '쇼', 'sho'],
    ['チャ', '챠', 'cha'], ['チュ', '츄', 'chu'], ['チョ', '쵸', 'cho'],
    ['ニャ', '냐', 'nya'], ['ニュ', '뉴', 'nyu'], ['ニョ', '뇨', 'nyo'],
    ['ヒャ', '햐', 'hya'], ['ヒュ', '휴', 'hyu'], ['ヒョ', '효', 'hyo'],
    ['ミャ', '먀', 'mya'], ['ミュ', '뮤', 'myu'], ['ミョ', '묘', 'myo'],
    ['リャ', '랴', 'rya'], ['リュ', '류', 'ryu'], ['リョ', '료', 'ryo'],
    ['ギャ', '갸', 'gya'], ['ギュ', '규', 'gyu'], ['ジョ', '교', 'gyo'],
    ['ビャ', '뱌', 'bya'], ['ビュ', '뷰', 'byu'], ['ビョ', '뵤', 'byo'],
    ['ピャ', '퍄', 'pya'], ['ピュ', '퓨', 'pyu'], ['ピョ', '표', 'pyo']
]
HOMONYM_RULES = [
    ['ジャ', '자', 'ja'],  ['ジュ', '주', 'ju'],  ['ギョ', '조', 'jo'],
    ['ヂャ', '자', 'ja'],  ['ヂュ', '주', 'ju'],  ['ヂョ', '조', 'jo'],
    ['イ', '이', 'i'], ['ヰ', '이', 'i'],
    ['エ', '에', 'e'], ['ヱ', '에', 'e'],
    ['オ', '오', 'o'], ['ヲ', '오', 'o'],
    ['ジ', '지', 'ji'], ['ヂ', '지', 'ji'],
    ['ズ', '주', 'zu'], ['ヅ', '주', 'zu']
]
KATAKANA = set(['ヶ', 'ォ', 'ゥ', 'ヴ', 'ヌ', 'ユ', 'ド', 'ー', 'グ', 'ゲ', 'ヅ', 'オ', 'チ', 'ゼ', 'ペ', 'ァ', 'ョ', 'コ', 'ミ', 'ロ', 'ェ', 'ビ', 'ダ', 'ト', 'ホ', 'エ', 'レ', 'ジ', 'ボ', 'タ', 'ャ', 'ヨ', 'ギ', 'イ', 'ケ', 'テ', 'フ', 'ヤ', 'ベ', 'ス', 'リ', 'ハ', 'ゴ', 'ネ', 'シ', 'モ', 'パ', 'ヘ', 'メ', 'ン', 'ニ', 'ブ', 'ム', 'デ', 'ウ', 'ピ', 'ヒ', 'ナ', 'ュ', 'ラ', 'ツ', 'ル', 'カ', 'セ', 'ク', 'ア', 'ノ', 'ガ', 'サ', 'ワ', 'ザ', 'マ', 'ィ', 'ズ', 'ソ', 'ゾ', 'プ', 'バ', 'ポ', 'ヲ', 'キ', 'ッ'] + [x[1] for x in NORMALIZATION_RULES] + [x[1] for x in HOMONYM_RULES])


def filter_katakana(text: str, check: bool) -> Tuple[str, bool]:
    """return:
        filtered_text (str): text with katakana characters only
        filtered (bool): True if the text contains non-katakana characters"""
    for rule in NORMALIZATION_RULES:
        text = text.replace(rule[0], rule[1])
    for rule in HOMONYM_RULES:
        text = text.replace(rule[0], rule[1])

    if check:
        for word in text:
            if word not in KATAKANA:
                return text, True
    return text, False
