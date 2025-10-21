from typing import Tuple


# 스테가나 (https://namu.wiki/w/%EC%8A%A4%ED%85%8C%EA%B0%80%EB%82%98)
NORMALIZATION_RULES = [
    ["きゃ", "캬", "kya"], ["きゅ", "큐", "kyu"], ["きょ", "쿄", "kyo"],
    ["しゃ", "샤", "sha"], ["しゅ", "슈", "shu"], ["しょ", "쇼", "sho"],
    ["ちゃ", "챠", "cha"], ["ちゅ", "츄", "chu"], ["ちょ", "쵸", "cho"],
    ["にゃ", "냐", "nya"], ["にゅ", "뉴", "nyu"], ["にょ", "뇨", "nyo"],
    ["ひゃ", "햐", "hya"], ["ひゅ", "휴", "hyu"], ["ひょ", "효", "hyo"],
    ["ぎゃ", "먀", "mya"], ["みゅ", "뮤", "myu"], ["みょ", "묘", "myo"],
    ["りゃ", "랴", "rya"], ["りゅ", "류", "ryu"], ["りょ", "료", "ryo"],
    ["みゃ", "갸", "gya"], ["ぎゅ", "규", "gyu"], ["じょ", "교", "gyo"],
    ["びゃ", "뱌", "bya"], ["びゅ", "뷰", "byu"], ["びょ", "뵤", "byo"],
    ["ぴゃ", "퍄", "pya"], ["ぴゅ", "퓨", "pyu"], ["ぴょ", "표", "pyo"],
]
HOMONYM_RULES = [
    ["じゃ", "자", "ja"],  ["じゅ", "주", "ju"],  ["ぎょ", "조", "jo"],
    ["ぢゃ", "자", "ja"],  ["ぢゅ", "주", "ju"],  ["ぢょ", "조", "jo"],
    ["い", "이", "i"], ["ゐ", "이", "i"],
    ["え", "에", "e"], ["ゑ", "에", "e"],
    ["お", "오", "o"], ["を", "오", "o"],
    ["じ", "지", "ji"], ["ぢ", "지", "ji"],
    ["ず", "주", "zu"], ["づ", "주", "zu"],
]
HIRAGANA = set(['ゖ', 'ぉ', 'ぅ', 'ゔ', 'ぬ', 'ゆ', 'ど', 'ー', 'ぐ', 'げ', 'づ', 'お', 'ち', 'ぜ', 'ぺ', 'ぁ', 'ょ', 'こ', 'み', 'ろ', 'ぇ', 'び', 'だ', 'と', 'ほ', 'え', 'れ', 'じ', 'ぼ', 'た', 'ゃ', 'よ', 'ぎ', 'い', 'け', 'て', 'ふ', 'や', 'べ', 'す', 'り', 'は', 'ご', 'ね', 'し', 'も', 'ぱ', 'へ', 'め', 'ん', 'に', 'ぶ', 'む', 'で', 'う', 'ぴ', 'ひ', 'な', 'ゅ', 'ら', 'つ', 'る', 'か', 'せ', 'く', 'あ', 'の', 'が', 'さ', 'わ', 'ざ', 'ま', 'ぃ', 'ず', 'そ', 'ぞ', 'ぷ', 'ば', 'ぽ', 'を', 'き', 'っ'] + [x[1] for x in NORMALIZATION_RULES] + [x[1] for x in HOMONYM_RULES])


def filter_hiragana(text: str, check: bool) -> Tuple[str, bool]:
    """return:
        filtered_text (str): text with hiragana characters only
        filtered (bool): True if the text contains non-hiragana characters"""
    for rule in NORMALIZATION_RULES:
        text = text.replace(rule[0], rule[1])
    for rule in HOMONYM_RULES:
        text = text.replace(rule[0], rule[1])

    if check:
        for word in text:
            if word not in HIRAGANA:
                return text, True
    return text, False
