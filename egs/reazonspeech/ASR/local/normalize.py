import re
import num2words


PUNCTUATIONS = {ord(x): " " for x in "、。「」『』，,？！!?⁉‼［］〈〉・♪《》−．“”◆）≪"}
ZENKAKU = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９"
HANKAKU = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
ZEN2HAN = str.maketrans(ZENKAKU, HANKAKU)


def normalize(s: str) -> str:
    """
    Convert full-width characters to half-width, and remove punctuations.
    :param s: str, input string.
    :return: str, normalized string.
    """
    s = s.translate(PUNCTUATIONS).translate(ZEN2HAN)
    conv = lambda m: num2words.num2words(m.group(0), lang="ja")
    return re.sub(r"\d+\.?\d*", conv, s)
