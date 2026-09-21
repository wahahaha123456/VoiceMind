# -*- coding: utf-8 -*-
"""
微调增益归因分析：区分"真实识别改善"与"适配了 AISHELL 标注书写习惯"。

AISHELL-1 标注把数字写成汉字（一点一一五亿元 / 二零一一年），基座模型习惯输出
阿拉伯数字（1.115亿元 / 2011年）。微调后模型学会了标注方的书写习惯，这类"改善"
在 CER 上很显眼，但属于标注约定而非识别能力——必须单独拆出来，否则会高估效果。

另外统计标点丢失情况：AISHELL 标注本身无标点，微调后模型会倾向于不再输出标点。
"""
import re
import sys

PUNCT_RE = re.compile(r"[。？！，、；：\s\.,?!;:·\-—_\"'“”‘’（）()\[\]【】<>《》~`|/\\]")
NUM_CHAR = re.compile(r"[零一二三四五六七八九十百千万亿两]")
DIGIT = re.compile(r"\d")
PUNCT_ANY = re.compile(r"[。，、？！；：]")


def load(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        uid, _, t = line.partition("\t")
        d[uid.strip()] = t.strip()
    return d


def norm(s):
    return PUNCT_RE.sub("", s).lower()


def dist(ref, hyp):
    m, n = len(ref), len(hyp)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[n]


def cer(r, h):
    r, h = norm(r), norm(h)
    return dist(r, h) / max(1, len(r))


def main():
    ref = load(sys.argv[1])
    base = load(sys.argv[2])
    ft = load(sys.argv[3])

    # 判定"数字书写习惯"类样本：标注含中文数字，且基线输出含阿拉伯数字
    num_style = set()
    for uid, r in ref.items():
        b = base.get(uid, "")
        if NUM_CHAR.search(norm(r)) and DIGIT.search(norm(b)):
            num_style.add(uid)
    print(f"总样本 {len(ref)}，其中'数字书写习惯'相关 {len(num_style)} 条\n")

    def agg(ids):
        eb = ef = ln = 0
        for uid in ids:
            r = ref[uid]
            ln += len(norm(r))
            eb += cer(r, base.get(uid, "")) * len(norm(r))
            ef += cer(r, ft.get(uid, "")) * len(norm(r))
        return eb / max(1, ln), ef / max(1, ln), len(ids)

    b1, f1, n1 = agg(list(ref))
    b2, f2, n2 = agg(list(num_style))
    other = [u for u in ref if u not in num_style]
    b3, f3, n3 = agg(other)

    print(f"{'子集':<26}{'条数':>7}{'微调前CER':>12}{'微调后CER':>12}{'相对变化':>12}")
    print(f"{'全部':<26}{n1:>7}{b1:>11.2%}{f1:>12.2%}{(f1-b1)/b1*100:>11.1f}%")
    print(f"{'数字书写习惯相关':<26}{n2:>7}{b2:>11.2%}{f2:>12.2%}{(f2-b2)/b2*100:>11.1f}%")
    print(f"{'其余（真实识别）':<26}{n3:>7}{b3:>11.2%}{f3:>12.2%}{(f3-b3)/b3*100:>11.1f}%")

    # 标点输出比例
    pb = sum(1 for t in base.values() if PUNCT_ANY.search(t)) / max(1, len(base))
    pf = sum(1 for t in ft.values() if PUNCT_ANY.search(t)) / max(1, len(ft))
    print(f"\n输出含标点的句子比例:  微调前 {pb:.1%}  ->  微调后 {pf:.1%}")
    print("（AISHELL 标注本身不含标点，模型会把这一习惯学过去；")
    print("  本 CER 已归一化去标点，故不受影响，但实际使用需另接标点模型）")


if __name__ == "__main__":
    main()
