# -*- coding: utf-8 -*-
"""
鲁棒性探针的交叉验证：把"数字书写习惯类"样本剔除后，再算一次 CER。

为什么要做这一步：上一次 AISHELL 微调就吃过这个亏——CER 的显著下降里 97% 来自
"学会把 1.115 写成一点一一五"这种标注书写约定，而不是识别能力。扰动条件下
CER 的变化同样可能被这类样本主导，所以必须拆开看。

用法：python verify_robustness.py
"""
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
LIST_DIR = os.path.join(PROJECT_ROOT, "dataset", "list")
ROB_DIR = os.path.join(SCRIPT_DIR, "robustness")

CONDITIONS = ["clean", "slow", "fast", "pitch", "noise"]

PUNCT_RE = re.compile(r"[。？！，、；：\s\.,?!;:·\-—_\"'“”‘’（）()\[\]【】<>《》~`|/\\]")
NUM_CHAR = re.compile(r"[零一二三四五六七八九十百千万亿两]")
DIGIT = re.compile(r"\d")


def norm(s):
    return PUNCT_RE.sub("", s).lower()


def dist(a, b):
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def load_tsv(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line:
            u, _, t = line.partition("\t")
            d[u.strip()] = t.strip()
    return d


def cer(ref, hyp, uids):
    e = t = 0
    for u in uids:
        rn = norm(ref[u])
        e += dist(rn, norm(hyp.get(u, "")))
        t += max(1, len(rn))
    return e / max(1, t)


def main():
    ref = load_tsv(os.path.join(LIST_DIR, "eval_text.txt"))
    hyps = {}
    for tag in ("baseline", "finetuned"):
        hyps[tag] = {c: load_tsv(os.path.join(ROB_DIR, f"hyp_{tag}_{c}.tsv"))
                     for c in CONDITIONS}

    uids = sorted(hyps["baseline"]["clean"])
    # 数字书写习惯类：参考含中文数字 且 基线原始输出含阿拉伯数字
    num_ids = [u for u in uids
               if NUM_CHAR.search(norm(ref[u]))
               and DIGIT.search(norm(hyps["baseline"]["clean"].get(u, "")))]
    other_ids = [u for u in uids if u not in set(num_ids)]

    print(f"抽样 {len(uids)} 条，其中『数字书写习惯类』{len(num_ids)} 条，其余 {len(other_ids)} 条")
    print(f"被剔除的样本: {', '.join(num_ids) if num_ids else '（无）'}\n")

    print("=" * 84)
    print(f"{'条件':<8}{'基线CER':>10}{'微调CER':>10}{'相对':>9}   |"
          f"{'剔除后·基线':>13}{'剔除后·微调':>13}{'相对':>9}")
    print("-" * 84)
    rows = {}
    for c in CONDITIONS:
        b = cer(ref, hyps["baseline"][c], uids)
        f = cer(ref, hyps["finetuned"][c], uids)
        bo = cer(ref, hyps["baseline"][c], other_ids) if other_ids else float("nan")
        fo = cer(ref, hyps["finetuned"][c], other_ids) if other_ids else float("nan")
        rows[c] = (b, f, bo, fo)
        print(f"{c:<8}{b:>9.2%}{f:>10.2%}{(f-b)/max(b,1e-9)*100:>8.0f}%   |"
              f"{bo:>12.2%}{fo:>13.2%}{(fo-bo)/max(bo,1e-9)*100:>8.0f}%")

    print("\n【相对 clean 的退化倍数（剔除数字书写类后）】")
    for c in CONDITIONS:
        bo, fo = rows[c][2], rows[c][3]
        print(f"  {c:<6}  基线 ×{bo/rows['clean'][2]:.2f}   微调 ×{fo/rows['clean'][3]:.2f}")


if __name__ == "__main__":
    sys.exit(main())
