# -*- coding: utf-8 -*-
"""
语音识别结果综合评测（CER / 整句正确率 / 标点率 / 逐条样例）

为什么不用 compute-wer：
  本机没有 Kaldi 的 compute-wer，也没有 whisper_mix_normalize.py。
  这里用与上一轮完全一致的归一化口径（去标点+去空白+小写），
  保证新旧结果可直接比较。

用法：
  python eval_report.py --ref <参考> --hyp <系统输出> [--hyp2 <另一系统>] [--name 场景名] [--samples 8]
"""
import os
import re
import sys
import argparse


PUNCT = re.compile(r"[，。？！、；：,.?!;:·\-—_\"'“”‘’（）()\[\]【】<>《》~`|/\\\s]")
CJK_PUNCT = re.compile(r"[，。？！、；：]")
NUM_CHAR = re.compile(r"[零一二三四五六七八九十百千万亿两]")
DIGIT = re.compile(r"\d")


def norm(s):
    return PUNCT.sub("", s).lower()


def edit_distance(a, b):
    """字符级编辑距离（O(len(a)*len(b))，句子短，够用）"""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def load_tsv(path):
    d = {}
    order = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            u, _, t = line.partition("\t")
            u = u.strip()
            d[u] = t.strip()
            order.append(u)
    return d, order


def score(ref, hyp, order):
    """返回 (总错字, 总字数, 错句数, 句数, 逐条列表)"""
    tot_err = tot_len = bad_sent = 0
    rows = []
    for u in order:
        if u not in ref:
            continue
        r = norm(ref[u])
        h = norm(hyp.get(u, ""))
        if not r:
            continue
        e = edit_distance(r, h)
        tot_err += e
        tot_len += len(r)
        bad_sent += 1 if e > 0 else 0
        rows.append((u, ref[u], hyp.get(u, ""), e, len(r)))
    n = len(rows)
    return tot_err, tot_len, bad_sent, n, rows


def punct_rate(d, order):
    if not order:
        return 0.0
    k = sum(1 for u in order if u in d and CJK_PUNCT.search(d[u]))
    return k / len(order) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--hyp2", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--samples", type=int, default=0, help="打印多少条最差样例")
    args = ap.parse_args()

    ref, order = load_tsv(args.ref)
    h1, _ = load_tsv(args.hyp)
    h2, _ = load_tsv(args.hyp2) if args.hyp2 else ({}, [])

    order = [u for u in order if u in h1]

    print("=" * 78)
    print("场景: %s   评测条数: %d" % (args.name or os.path.basename(args.ref), len(order)))
    print("=" * 78)

    e1, l1, b1, n1, rows1 = score(ref, h1, order)
    print("%-22s CER=%6.2f%%  整句正确=%6.2f%%  标点率=%5.1f%%"
          % (os.path.basename(args.hyp), e1 / max(1, l1) * 100,
             (n1 - b1) / max(1, n1) * 100, punct_rate(h1, order)))

    rows2 = None
    if h2:
        e2, l2, b2, n2, rows2 = score(ref, h2, order)
        print("%-22s CER=%6.2f%%  整句正确=%6.2f%%  标点率=%5.1f%%"
              % (os.path.basename(args.hyp2), e2 / max(1, l2) * 100,
                 (n2 - b2) / max(1, n2) * 100, punct_rate(h2, order)))
        print("-" * 78)
        d = (e2 - e1)
        rel = d / max(1e-9, e1) * 100
        print("CER 变化: %+.2f 个百分点 (相对 %+.1f%%)   错字 %d -> %d"
              % ((e2 / max(1, l2) - e1 / max(1, l1)) * 100, rel, e1, e2))
        print("整句正确率变化: %+.2f 个百分点" % ((n2 - b2) / max(1, n2) * 100 - (n1 - b1) / max(1, n1) * 100))
        # 逐条胜负
        m2 = {u: e for u, _, _, e, _ in rows2}
        better = sum(1 for u, _, _, e, _ in rows1 if m2.get(u, e) < e)
        worse = sum(1 for u, _, _, e, _ in rows1 if m2.get(u, e) > e)
        same = n1 - better - worse
        print("逐条: 变好 %d / 变差 %d / 持平 %d" % (better, worse, same))

    if args.samples:
        print("\n--- 变差样例（前 %d 条）---" % args.samples)
        if rows2:
            m2 = {u: e for u, _, _, e, _ in rows2}
            delta = sorted(rows1, key=lambda x: -(m2.get(x[0], x[3]) - x[3]))
            for u, r, hh, e, L in delta[:args.samples]:
                if m2.get(u, e) <= e:
                    break
                print("  [%s]" % u)
                print("    参考: %s" % r)
                print("    前  : %s" % hh)
                print("    后  : %s" % h2.get(u, ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
