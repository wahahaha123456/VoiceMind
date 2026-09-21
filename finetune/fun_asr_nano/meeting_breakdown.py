# -*- coding: utf-8 -*-
"""会议集退化归因：按参考句长/说话人分桶，看 v2 相对基线的退化集中在哪。"""
import re, sys, os

LIST_DIR = r"C:\Users\iiiis\Desktop\FunASR\dataset\list"
PUNCT = re.compile(r"[，。？！、；：,.?!;:·\-—_\"'“”‘’（）()\[\]【】<>《》~`|/\\\s]")


def norm(s):
    return PUNCT.sub("", s).lower()


def dist(a, b):
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[n]


def load(p):
    d = {}
    for l in open(p, encoding="utf-8"):
        if l.strip():
            u, _, t = l.rstrip("\n").partition("\t")
            d[u.strip()] = t.strip()
    return d


def bucket_cer(rows, key_fn, label):
    """rows: list of (u, L, eb, ev)"""
    groups = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    print("--- %s ---" % label)
    print("%-14s %6s %10s %10s %10s" % ("组", "条数", "基线错字", "v2错字", "CER变化"))
    for k in sorted(groups):
        g = groups[k]
        tb = sum(x[2] for x in g)
        tv = sum(x[3] for x in g)
        tl = sum(x[1] for x in g)
        print("%-14s %6d %10d %10d %+9.2fpp" % (
            k, len(g), tb, tv, (tv - tb) / max(1, tl) * 100))
    print()


def main():
    ref = load(os.path.join(LIST_DIR, "meeting_eval1500_text.txt"))
    base = load(os.path.join(LIST_DIR, "meeting_baseline.txt"))
    v2 = load(os.path.join(LIST_DIR, "meeting_v2.txt"))
    rows = []
    for u, r in ref.items():
        if u not in base or u not in v2:
            continue
        rn = norm(r)
        if not rn:
            continue
        rows.append((u, len(rn), dist(rn, norm(base.get(u, ""))),
                     dist(rn, norm(v2.get(u, "")))))
    print("有效 %d 条\n" % len(rows))

    bucket_cer(rows, lambda r: "短(<12字)" if r[1] < 12 else ("中(12-25)" if r[1] <= 25 else "长(>25字)"),
               "按参考句长")
    # 按说话人（uid 里 S00000 是说话人分段编号，前面是录音 ID）
    bucket_cer(rows, lambda r: r[0].split("_")[1][:9], "按录音（前 9 个）")


if __name__ == "__main__":
    sys.exit(main())
