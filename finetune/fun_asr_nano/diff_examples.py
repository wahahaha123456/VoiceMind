# -*- coding: utf-8 -*-
"""
训练前后逐条差异对比：给出变好/变差/持平统计，并按类别抽取典型例子。

用法：
  python diff_examples.py <参考文本> <微调前输出> <微调后输出> [每类例子数]

输出：统计表 + 分类例子（数字书写类 / 真实识别改善 / 真实识别退步 / 标点变化）
"""
import re
import sys

PUNCT_RE = re.compile(r"[。？！，、；：\s\.,?!;:·\-—_\"'“”‘’（）()\[\]【】<>《》~`|/\\]")
NUM_CHAR = re.compile(r"[零一二三四五六七八九十百千万亿两]")
DIGIT = re.compile(r"\d")
PUNCT_ANY = re.compile(r"[。，、？！；：]")


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


def load(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        uid, _, t = line.partition("\t")
        d[uid.strip()] = t.strip()
    return d


def main():
    ref = load(sys.argv[1])
    before = load(sys.argv[2])
    after = load(sys.argv[3])
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 6

    better, worse, same = [], [], []
    for uid, r in ref.items():
        rn = norm(r)
        eb, ea = dist(rn, norm(before.get(uid, ""))), dist(rn, norm(after.get(uid, "")))
        rec = (uid, r, before.get(uid, ""), after.get(uid, ""), eb, ea, len(rn))
        if ea < eb:
            better.append(rec)
        elif ea > eb:
            worse.append(rec)
        else:
            same.append(rec)

    total = len(ref)
    print("=" * 78)
    print(f"总样本 {total}")
    print(f"  变好 {len(better):>5} 条 ({len(better)/total:6.1%})")
    print(f"  变差 {len(worse):>5} 条 ({len(worse)/total:6.1%})")
    print(f"  持平 {len(same):>5} 条 ({len(same)/total:6.1%})")

    # 按"是否涉及数字书写习惯"切开
    num_ids = {u for u, r in ref.items()
               if NUM_CHAR.search(norm(r)) and DIGIT.search(norm(before.get(u, "")))}
    print(f"\n其中『数字书写习惯』相关样本 {len(num_ids)} 条，"
          f"占全部变好样本的 {sum(1 for x in better if x[0] in num_ids)/max(1,len(better)):.0%}")

    # 错字数来源拆分：CER 的降幅到底来自哪一部分
    ebn = efn = lbn = eb = ef = lb = 0
    for uid, r in ref.items():
        rn = norm(r)
        x, y = dist(rn, norm(before.get(uid, ""))), dist(rn, norm(after.get(uid, "")))
        if uid in num_ids:
            ebn += x; efn += y; lbn += len(rn)
        else:
            eb += x; ef += y; lb += len(rn)
    net = (ebn + eb) - (efn + ef)
    print(f"\n逐字错总计: {ebn+eb} -> {efn+ef} 字（净减 {net} 字）")
    print(f"  ├ 数字书写类 {len(num_ids):>4} 条: {ebn:>4} -> {efn:>4} 字"
          f"（净减 {ebn-efn:>4} 字，占净减 {100*(ebn-efn)/max(1,net):.0f}%）"
          f"  字错率 {ebn/max(1,lbn):.2%} → {efn/max(1,lbn):.2%}")
    print(f"  └ 真实识别类 {len(ref)-len(num_ids):>4} 条: {eb:>4} -> {ef:>4} 字"
          f"（净变 {ef-eb:+d} 字）"
          f"  字错率 {eb/max(1,lb):.2%} → {ef/max(1,lb):.2%}")

    def show(title, recs, limit=None):
        print("\n" + "-" * 78)
        print(f"【{title}】")
        for uid, r, b, a, eb, ea, ln in (recs[:limit] if limit else recs):
            print(f"  {uid}  ({eb}字错 → {ea}字错 / {ln}字)")
            print(f"    参考  : {r}")
            print(f"    改前  : {b}")
            print(f"    改后  : {a}")

    show("变好 · 真实识别（非数字书写）", [x for x in better if x[0] not in num_ids], k)
    show("变好 · 数字书写习惯（标注约定类）", [x for x in better if x[0] in num_ids], 3)
    show("变差 · 真实退步", worse, k)

    pb = sum(1 for t in before.values() if PUNCT_ANY.search(t))
    pa = sum(1 for t in after.values() if PUNCT_ANY.search(t))
    print("\n" + "-" * 78)
    print(f"【标点】含标点的输出：改前 {pb}/{len(before)} ({pb/len(before):.1%})"
          f"  →  改后 {pa}/{len(after)} ({pa/len(after):.1%})")
    print(f"  参考文本（人工标注）含标点：{sum(1 for t in ref.values() if PUNCT_ANY.search(t))}/{len(ref)}")


if __name__ == "__main__":
    main()
