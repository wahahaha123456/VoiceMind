# -*- coding: utf-8 -*-
"""
CER 评测：对比基线 / 微调 在 AISHELL-1 dev 评测集上的识别效果。

用法:
  python compute_cer.py <ref.txt> <hyp1.txt> [<hyp2.txt> ...]
  （每个文件都是 uid<TAB>text 格式）
"""
import os
import re
import sys

PUNCT_RE = re.compile(r"[。？！，、；：\s\.,?!;:·\-—_\"'“”‘’（）()\[\]【】<>《》~`|/\\]")


def load(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            uid, _, text = line.partition("\t")
            d[uid.strip()] = text.strip()
    return d


def norm(s):
    """去掉标点/空白，英文转小写；中文按字比较。"""
    return PUNCT_RE.sub("", s).lower()


def edit_ops(ref, hyp):
    """返回 (距离, 正确/替换/删除/插入 计数)。"""
    m, n = len(ref), len(hyp)
    # dp 只保留上一行
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def cer_of(ref_raw, hyp_raw):
    ref, hyp = norm(ref_raw), norm(hyp_raw)
    if not ref:
        return 0.0 if not hyp else 1.0, 0
    return edit_ops(ref, hyp) / len(ref), len(ref)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    ref_path, hyps = sys.argv[1], sys.argv[2:]
    ref = load(ref_path)
    results = {}
    for hp in hyps:
        hyp = load(hp)
        tot_err = tot_len = 0
        exact = 0
        missing = 0
        per = {}
        for uid, r in ref.items():
            if uid not in hyp:
                missing += 1
                continue
            h = hyp[uid]
            err, length = cer_of(r, h)
            tot_err += err * length
            tot_len += length
            if norm(r) == norm(h):
                exact += 1
            per[uid] = (err, h)
        n = len(ref) - missing
        results[hp] = {
            "cer": tot_err / max(1, tot_len),
            "sent_acc": exact / max(1, n),
            "n": n,
            "missing": missing,
            "per": per,
        }

    print("=" * 72)
    print(f"参考文本: {ref_path}   评测条数: {len(ref)}")
    print("=" * 72)
    print(f"{'解码结果文件':<44}{'CER':>9}{'整句正确率':>12}")
    for hp in hyps:
        r = results[hp]
        print(f"{os.path.basename(hp):<44}{r['cer']:>8.2%}{r['sent_acc']:>11.2%}"
              f"   (n={r['n']}{', 缺失%d' % r['missing'] if r['missing'] else ''})")

    # 多份结果时给出逐条对比
    if len(hyps) == 2:
        a, b = hyps
        improved = worsened = same = 0
        samples_imp, samples_wor = [], []
        for uid, r in ref.items():
            if uid not in results[a]["per"] or uid not in results[b]["per"]:
                continue
            ca = results[a]["per"][uid][0]
            cb = results[b]["per"][uid][0]
            if cb < ca - 1e-9:
                improved += 1
                if len(samples_imp) < 5:
                    samples_imp.append((uid, r, results[a]["per"][uid][1], results[b]["per"][uid][1], ca, cb))
            elif cb > ca + 1e-9:
                worsened += 1
                if len(samples_wor) < 5:
                    samples_wor.append((uid, r, results[a]["per"][uid][1], results[b]["per"][uid][1], ca, cb))
            else:
                same += 1
        print("-" * 72)
        print(f"逐条变化: 变好 {improved} 条 / 变差 {worsened} 条 / 持平 {same} 条")
        d = results[b]["cer"] - results[a]["cer"]
        rel = d / results[a]["cer"] * 100 if results[a]["cer"] else 0
        print(f"CER 变化: {results[a]['cer']:.2%} -> {results[b]['cer']:.2%}  "
              f"({'+' if d>=0 else ''}{d:.2%}, 相对 {rel:+.1f}%)")
        label_a = os.path.basename(a).replace("output_", "").replace(".txt", "")
        label_b = os.path.basename(b).replace("output_", "").replace(".txt", "")

        def show(title, items):
            if not items:
                return
            print("-" * 72)
            print(title)
            for uid, r, ta, tb, ca, cb in items:
                print(f"  {uid}  CER {ca:.0%} -> {cb:.0%}")
                print(f"    标注  : {r}")
                print(f"    {label_a:<6}: {ta}")
                print(f"    {label_b:<6}: {tb}")
        show("【变好的例子】", samples_imp)
        show("【变差的例子】", samples_wor)


if __name__ == "__main__":
    main()
