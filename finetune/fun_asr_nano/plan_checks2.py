# -*- coding: utf-8 -*-
"""
方案 A 补充预检：
  A) 用 app.py 完全一致的调用参数跑 SenseVoice，确认标点输出形态
  B) ct-punc 不支持批量（inference 里 assert len(data_in)==1），
     所以测一下"逐条循环"的吞吐，用来估算给 1.2 万条参考文本补标点要多久
"""
import os
import re
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
LIST_DIR = os.path.join(PROJECT_ROOT, "dataset", "list")

_RICH_TAG_RE = re.compile(r"<\|[^|]*\|>")


def load_tsv(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line:
            u, _, t = line.partition("\t")
            d[u.strip()] = t.strip()
    return d


def main():
    from funasr import AutoModel

    wavs = load_tsv(os.path.join(LIST_DIR, "eval_wav.scp"))
    refs = load_tsv(os.path.join(LIST_DIR, "eval_text.txt"))
    uids = list(wavs)

    print("=" * 70)
    print("【A】app.py 口径的 SenseVoice 输出（get_base_model().generate(input=..., sentence_timestamp=True, batch_size_s=300)）")
    sv = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                   punc_model="ct-punc", disable_update=True)
    for uid in uids[:4]:
        res = sv.generate(input=wavs[uid], sentence_timestamp=True, batch_size_s=300)
        txt = _RICH_TAG_RE.sub("", res[0].get("text", "")).strip()
        print(f"  参考: {refs.get(uid, '')}")
        print(f"  输出: {txt}")

    print("\n" + "=" * 70)
    print("【B】ct-punc 逐条循环吞吐（用于估算补标点耗时）")
    punc = AutoModel(model="ct-punc", disable_update=True)
    texts = [refs[u] for u in uids[:100]]
    t0 = time.time()
    outs = []
    for t in texts:
        try:
            r = punc.generate(input=t)
            outs.append(r[0].get("text", "") if r else "")
        except Exception as e:
            outs.append("")
    dt = time.time() - t0
    ok = sum(1 for o in outs if any(c in o for c in "。，、？！"))
    per = dt / len(texts)
    print(f"  {len(texts)} 条耗时 {dt:.1f}s  ({per*1000:.0f} ms/条, {len(texts)/dt:.1f} 条/秒)")
    print(f"  补上标点的比例: {ok}/{len(texts)}")
    print(f"  => 1.2 万条训练文本预计需要 {12000*per/60:.1f} 分钟")
    print("\n  样例：")
    for t, o in list(zip(texts, outs))[:5]:
        print(f"    原: {t}")
        print(f"    后: {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
