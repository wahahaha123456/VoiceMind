# -*- coding: utf-8 -*-
"""
方案 A 的可行性预检（不改任何模型，只读）：
  1) 现有 app 用的 SenseVoiceSmall 数字/标点风格是什么样（决定微调标签该往哪边对齐）
  2) ct-punc 能否批量给 AISHELL 参考文本补标点（决定"标注约定修正"能不能自动化）

用法：python plan_checks.py
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
LIST_DIR = os.path.join(PROJECT_ROOT, "dataset", "list")


def load_tsv(path):
    d = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line:
            u, _, t = line.partition("\t")
            d.append((u.strip(), t.strip()))
    return d


def main():
    from funasr import AutoModel

    wavs = load_tsv(os.path.join(LIST_DIR, "eval_wav.scp"))
    refs = dict(load_tsv(os.path.join(LIST_DIR, "eval_text.txt")))

    # ---- 1) SenseVoiceSmall（app 当前实际使用的模型）的输出风格 ----
    print("=" * 70)
    print("【1】app 当前模型 SenseVoiceSmall 的数字/标点风格")
    sv = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                   punc_model="ct-punc", disable_update=True)
    sample = wavs[:3]
    for uid, path in sample:
        res = sv.generate(input=path, cache={}, language="auto", use_itn=True,
                          batch_size_s=60, merge_vad=True, merge_length_s=15)
        txt = res[0]["text"] if res else ""
        print(f"  参考: {refs.get(uid,'')}")
        print(f"  SV  : {txt}")
        print()

    # ---- 2) ct-punc 批量补标点（用于修正 AISHELL 标注约定）----
    print("=" * 70)
    print("【2】ct-punc 批量给 AISHELL 参考文本补标点")
    punc = AutoModel(model="ct-punc", disable_update=True)
    texts = [refs[u] for u, _ in wavs[:6]]
    res = punc.generate(input=texts, batch_size=16)
    outs = [r.get("text", "") for r in res]
    for t, o in zip(texts, outs):
        print(f"  原: {t}")
        print(f"  后: {o}")
    print(f"\n  批量调用返回条数 {len(outs)} / 输入 {len(texts)}"
          f"  -> {'批量可用' if len(outs) == len(texts) else '批量异常，需逐条'}")
    print(f"  补标点成功率: {sum(1 for o in outs if any(c in o for c in '。，、？！'))}/{len(outs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
