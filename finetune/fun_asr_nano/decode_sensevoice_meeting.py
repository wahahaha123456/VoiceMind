# -*- coding: utf-8 -*-
"""测 SenseVoiceSmall(app 当前在用的模型)在会议测试集上的 CER,补齐三方对比。"""
import sys, time
sys.path.insert(0, r"C:\Users\iiiis\Desktop\FunASR\code")

LIST_DIR = r"C:\Users\iiiis\Desktop\FunASR\dataset\list"
OUT = r"C:\Users\iiiis\Desktop\FunASR\dataset\list\meeting_sensevoice.txt"

def main():
    from funasr import AutoModel
    t0 = time.time()
    model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                      punc_model="ct-punc", disable_update=True)
    print("[sv] 模型加载 %.0fs" % (time.time() - t0), flush=True)

    pairs = []
    for line in open(LIST_DIR + r"\meeting_eval1500_wav.scp", encoding="utf-8"):
        if line.strip():
            u, _, p = line.rstrip("\n").partition("\t")
            pairs.append((u.strip(), p.strip()))
    print("[sv] 待解码 %d 条" % len(pairs), flush=True)

    t1 = time.time()
    with open(OUT, "w", encoding="utf-8") as f:
        for i, (u, p) in enumerate(pairs, 1):
            try:
                res = model.generate(input=p, cache={}, language="zh",
                                     use_itn=True, batch_size_s=60)
                text = res[0]["text"] if res else ""
            except Exception as e:
                print("[sv] %s 失败: %s" % (u, e), flush=True)
                text = ""
            f.write("%s\t%s\n" % (u, text))
            f.flush()
            if i % 200 == 0:
                print("[sv] %d/%d  (%.0fs, %.1f 条/秒)" % (
                    i, len(pairs), time.time() - t1, i / (time.time() - t1)), flush=True)
    print("[sv] 完成,总耗时 %.1f 分钟 -> %s" % ((time.time() - t1) / 60, OUT), flush=True)

if __name__ == "__main__":
    main()
