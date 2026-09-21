# -*- coding: utf-8 -*-
"""
鲁棒性探针：量化模型在"非标准发音/非理想声学条件"下的退化程度。

动机
----
AISHELL-1 是标准普通话朗读语料，用它微调出来的模型在**规范发音**上表现好，
但真实使用中用户抱怨"发音不标准时识别略差"。CER 的单一数字看不出这个差异，
所以这里做一次受控探针：把同一批音频施加几类扰动（语速、音高、噪声），
在同一套参考文本上分别算 CER，得到"条件 × 模型"的退化矩阵。

扰动选择（均为 ffmpeg 纯 DSP，不引入外部数据，可复现）：
  clean  原始音频（对照组）
  slow   语速 ×0.85（atempo，音高不变）——模拟拖音/迟缓
  fast   语速 ×1.15（atempo，音高不变）——模拟语速快/连读
  pitch  音高 ×0.90（asetrate 降调 + atempo 恢复时长）——模拟不同音色/发声习惯
  noise  叠加白噪（约 15~20dB SNR）——模拟非理想录音环境

用法（在 fun_asr_nano 目录下执行，model.py 需在 cwd）：
  基线:  python robustness_probe.py ++base_model="<基座目录>"
  微调:  python robustness_probe.py ++base_model="<合并后目录>"

结果：控制台表格 + robustness/report_<标签>.json
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

# 脚本自身的目录（model.py 就在这里，AutoModel 用相对路径找它）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))      # .../FunASR
LIST_DIR = os.path.join(PROJECT_ROOT, "dataset", "list")
OUT_DIR = os.path.join(SCRIPT_DIR, "robustness")

SAMPLE_N = 80          # 抽样条数：够看出方向性差异，又不至于解码太久

CONDITIONS = ["clean", "slow", "fast", "pitch", "noise"]

# ---------------------------------------------------------------- CER 工具
PUNCT_RE = re.compile(r"[。？！，、；：\s\.,?!;:·\-—_\"'“”‘’（）()\[\]【】<>《》~`|/\\]")


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


def load_tsv(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        uid, _, t = line.partition("\t")
        d[uid.strip()] = t.strip()
    return d


def score(ref, hyp):
    """返回 (加权CER, 整句正确率)"""
    err = tot = sent_ok = 0
    for uid, r in ref.items():
        rn = norm(r)
        hn = norm(hyp.get(uid, ""))
        err += dist(rn, hn)
        tot += max(1, len(rn))
        if rn == hn:
            sent_ok += 1
    return err / max(1, tot), sent_ok / max(1, len(ref))


# ---------------------------------------------------------------- 扰动生成
def run_ffmpeg(args):
    return subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args,
                          capture_output=True, text=True, errors="ignore")


def make_perturbed(src, dst, cond):
    """按条件生成扰动音频；失败返回 False（该条跳过，不污染统计口径）"""
    if cond == "clean":
        shutil.copyfile(src, dst)
        return True
    if cond == "slow":
        f = "atempo=0.85"
    elif cond == "fast":
        f = "atempo=1.15"
    elif cond == "pitch":
        # asetrate 降调同时拖慢，再用 atempo 把时长拉回来 → 只改音高
        f = "asetrate=16000*0.90,aresample=16000,atempo=1.111111"
    elif cond == "noise":
        # amix 混入白噪（输入语音幅度约 0.05~0.1，噪声 0.008 约合 15~20dB SNR）
        f = None
    else:
        return False
    if cond == "noise":
        r = run_ffmpeg(["-i", src, "-f", "lavfi", "-i",
                        "anoisesrc=color=white:amplitude=0.008:sample_rate=16000",
                        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0",
                        "-ar", "16000", "-ac", "1", dst])
    else:
        r = run_ffmpeg(["-i", src, "-filter:a", f, "-ar", "16000", "-ac", "1", dst])
    return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1000


def build_conditions(pairs):
    """为一个 uid→wav 的列表生成全部扰动版本，返回 {cond: scp_path}"""
    os.makedirs(OUT_DIR, exist_ok=True)
    scps = {}
    for cond in CONDITIONS:
        cdir = os.path.join(OUT_DIR, cond)
        os.makedirs(cdir, exist_ok=True)
        scp_path = os.path.join(OUT_DIR, f"{cond}.scp")
        lines, skipped = [], []
        for uid, wav in pairs:
            dst = os.path.join(cdir, uid + ".wav")
            if not (os.path.exists(dst) and os.path.getsize(dst) > 1000):
                if not make_perturbed(wav, dst, cond):
                    skipped.append(uid)
                    continue
            lines.append(f"{uid}\t{dst}")
        with open(scp_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[perturb] {cond:<6} 生成 {len(lines)} 条"
              + (f"（{len(skipped)} 条失败，已剔除）" if skipped else ""), flush=True)
        scps[cond] = scp_path
    return scps


# ---------------------------------------------------------------- 主流程
@hydra.main(config_name=None, version_base=None)
def main(cfg: DictConfig):
    kw = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, (DictConfig, ListConfig)) else dict(cfg)
    tag = kw.get("tag", "baseline")
    base_model = kw.get("base_model")
    if not base_model:
        print("必须传 ++base_model=<模型目录>", flush=True)
        return 1

    # 1) 参考文本 + 所有评测音频
    ref_all = load_tsv(os.path.join(LIST_DIR, "eval_text.txt"))
    wav_all = load_tsv(os.path.join(LIST_DIR, "eval_wav.scp"))

    # 2) 均匀抽样（跨说话人分布，避免只取前若干条造成偏差）
    uids = [u for u in ref_all if u in wav_all]
    uids.sort()
    total_n = len(uids)
    step = max(1, total_n // SAMPLE_N)
    uids = uids[::step][:SAMPLE_N]
    ref = {u: ref_all[u] for u in uids}
    pairs = [(u, wav_all[u]) for u in uids]
    print(f"[probe] 抽样 {len(pairs)} 条（可用全集 {total_n} 条，步长 {step}）\n", flush=True)

    # 3) 生成扰动
    scps = build_conditions(pairs)

    # 4) 加载模型并逐条件解码
    import torch
    from funasr import AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = AutoModel(model=base_model, trust_remote_code=True,
                      remote_code="./model.py", device=device)
    print(f"\n[probe] 模型加载 {time.time()-t0:.1f}s  设备={device}  标签={tag}\n", flush=True)

    report = {"tag": tag, "model": base_model, "n_sample": len(pairs), "conditions": {}}
    for cond in CONDITIONS:
        items = []
        for line in open(scps[cond], encoding="utf-8"):
            if line.strip():
                uid, p = line.rstrip("\n").split("\t", 1)
                items.append((uid, p))
        hyp = {}
        t1 = time.time()
        for i in range(0, len(items), 16):
            grp = items[i:i + 16]
            try:
                res = model.generate(input=[p for _, p in grp], cache={}, batch_size=len(grp))
                texts = [r.get("text", "") for r in res]
            except Exception as e:
                print(f"[probe] {cond} 批处理降级逐条: {type(e).__name__}: {e}", flush=True)
                texts = []
                for _, p in grp:
                    try:
                        texts.append(model.generate(input=[p], cache={}, batch_size=1)[0].get("text", ""))
                    except Exception:
                        texts.append("")
            for (uid, _), t in zip(grp, texts):
                hyp[uid] = t
        c, acc = score(ref, hyp)
        report["conditions"][cond] = {"cer": c, "sent_acc": acc,
                                      "n": len(hyp), "sec": time.time() - t1}
        # 落盘逐条输出：事后才能做"剔除数字书写类样本"的交叉验证，
        # 否则无法区分"声学鲁棒性真提升"与"又学会了标注书写习惯"。
        with open(os.path.join(OUT_DIR, f"hyp_{tag}_{cond}.tsv"), "w", encoding="utf-8") as fh:
            for u in sorted(hyp):
                fh.write(f"{u}\t{hyp[u]}\n")
        print(f"[probe] {cond:<6} CER={c:7.2%}  整句={acc:6.2%}  ({len(hyp)} 条, {time.time()-t1:.0f}s)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"report_{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n==== 退化汇总（相对 clean 的 CER 倍数）====")
    base_cer = report["conditions"]["clean"]["cer"]
    for cond in CONDITIONS:
        c = report["conditions"][cond]["cer"]
        print(f"  {cond:<6} CER={c:7.2%}   ×{c/max(base_cer,1e-9):.2f}")
    print(f"\n[probe] 报告已写入 {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
