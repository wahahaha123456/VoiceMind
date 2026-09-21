# -*- coding: utf-8 -*-
"""
从项目内音视频 + 系统自产 SRT 构建迷你微调数据集（仅用于跑通训练流程）。

流程：
  1. 解析 audio/ 与 video/ 对应的 SRT 字幕（处理 BOM）
  2. 把相邻字幕条合并成 3~30 秒的句段（太短训不稳，太长显存吃紧）
  3. 用 ffmpeg 切出 16kHz 单声道 wav
  4. 生成 train_wav.scp / train_text.txt / val_wav.scp / val_text.txt
     （之后用 scp2jsonl.py 转成 ChatML jsonl）

用法：
  funasr_env/Scripts/python.exe finetune/build_data.py
"""
import os
import re
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_WAV_DIR = os.path.join(PROJECT_ROOT, "finetune", "data", "wav")
OUT_DIR = os.path.join(PROJECT_ROOT, "finetune", "data")

# (音频源文件, 对应 SRT) —— 都是系统此前识别自产的数据，仅用于流程验证
PAIRS = [
    (os.path.join(PROJECT_ROOT, "audio", "20260904_223022.m4a"),
     os.path.join(PROJECT_ROOT, "subtitle", "20260904_223022.srt")),
    (os.path.join(PROJECT_ROOT, "audio", "20260909_225539.m4a"),
     os.path.join(PROJECT_ROOT, "subtitle", "20260909_225539.srt")),
    (os.path.join(PROJECT_ROOT, "audio", "20260912_121741.m4a"),
     os.path.join(PROJECT_ROOT, "subtitle", "20260912_121741.srt")),
    (os.path.join(PROJECT_ROOT, "video", "VID_20231014_180338.mp4"),
     os.path.join(PROJECT_ROOT, "subtitle", "VID_20231014_180338.srt")),
]

MIN_DUR = 3.0    # 句段最短时长（秒）
MAX_DUR = 30.0   # 句段最长时长（秒）
VAL_EVERY = 8    # 每 8 条取 1 条进验证集

TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def _to_sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path):
    """解析 SRT，返回 [(start_sec, end_sec, text), ...]"""
    with open(path, "r", encoding="utf-8-sig") as f:   # utf-8-sig 自动去 BOM
        content = f.read()
    blocks = re.split(r"\n\s*\n", content.strip())
    cues = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        m = TIME_RE.search(lines[1] if not TIME_RE.search(lines[0]) else lines[0])
        if not m:
            continue
        g = m.groups()
        start = _to_sec(*g[:4])
        end = _to_sec(*g[4:])
        text = " ".join(ln.strip() for ln in lines[2:] if ln.strip())
        if text:
            cues.append((start, end, text))
    return cues


def merge_cues(cues):
    """把相邻字幕合并成 MIN_DUR~MAX_DUR 的句段"""
    utterances = []
    cur_start, cur_end, cur_text = None, None, ""
    for start, end, text in cues:
        if cur_start is None:
            cur_start, cur_end, cur_text = start, end, text
            continue
        # 已达标且再合并会超上限 → 先收当前句段
        if (cur_end - cur_start) >= MIN_DUR and (end - cur_start) > MAX_DUR:
            utterances.append((cur_start, cur_end, cur_text))
            cur_start, cur_end, cur_text = start, end, text
            continue
        cur_end = max(cur_end, end)
        cur_text += text
    if cur_start is not None and (cur_end - cur_start) >= 1.0:
        utterances.append((cur_start, cur_end, cur_text))
    return utterances


def cut_wav(src, start, end, dst):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", dst,
    ]
    subprocess.run(cmd, check=True)


def main():
    os.makedirs(OUT_WAV_DIR, exist_ok=True)
    all_utts = []   # (utt_id, wav_path, text)

    for src, srt in PAIRS:
        if not os.path.exists(src) or not os.path.exists(srt):
            print(f"[跳过] 缺文件: {src} 或 {srt}")
            continue
        stem = os.path.splitext(os.path.basename(src))[0]
        cues = parse_srt(srt)
        utts = merge_cues(cues)
        print(f"[{stem}] 字幕 {len(cues)} 条 → 合并为 {len(utts)} 个句段")
        for i, (start, end, text) in enumerate(utts):
            utt_id = f"{stem}_{i:03d}"
            wav_path = os.path.join(OUT_WAV_DIR, utt_id + ".wav").replace("\\", "/")
            try:
                cut_wav(src, start, end, wav_path)
                all_utts.append((utt_id, wav_path, text))
            except subprocess.CalledProcessError as e:
                print(f"  [失败] {utt_id}: {e}")

    # 打乱并切分 train/val
    import random
    random.seed(42)
    random.shuffle(all_utts)
    val = [u for i, u in enumerate(all_utts) if i % VAL_EVERY == 0]
    train = [u for i, u in enumerate(all_utts) if i % VAL_EVERY != 0]

    for name, items in (("train", train), ("val", val)):
        with open(os.path.join(OUT_DIR, f"{name}_wav.scp"), "w", encoding="utf-8") as f1, \
             open(os.path.join(OUT_DIR, f"{name}_text.txt"), "w", encoding="utf-8") as f2:
            for utt_id, wav_path, text in items:
                f1.write(f"{utt_id} {wav_path}\n")
                f2.write(f"{utt_id} {text}\n")

    import soundfile as sf
    total_sec = sum(sf.info(w).duration for _, w, _ in all_utts) if all_utts else 0
    print(f"\n完成：共 {len(all_utts)} 条（train {len(train)} / val {len(val)}），"
          f"约 {total_sec/60:.1f} 分钟音频")
    print(f"wav 目录: {OUT_WAV_DIR}")
    print(f"清单目录: {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
