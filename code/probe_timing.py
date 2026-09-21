# -*- coding: utf-8 -*-
"""探测字幕时间轴 vs 实际语音活动：定位"没说话也在显示字幕"的原因。"""
import re
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

from app import _generate_full  # noqa: E402

VIDEOS = [
    "video/VID_20240106_173605.mp4",
    "video/VID_20231014_180338.mp4",
]


def silencedetect(path):
    """返回 [(sil_start, sil_end)]（秒），-35dB / ≥0.6s 视为静音。"""
    out = subprocess.run(
        ["ffmpeg", "-i", path, "-af", "silencedetect=noise=-35dB:d=0.6", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    spans, cur = [], None
    for line in out.splitlines():
        m = re.search(r"silence_start: ([\d.]+)", line)
        if m:
            cur = float(m.group(1))
        m = re.search(r"silence_end: ([\d.]+)", line)
        if m and cur is not None:
            spans.append((cur, float(m.group(1))))
            cur = None
    return spans


for vid in VIDEOS:
    print("=" * 60)
    print(vid)
    sil = silencedetect(vid)
    print("静音区间(秒):", [(round(a, 1), round(b, 1)) for a, b in sil])

    wav = vid + ".probe16k.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid,
                    "-vn", "-ac", "1", "-ar", "16000", wav])
    try:
        text, segs = _generate_full(wav)
        print("分段数:", len(segs))
        for s in segs:
            print(f"  [{s['start']/1000:7.2f} -> {s['end']/1000:7.2f}] {s['text'][:30]}")
    finally:
        os.remove(wav)
