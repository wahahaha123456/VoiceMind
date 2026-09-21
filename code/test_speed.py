# -*- coding: utf-8 -*-
"""
处理速度测试（RTF）：处理耗时 / 音频时长，越小于 1 越快于实时。

用法: python test_speed.py
输出: 屏幕打印 + docs/speed_test.txt

说明：
- 音频时长用 ffprobe 读取（soundfile 不支持 m4a，ffprobe 什么格式都能读）
- 模型用 SenseVoiceSmall + fsmn-vad + ct-punc（与线上服务兜底链路一致）
- 只测 audio/ 下前 5 个文件，控制总耗时
"""
import glob
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))          # .../code
ROOT = os.path.dirname(BASE)                               # 项目根
OUT = os.path.join(ROOT, "docs", "speed_test.txt")


def ffprobe_duration(path: str) -> float:
    """用 ffprobe 读音频时长（秒），失败返回 0。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return float(r.stdout.strip())
    except Exception as e:
        print(f"  [警告] ffprobe 读时长失败: {e}")
        return 0.0


def main():
    files = sorted(
        glob.glob(os.path.join(ROOT, "audio", "*.wav"))
        + glob.glob(os.path.join(ROOT, "audio", "*.m4a"))
        + glob.glob(os.path.join(ROOT, "audio", "*.mp3"))
    )
    if not files:
        print("audio/ 目录下没有音频文件")
        sys.exit(1)

    import torch
    print("=" * 60)
    print("处理速度测试（RTF）")
    print("=" * 60)
    print(f"设备: {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    from funasr import AutoModel
    print("加载模型中（SenseVoiceSmall + fsmn-vad + ct-punc）...")
    t0 = time.time()
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        disable_update=True,
    )
    print(f"模型加载完成（{time.time() - t0:.1f}s）\n")

    results = []
    for filepath in files[:5]:
        name = os.path.basename(filepath)
        duration = ffprobe_duration(filepath)
        if duration <= 0:
            print(f"{name}: 读不到时长，跳过")
            continue
        # 预热：第一次调用含 VAD/标点模型加载，单独跑一次不计入
        if not results:
            model.generate(input=filepath)

        start = time.time()
        res = model.generate(input=filepath)
        cost = time.time() - start
        rtf = cost / duration
        text = (res[0].get("text", "") or "")[:24]
        results.append({"file": name, "duration": duration, "cost": cost, "rtf": rtf})
        print(f"{name}: 时长{duration:.1f}s, 处理{cost:.2f}s, RTF={rtf:.3f}  | {text}")

    if not results:
        print("没有可测文件")
        sys.exit(1)
    avg = sum(r["rtf"] for r in results) / len(results)
    avg_speed = sum(r["duration"] for r in results) / sum(r["cost"] for r in results)
    print(f"\n平均RTF: {avg:.3f}  （即 1 秒音频平均处理 {1 / avg:.1f} 秒能跑完 → 约 {avg_speed:.1f} 倍速）")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("处理速度测试结果（RTF = 处理耗时 / 音频时长）\n")
        f.write("=" * 50 + "\n")
        for r in results:
            f.write(f"{r['file']}: 时长{r['duration']:.1f}s, 处理{r['cost']:.2f}s, RTF={r['rtf']:.3f}\n")
        f.write(f"\n平均RTF: {avg:.3f}\n")
    print(f"结果已保存到 {OUT}")


if __name__ == "__main__":
    main()
