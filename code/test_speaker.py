# -*- coding: utf-8 -*-
"""
说话人分离测试：CAM++ 区分不同说话人。

用法: python test_speaker.py [音频路径，默认 audio/test.wav]
输出: 逐句带说话人标签的转写 + 说话人数量统计

说明：
- funasr 开 spk_model 后，结果里每句带 sentence_info[{text, start, end, spk}]
- spk 是聚类编号（0/1/2...），编号本身没有"谁是 1 号"的含义，
  准确性需要人工核对：同一人说的话是否标了同一个编号
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

audio_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "audio", "test.wav")
if not os.path.exists(audio_file):
    print(f"找不到音频: {audio_file}")
    sys.exit(1)

from funasr import AutoModel

print("=" * 60)
print("说话人分离测试")
print("=" * 60)
print("加载模型中（SenseVoiceSmall + fsmn-vad + ct-punc + cam++）...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
    disable_update=True,
)

start = time.time()
res = model.generate(input=audio_file, batch_size_s=300)
cost = time.time() - start

print(f"\n识别结果（{cost:.2f}秒）：")
sents = (res[0].get("sentence_info") or []) if res else []
for s in sents:
    who = s.get("spk", "?")
    st, ed = s.get("start", 0) / 1000, s.get("end", 0) / 1000
    print(f"[说话人{who}] {st:6.1f}s - {ed:6.1f}s : {s.get('text', '')}")

speakers = sorted({s.get("spk", "?") for s in sents})
per = {k: sum(1 for s in sents if s.get("spk") == k) for k in speakers}
print(f"\n检测到 {len(speakers)} 个说话人: {speakers}")
print(f"各说话人句数: {per}")
print("请人工核对：同一人是否始终对应同一编号（编号本身不代表第几个人）。")
