"""早期的最小验证脚本：批量识别 audio/ 目录下的音频并打印结果。

项目结构：<项目根>/code/test.py、<项目根>/audio/
注意：扫描目录以 __file__ 为基准，因此在任意工作目录下运行都能找到音频。
"""
import glob
import os

from funasr import AutoModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../code
PROJECT_ROOT = os.path.dirname(BASE_DIR)                     # 项目根目录
AUDIO_DIR = os.path.join(PROJECT_ROOT, "audio")

model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    disable_update=True
)

# 自动获取 audio/ 下的所有音频文件
audio_files = sorted(
    p for ext in ("*.wav", "*.mp3", "*.m4a") for p in glob.glob(os.path.join(AUDIO_DIR, ext))
)
print(f"音频目录：{AUDIO_DIR}")
print(f"找到 {len(audio_files)} 个音频文件")

if not audio_files:
    raise SystemExit("没有可识别的音频文件，请把音频放入 audio/ 目录")

res = model.generate(input=audio_files)

for item in res:
    print(f"{item.get('key', '未知')}: {item['text']}")
