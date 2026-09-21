"""WebSocket 流式识别测试客户端（对应 ws_server.py :8001）

用法：python test_ws.py [音频文件]
音频需为 16k 单声道 wav；只给文件名时优先到 ../audio 下查找。
默认样本 test_16k.wav 不存在时，会用 ffmpeg 从 audio/test.wav 自动转一份到系统临时目录。
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np
import websockets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../code
AUDIO_DIR = os.path.join(os.path.dirname(BASE_DIR), "audio")

CHUNK_SAMPLES = 9600  # 600ms @16k
WS_URL = "ws://127.0.0.1:8001"


def resolve_path(name: str) -> str:
    """只给文件名时优先到 audio/ 下找，便于脚本移动后仍能直接运行。"""
    if os.path.isabs(name) or os.path.dirname(name):
        return name
    cand = os.path.join(AUDIO_DIR, name)
    return cand if os.path.exists(cand) else name


def ensure_pcm_wav(path: str) -> str:
    """
    默认样本 test_16k.wav 缺失时自动补一份：
    用 ffmpeg 把 audio/test.wav 转成 16k 单声道，输出到系统临时目录（不弄脏项目）。
    其它路径缺失则原样返回，由调用方给出友好提示。
    """
    if os.path.exists(path) or os.path.basename(path) != "test_16k.wav":
        return path

    src = os.path.join(AUDIO_DIR, "test.wav")
    if not os.path.exists(src):
        return path

    ffmpeg = shutil.which("ffmpeg") or r"C:\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"
    dst = os.path.join(tempfile.gettempdir(), "funasr_test_16k.wav")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-ar", "16000", "-ac", "1", dst],
            check=True, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        print(f">>> 已自动生成测试音频：{dst}")
        return dst
    except Exception as e:
        print(f">>> 自动转换测试音频失败（{e}），请手动指定音频文件路径")
        return path


def load_pcm16(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate != 16000:
        n_out = int(len(audio) * 16000 / rate)
        idx = np.clip((np.arange(n_out) * rate / 16000).astype(int), 0, len(audio) - 1)
        audio = audio[idx]
    return audio


async def main(path: str):
    audio = load_pcm16(path)
    async with websockets.connect(WS_URL, max_size=10 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "mode": "2pass", "wav_name": "test", "wav_format": "pcm",
            "is_speaking": True, "chunk_size": [0, 10, 5]
        }))
        print(f">>> 已连接 {WS_URL}，流式发送 {len(audio)} 采样（{len(audio)/16000:.1f}s）")

        done = asyncio.Event()
        ready = asyncio.Event()      # 服务端模型就绪信号（首次连接要加载，约 1 分钟）

        async def sender():
            # 先等服务端 ready：模型未就绪就推流会让音频积压、结果严重滞后
            try:
                await asyncio.wait_for(ready.wait(), timeout=300)
            except asyncio.TimeoutError:
                print("（等待服务端模型就绪超时）")
                return
            for i in range(0, len(audio), CHUNK_SAMPLES):
                await ws.send(audio[i:i + CHUNK_SAMPLES].tobytes())
                await asyncio.sleep(0.3)  # 模拟实时语速
            await ws.send(json.dumps({"is_speaking": False}))
            print(">>> 已发送结束标志")

        async def receiver():
            try:
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get("type") == "status":
                        print(f"<<< 状态: {data.get('message')}")
                    elif data.get("type") == "ready":
                        print("<<< 模型已就绪，开始发送音频")
                        ready.set()
                    elif data.get("type") == "partial":
                        print(f"<<< 增量: {data['text']}")
                    elif data.get("type") == "final":
                        print(f"<<< 最终: {data['text']}")
                        done.set()
                        return
                    elif data.get("type") == "error":
                        print(f"<<< 错误: {data['message']}")
                        done.set()
                        return
            except websockets.ConnectionClosed:
                done.set()

        recv = asyncio.create_task(receiver())
        send = asyncio.create_task(sender())
        await send
        try:
            await asyncio.wait_for(done.wait(), timeout=60)
        except asyncio.TimeoutError:
            print("（等待最终结果超时）")
        recv.cancel()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "test_16k.wav"
    target = ensure_pcm_wav(resolve_path(arg))
    if not os.path.exists(target):
        print(f"✗ 找不到音频文件：{target}")
        print("  用法：python test_ws.py [16k单声道wav路径]")
        sys.exit(1)
    asyncio.run(main(target))
