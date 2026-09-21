"""
独立 WebSocket 流式识别服务（端口 8001）
协议：
  客户端 -> 服务端：{"is_speaking": true, ...} 配置 → 二进制 PCM16 音频 → {"is_speaking": false} 结束
  服务端 -> 客户端：{"type":"partial","text":...} 增量结果 / {"type":"final","text":...} 完整结果 / {"type":"error",...}
"""
import asyncio
import json
import threading
import time

import numpy as np
import websockets

# chunk 大小：600ms @ 16kHz = 9600 采样点，对应 chunk_size=[0,10,5]
CHUNK_STRIDE = 9600
CHUNK_SIZE = [0, 10, 5]
ENCODER_LOOK_BACK = 4
DECODER_LOOK_BACK = 1

# ==================== 模型按需加载 ====================
# 原先在模块顶层同步加载 881MB 的流式模型 + 标点模型，导致端口十几秒到一分钟
# 才就绪，启动器迟迟不开浏览器。改成首个 WebSocket 连接时才加载：
# 端口 1~2 秒可用，用户点"开始流式识别"时再等模型（前端有状态提示）。
online_model = None
punc_model = None
_punc_tried = False
_model_lock = threading.Lock()


def get_models():
    """加载流式识别模型与标点模型（只加载一次，首次连接时调用）。"""
    global online_model, punc_model, _punc_tried
    if online_model is None:
        with _model_lock:
            if online_model is None:
                from funasr import AutoModel      # 延迟导入：连带的 torch 很慢
                t0 = time.time()
                print("正在加载流式识别模型...")
                online_model = AutoModel(
                    model="paraformer-zh-streaming",
                    device="cpu",
                    disable_update=True
                )
                print(f"流式识别模型加载完成，耗时 {time.time() - t0:.1f} 秒")

                # 标点模型（离线通用标点，用于最终结果润色）
                print("正在加载标点模型...")
                try:
                    punc_model = AutoModel(model="ct-punc", disable_update=True)
                    print("标点模型加载完成！")
                except Exception as e:
                    punc_model = None
                    print(f"标点模型加载失败（不影响基本识别）: {e}")
                _punc_tried = True
    elif punc_model is None and not _punc_tried:
        # 标点模型上次加载失败时允许再试一次
        with _model_lock:
            if punc_model is None and not _punc_tried:
                _punc_tried = True
                try:
                    from funasr import AutoModel
                    punc_model = AutoModel(model="ct-punc", disable_update=True)
                    print("标点模型加载完成！")
                except Exception as e:
                    print(f"标点模型加载失败（不影响基本识别）: {e}")


def add_punc(text: str) -> str:
    """给最终结果加标点（只在结束时调用，避免拖慢流式输出）"""
    if not punc_model or not text:
        return text
    try:
        res = punc_model.generate(input=text)
        if res and res[0].get("text"):
            return res[0]["text"]
    except Exception as e:
        print(f"标点恢复失败: {e}")
    return text


async def handle_connection(websocket):
    """处理单个 WebSocket 连接"""
    print("客户端已连接")
    loop = asyncio.get_event_loop()

    # 首次连接才加载模型（阻塞在连接协程里，不会拖慢服务端口就绪）
    try:
        await websocket.send(json.dumps(
            {"type": "status", "message": "首次连接需加载流式模型（约 1 分钟），请稍候…"},
            ensure_ascii=False))
        await loop.run_in_executor(None, get_models)
        # 通知客户端可以开始推流（模型就绪前推流会让字幕严重滞后）
        await websocket.send(json.dumps({"type": "ready"}, ensure_ascii=False))
    except websockets.exceptions.ConnectionClosed:
        # 用户等不及直接关掉了页面
        print("客户端在模型加载完成前断开")
        return

    cache = {}
    audio_buffer = np.array([], dtype=np.float32)
    full_pieces = []          # 累积所有增量文本，用于输出完整结果

    def infer(chunk, is_final):
        """流式推理（CPU 上是阻塞的，放线程池里跑，避免卡住事件循环）"""
        res = online_model.generate(
            input=chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=CHUNK_SIZE,
            encoder_chunk_look_back=ENCODER_LOOK_BACK,
            decoder_chunk_look_back=DECODER_LOOK_BACK
        )
        return (res[0].get("text", "") if res and res[0] else "")

    try:
        async for message in websocket:
            # ---------- 文本控制消息 ----------
            if isinstance(message, str):
                try:
                    config = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if config.get("is_speaking") is False:
                    # 说话结束：冲刷剩余不足一个 chunk 的音频，输出完整结果
                    tail = audio_buffer
                    audio_buffer = np.array([], dtype=np.float32)
                    if len(tail) > 0:
                        text = await loop.run_in_executor(None, infer, tail, True)
                        if text and text.strip():   # 同样过滤纯空白结果
                            full_pieces.append(text)
                    full_text = add_punc("".join(full_pieces))
                    await websocket.send(json.dumps(
                        {"type": "final", "text": full_text}, ensure_ascii=False))
                    # 重置，支持同一连接连续说多段
                    cache = {}
                    full_pieces = []
                continue

            # ---------- 二进制音频（PCM16） ----------
            if isinstance(message, bytes):
                pcm = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, pcm])

                # 累积够一个 chunk 就推理一次
                while len(audio_buffer) >= CHUNK_STRIDE:
                    chunk = audio_buffer[:CHUNK_STRIDE]
                    audio_buffer = audio_buffer[CHUNK_STRIDE:]
                    text = await loop.run_in_executor(None, infer, chunk, False)
                    # 关键：流式模型对静音/噪点段偶尔返回纯空白文本（如 " "），
                    # 不过滤的话前端会为它建一行，只剩时间戳没有内容（空行 bug）
                    if text and text.strip():
                        full_pieces.append(text)
                        await websocket.send(json.dumps(
                            {"type": "partial", "text": text}, ensure_ascii=False))

    except websockets.exceptions.ConnectionClosed:
        print("客户端断开连接")
    except Exception as e:
        print(f"处理连接时出错: {e}")
        try:
            await websocket.send(json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False))
        except Exception:
            pass


async def main():
    async with websockets.serve(
        handle_connection,
        "0.0.0.0",
        8001,
        max_size=10 * 1024 * 1024,   # 单条消息最大 10MB
        ping_interval=20,
        ping_timeout=20
    ):
        print("WebSocket 服务已启动: ws://0.0.0.0:8001")
        await asyncio.Future()       # 永久运行


if __name__ == "__main__":
    asyncio.run(main())
