from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel          # /punctuate 的请求体模型（PunctuateRequest）依赖它
from rapidfuzz import fuzz
import asyncio
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np

# 注意：`from funasr import AutoModel` 刻意不在模块顶层导入。
# funasr 会连带加载 torch（数秒~十几秒），放顶层会让端口晚十秒才就绪；
# 改为在 get_base_model / get_spk_model / get_stream_model 内部按需导入，
# 启动只等 fastapi/uvicorn，端口 2~3 秒即可用（模型本身仍是首次调用时才加载）。

# ---------------------------------------------------------------- 路径配置
# 项目结构：<项目根>/code/app.py 与 <项目根>/web/index.html
# 用 __file__ 定位，保证从任何工作目录启动都能找到前端文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../code
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # 项目根目录
WEB_DIR = os.path.join(PROJECT_ROOT, "web")             # 前端页面目录
TEMP_DIR = tempfile.gettempdir()                        # 上传临时文件放系统临时目录，不弄脏项目

app = FastAPI(title="VoiceMind 语音智能助手", description="语音识别 + RAG 知识库问答，支持说话人分离与字幕生成")

# 允许跨域（便于本地直接打开 html 或其他前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 模型懒加载 ====================
# 设计：启动只绑定端口（几秒），模型在首次用到时才加载（30~90 秒）。
# 前端通过 /health 感知加载状态，并在页面打开时调 /warmup 后台预热，
# 用户挑文件的这几秒就是模型的加载时间，往往点"开始识别"时已就绪。
_model_cache = {}
_model_loading = {"base": False, "spk": False}
_model_lock = threading.Lock()   # 防止"预热线程"与"识别请求"同时加载同一模型


# ==================== 识别模型选择 ====================
# 默认用本地微调的 Fun-ASR-Nano v2（真实会议/嘈杂场景 CER 7.1%，
# 对比 SenseVoice 的 10.3%，相对改善约 31%；评测见 finetune/微调说明_v2.md）。
# 如需回退旧模型，启动前设环境变量 ASR_MODEL=sensevoice
_ASR_MODEL_V2_DIR = r"C:\Users\iiiis\Desktop\FunASR\finetune\fun_asr_nano\outputs_v2_merged"
if os.environ.get("ASR_MODEL", "v2").lower() in ("sensevoice", "v1"):
    ASR_MODEL_NAME = "iic/SenseVoiceSmall"
else:
    ASR_MODEL_NAME = _ASR_MODEL_V2_DIR


def get_base_model():
    """基础模型：识别 + VAD + 标点（首次调用识别/检索接口时加载）"""
    if "base" not in _model_cache:
        with _model_lock:                    # 双检锁：并发时只加载一次，其余线程等待
            if "base" not in _model_cache:
                print("[模型] 加载基础模型（识别+VAD+标点）: %s" % ASR_MODEL_NAME)
                t0 = time.time()
                _model_loading["base"] = True
                try:
                    from funasr import AutoModel   # 延迟导入：连带的 torch 加载很慢
                    _model_cache["base"] = AutoModel(
                        model=ASR_MODEL_NAME,
                        vad_model="fsmn-vad",
                        punc_model="ct-punc",
                        disable_update=True
                    )
                    print(f"[模型] 基础模型加载完成，耗时 {time.time() - t0:.1f} 秒")
                finally:
                    _model_loading["base"] = False
    return _model_cache["base"]


def get_spk_model():
    """说话人分离模型：基础模型 + CAM++（首次调用 /asr_spk 时加载）"""
    if "spk" not in _model_cache:
        with _model_lock:
            if "spk" not in _model_cache:
                print("[模型] 加载说话人分离模型（基础+CAM++）...")
                t0 = time.time()
                _model_loading["spk"] = True
                try:
                    from funasr import AutoModel
                    _model_cache["spk"] = AutoModel(
                        model="iic/SenseVoiceSmall",
                        vad_model="fsmn-vad",
                        punc_model="ct-punc",
                        spk_model="cam++",
                        disable_update=True
                    )
                    print(f"[模型] 说话人分离模型加载完成，耗时 {time.time() - t0:.1f} 秒")
                finally:
                    _model_loading["spk"] = False
    return _model_cache["spk"]


# 流式识别模型（懒加载，首次 WebSocket 连接时加载）+ 全局推理锁（模型非线程安全）
stream_model = None
stream_lock = threading.Lock()

STREAM_CHUNK_SIZE = [0, 10, 5]            # [历史, 当前, 未来]，600ms 粒度
STREAM_ENC_LOOK_BACK = 4
STREAM_DEC_LOOK_BACK = 1


def get_stream_model():
    global stream_model
    if stream_model is None:
        print("正在加载流式识别模型（paraformer-zh-streaming）...")
        from funasr import AutoModel   # 延迟导入（与 app.py 的其他模型加载一致）
        stream_model = AutoModel(model="paraformer-zh-streaming", disable_update=True)
        print("流式识别模型加载完成！")
    return stream_model


def _stream_generate(audio: np.ndarray, cache: dict, is_final: bool):
    """流式推理（在锁内串行执行，返回本次增量文本）"""
    with stream_lock:
        res = stream_model.generate(
            input=audio,
            cache=cache,
            is_final=is_final,
            chunk_size=STREAM_CHUNK_SIZE,
            encoder_chunk_look_back=STREAM_ENC_LOOK_BACK,
            decoder_chunk_look_back=STREAM_DEC_LOOK_BACK,
        )
    text = ""
    if res and isinstance(res, list) and len(res) > 0:
        text = (res[0] or {}).get("text", "")
    return text


@app.websocket("/ws/asr")
async def ws_asr(websocket: WebSocket):
    """
    WebSocket 流式识别：
    - 客户端先发一条 JSON 配置（可空），随后持续发送 16k 单声道 PCM16 二进制
    - 服务端每收到一段就做一次流式推理，回送 {"code":0,"text":"增量文本"}
    - 客户端发 {"is_final":true} 结束，服务端冲刷剩余结果并回 {"code":0,"final":true}
    """
    await websocket.accept()
    get_stream_model()
    cache = {}
    loop = asyncio.get_event_loop()
    print("WebSocket 客户端已连接")
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("bytes"):
                pcm16 = np.frombuffer(msg["bytes"], dtype=np.int16)
                audio = pcm16.astype(np.float32) / 32768.0
                text = await loop.run_in_executor(None, _stream_generate, audio, cache, False)
                if text:
                    await websocket.send_text(json.dumps({"code": 0, "text": text}, ensure_ascii=False))

            elif msg.get("text"):
                cfg = json.loads(msg["text"])
                if cfg.get("is_final"):
                    # 冲刷结尾（空音频 + is_final）
                    text = await loop.run_in_executor(
                        None, _stream_generate, np.zeros(0, dtype=np.float32), cache, True
                    )
                    if text:
                        await websocket.send_text(json.dumps({"code": 0, "text": text}, ensure_ascii=False))
                    await websocket.send_text(json.dumps({"code": 0, "final": True}, ensure_ascii=False))
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket 异常: {e}")
    finally:
        print("WebSocket 客户端已断开")


def _index_response():
    """
    返回前端页面。
    说明：浏览器只在 https 或 localhost / 127.0.0.1 这类"可信来源"下才允许
    使用麦克风，file:// 双击打开会被直接拒绝，因此页面必须由本服务托管。
    """
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "VoiceMind 服务已启动", "docs": "/docs"})


@app.get("/")
async def root():
    """根路径返回中文前端页面"""
    return _index_response()


@app.get("/health")
async def health():
    """健康检查：返回已加载/正在加载的模型，前端据此显示"模型加载中/已就绪"

    注意：服务启动即监听端口（几秒），模型改成首次使用时才加载（30~90 秒），
    所以端口可访问 ≠ 模型已就绪，必须看 loaded_models / loading。
    """
    # 知识库/大模型状态也一并返回：前端"语音问答"页据此显示"已就绪 / 未配置 Key"。
    # 注意 kb_chunks 只在引擎已加载时统计——没加载就不去读库，避免为了看一眼
    # 数字而白白加载几百 MB 向量模型。
    kb_chunks = 0
    if rag_is_ready():
        try:
            kb_chunks = get_rag_engine().count()
        except Exception:
            kb_chunks = 0
    try:
        llm_ready = get_agent().has_key()
    except Exception:
        llm_ready = False

    return {
        "status": "ok",
        "loaded_models": list(_model_cache.keys()),
        "loading": [k for k, v in _model_loading.items() if v],
        "base_ready": "base" in _model_cache,
        "rag_ready": rag_is_ready(),      # 向量模型是否已加载
        "kb_chunks": kb_chunks,           # 知识库块数
        "llm_ready": llm_ready,           # 是否已配置大模型 Key
    }


@app.post("/warmup")
async def warmup():
    """后台预热基础模型：前端页面打开时调用，让首次识别少等一会儿。

    在独立线程里加载模型，立即返回——不阻塞事件循环（否则 /health 会失去响应）；
    与识别请求共用 _model_lock，不会重复加载。
    """
    if "base" in _model_cache:
        return {"code": 0, "status": "ready"}
    if not _model_loading["base"]:
        def _load():
            try:
                get_base_model()
            except Exception as e:          # 预热失败不影响正常请求（请求时会再试）
                print(f"[模型] 预热失败：{type(e).__name__}: {e}")

        _model_loading["base"] = True
        threading.Thread(target=_load, daemon=True, name="model-warmup").start()
    return {"code": 0, "status": "loading"}


# ==================== 标点恢复（实时字幕用） ====================
# 场景：实时录音时，句子由前端按停顿断行，流式增量本身不带标点。
# 前端把定稿的行 POST 到这里，用独立标点模型补全标点后原位更新。
_punc_model = None


def get_punc_model():
    """独立标点模型（懒加载，与基础模型里的 punc_model 互不影响）。"""
    global _punc_model
    if _punc_model is None:
        from funasr import AutoModel
        _punc_model = AutoModel(model="ct-punc", disable_update=True)
    return _punc_model


class PunctuateRequest(BaseModel):
    text: str = ""


@app.post("/punctuate")
async def punctuate(req: PunctuateRequest):
    """给一段无标点文本补标点。

    失败时原样返回 text（前端可静默降级，不影响字幕展示）。
    """
    text = (req.text or "").strip()
    if not text:
        return {"text": ""}
    try:
        res = get_punc_model().generate(input=text)
        out = res[0].get("text", "") if res and res[0] else ""
        return {"text": out or text}
    except Exception as e:
        return {"text": text, "error": str(e)}


@app.get("/index.html")
async def serve_index():
    """显式支持 http://127.0.0.1:8000/index.html 访问（麦克风需要可信来源）"""
    return _index_response()


@app.get("/audio-processor.js")
async def audio_processor_js():
    """AudioWorklet 处理器文件（浏览器必须能按相对路径加载到它）"""
    path = os.path.join(WEB_DIR, "audio-processor.js")
    if not os.path.exists(path):
        return JSONResponse({"error": "audio-processor.js 不存在"}, status_code=404)
    return FileResponse(path, media_type="application/javascript")


def _save_upload(file: UploadFile) -> str:
    """保存上传文件到系统临时目录，保留原始扩展名（避免 m4a/mp3 内容被存成 .wav 导致解码失败）"""
    ext = os.path.splitext(file.filename)[1].lower() or ".wav"
    temp_filename = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}{ext}")
    content = file.file  # 同步读取，调用方已 await
    with open(temp_filename, "wb") as f:
        f.write(content.read())
    return temp_filename


def _cleanup(temp_filename: str):
    try:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    except OSError:
        pass


@app.post("/asr")
async def recognize(file: UploadFile = File(...), hotword: str = Form("")):
    """
    普通语音识别：上传音频或视频，返回识别文本
    支持音频：wav, mp3, m4a, flac, aac, ogg 等
    支持视频：mp4, mov, avi, mkv, flv, wmv, webm 等
    视频会自动提取音频，并额外返回 segments（带毫秒时间戳的分段）
    hotword：空格/逗号分隔的热词（专有名词），识别后做模糊匹配写法校正
    """
    temp_filename = _save_upload(file)
    conv_file = None
    try:
        is_video = temp_filename.lower().endswith(VIDEO_EXTS)
        if is_video:
            print(f"检测到视频文件，正在提取音频: {file.filename}")
        # 统一转 16k 单声道（-vn 丢弃视频流）；顺带解决 m4a 等格式直接喂模型解码不稳的问题
        src = _to_wav_16k(temp_filename)
        conv_file = src
        text, segments = _generate_full(src, hotword=hotword)
        resp = {
            "code": 0,
            "text": text,
            "filename": file.filename,
            "type": "video" if is_video else "audio",
        }
        if is_video:
            resp["segments"] = segments
        return resp
    except Exception as e:
        return {
            "code": -1,
            "error": str(e),
            "filename": file.filename
        }
    finally:
        _cleanup(temp_filename)
        if conv_file:
            _cleanup(conv_file)


@app.post("/asr_spk")
async def recognize_with_speaker(file: UploadFile = File(...)):
    """
    说话人分离识别：上传音频，返回带说话人标签的分段文本
    适用于多人会议、访谈等场景
    """
    temp_filename = _save_upload(file)
    try:
        spk_model = get_spk_model()
        res = spk_model.generate(input=temp_filename)

        # FunASR 的 spk 结果在 sentence_info 中：每句带 text/start/end/spk
        segments = []
        for item in res:
            for sent in item.get("sentence_info", []):
                segments.append({
                    "speaker": f"说话人{sent.get('spk', '未知')}",
                    "text": sent.get("text", ""),
                    "start": sent.get("start", 0),   # 毫秒
                    "end": sent.get("end", 0),       # 毫秒
                })

        # 兜底：某些版本没有 sentence_info，退回整段文本
        if not segments:
            text = res[0].get("text", "") if res else ""
            segments = [{"speaker": "说话人0", "text": text, "start": 0, "end": 0}]

        return {
            "code": 0,
            "segments": segments,
            "filename": file.filename
        }
    except Exception as e:
        return {
            "code": -1,
            "error": str(e),
            "filename": file.filename
        }
    finally:
        _cleanup(temp_filename)


def _segments_from_result(res) -> list:
    """从 FunASR 结果中提取分段（sentence_info 内含 text/start(毫秒)/end/spk）"""
    segments = []
    for item in res:
        for sent in item.get("sentence_info", []):
            segments.append({
                "speaker": f"说话人{sent.get('spk', '未知')}",
                "text": sent.get("text", ""),
                "start": sent.get("start", 0),   # 毫秒
                "end": sent.get("end", 0),       # 毫秒
            })
    if not segments:
        text = res[0].get("text", "") if res else ""
        segments = [{"speaker": "说话人0", "text": text, "start": 0, "end": 0}]
    return segments


def format_srt_time(ms: int) -> str:
    """把毫秒转换为SRT时间格式：00:00:00,000"""
    total_secs, millis = divmod(int(ms), 1000)
    hours, rem = divmod(total_secs, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _download(content: str, filename: str, media_type: str) -> StreamingResponse:
    """构建文件下载响应（文件名含中文时需 RFC 5987 编码）"""
    from urllib.parse import quote
    ext = os.path.splitext(filename)[1] or ".txt"
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8-sig")),  # 带 BOM，Windows 记事本打开不乱码
        media_type=media_type,
        headers={
            # ASCII 回退名（纯英文）+ RFC 5987 编码的中文文件名
            "Content-Disposition": f"attachment; filename=export_{uuid.uuid4().hex[:8]}{ext}; filename*=UTF-8''{quote(filename)}"
        }
    )


def _ensure_wav(src: str) -> str:
    """非 wav 音频（浏览器录音的 webm/ogg、手机 m4a 等）用 ffmpeg 转 16k 单声道 wav"""
    if src.lower().endswith(".wav"):
        return src
    dst = os.path.splitext(src)[0] + "_conv.wav"
    ffmpeg = shutil.which("ffmpeg") or r"C:\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-ar", "16000", "-ac", "1", dst],
        check=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return dst


@app.post("/asr_stream")
async def recognize_stream(file: UploadFile = File(...)):
    """
    实时识别接口：接收短音频片段（浏览器录音），快速返回识别文本
    """
    temp_filename = _save_upload(file)
    conv_file = None
    try:
        src = _ensure_wav(temp_filename)
        conv_file = src if src != temp_filename else None
        res = get_base_model().generate(input=src)
        text = res[0]["text"] if res else ""
        return {"code": 0, "text": text}
    except Exception as e:
        return {"code": -1, "error": str(e)}
    finally:
        _cleanup(temp_filename)
        if conv_file:
            _cleanup(conv_file)


@app.post("/export_txt")
async def export_txt(file: UploadFile = File(...)):
    """
    识别音频（含说话人分离）并导出为TXT文本文件
    """
    temp_filename = _save_upload(file)
    try:
        spk_model = get_spk_model()
        res = spk_model.generate(input=temp_filename)
        segments = _segments_from_result(res)
        content = "\n".join(f"【{seg['speaker']}】{seg['text']}" for seg in segments)
        return _download(content, "识别记录.txt", "text/plain; charset=utf-8")
    except Exception as e:
        return JSONResponse({"code": -1, "error": str(e)}, status_code=500)
    finally:
        _cleanup(temp_filename)


@app.post("/export_srt")
async def export_srt(file: UploadFile = File(...)):
    """
    识别音频（含说话人分离）并导出为SRT字幕文件
    """
    temp_filename = _save_upload(file)
    try:
        spk_model = get_spk_model()
        res = spk_model.generate(input=temp_filename)
        segments = _segments_from_result(res)
        srt_parts = []
        for i, seg in enumerate(segments, 1):
            srt_parts.append(f"{i}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n[{seg['speaker']}] {seg['text']}\n")
        srt = jianying_safe_srt("\n".join(srt_parts))   # 防剪映自动拉长
        return _download(srt, "字幕.srt", "application/x-subrip")
    except Exception as e:
        return JSONResponse({"code": -1, "error": str(e)}, status_code=500)
    finally:
        _cleanup(temp_filename)


# ================================================================ 关键词检索
# SenseVoice 输出里的富文本标记，如 <|zh|><|NEUTRAL|><|Speech|><|woitn|>
_RICH_TAG_RE = re.compile(r"<\|[^|]*\|>")

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm", ".m4v", ".ts")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma", ".opus")
SEARCH_EXTS = VIDEO_EXTS + AUDIO_EXTS


def _clean_tags(text) -> str:
    return _RICH_TAG_RE.sub("", str(text or "")).strip()


# ================================================================ 字幕工作台
# 热词：FunASR 的 generate(hotword=...) 参数对当前两个模型（Fun-ASR-Nano v2、
# SenseVoiceSmall）实测均不改变输出（参数被忽略，2026-09-19 验证），
# 因此这里改为「识别后模糊匹配写法校正」：在输出文本中找与热词写法相近的
# 片段，替换为热词的标准写法。专有名词场景（人名/品牌/术语）下效果等价于
# 热词偏置——声学识别近似音后，把写法纠正过来。
_HOTWORD_SIM_THRESHOLD = 60   # 3字词错1字=67分可纠正；2字词错1字=50分不纠（防误替换）


def _apply_hotwords(text: str, hotword_str: str, threshold: int = _HOTWORD_SIM_THRESHOLD):
    """热词写法校正。返回 (新文本, 校正记录 [{from,to,similarity}])。

    匹配规则：滑动窗口找与热词相近的片段（窗口长度 L-1/L/L+1），
    相似度 ≥ threshold 且「首字或尾字至少一头与热词一致」才替换——
    这条约束挡住了「的蓝心→蓝心湄」这类把虚词裹进来、留下尾巴字的错配。
    """
    words = [w for w in re.split(r"[\s,，、;；]+", hotword_str or "") if w]
    if not words or not text:
        return text, []
    fixes = []
    result = text
    for w in sorted(set(words), key=len, reverse=True):   # 长词优先，避免短词抢先匹配
        L = len(w)
        if L < 2:
            continue
        out, i = [], 0
        while i < len(result):
            hit = None
            # 1) 精确长度 L 优先：最常见的错字形态是等长替换（蓝心妹→蓝心湄），
            #    若先看 L±1 的高分短窗口，会把"蓝心"匹配走、留下尾巴"妹"
            if i + L <= len(result):
                cand = result[i:i + L]
                if cand == w:
                    hit = (L, cand, 100.0)
                elif not any(ch in _FUNCTION_CHARS and ch not in w for ch in cand):
                    # 窗口裹进了正文虚词（"蓝心的"）→ 误配，跳过（不能 continue，
                    # 那会跳过 while 的 i 推进造成死循环——已经踩过一次）
                    sim = fuzz.ratio(cand, w)
                    if (cand[0] == w[0] or cand[-1] == w[-1]) and sim >= threshold:
                        hit = (L, cand, sim)
                    else:
                        # 拼音层：兜住「一字不差全错但读音相近」的人名/专名
                        # （"九九几"→"周潇济"，字形 0 分、拼音 55 分）
                        py_sim = _pinyin_hit(cand, w)
                        if py_sim is not None:
                            hit = (L, cand, py_sim)
            # 2) 只做等长匹配，不做 L±1：L+1 会吞热词旁的虚词（"我们在剪映里"
            #    →"我们剪映里"），L-1 会吞后面的字（"蓝心的天"→"蓝心湄天"）。
            #    识别错 2 个字以上的片段本来也不该强行纠（误伤风险大于收益）。
            if hit:
                l, cand, sim = hit
                if cand != w:
                    fixes.append({"from": cand, "to": w, "similarity": round(sim, 1)})
                out.append(w)
                i += l
            else:
                out.append(result[i])
                i += 1
        result = "".join(out)
    return result, fixes


# 硬切字幕行时优先在「轻边界字」后断开，避免把"球队"劈成"足/球队"
_SOFT_CUT_CHARS = "的了是这那个在与和或及就把对从被也都还能去会想很"
# 句内字间停顿达到该毫秒数就切开字幕段：长句常横跨多段静音，
# 不切的话"没说话字幕还一直挂着"（用户实测 15 秒长句跨 4 段静音）
_PAUSE_SPLIT_MS = 1000
# 热词窗口含这些虚词字（且热词本身不含该字）时拒绝匹配：
# "蓝心的"→"蓝心湄"这类误配，窗口里裹进了正文虚词
_FUNCTION_CHARS = "的了是那个在与和或也就很吗呢吧啊呀把被对从会想去很"

# ---- 热词拼音层 ------------------------------------------------------------
# 字形模糊匹配的盲区：识别结果与热词一字都不重叠（"九九几"→"周潇济"），
# 字形相似度 0 分。但人名/专名的误识别通常是「音近字错」，所以再加一层
# 拼音匹配：候选窗口与热词读音相近（无调拼音整体相似 + 首或尾音节一致）
# 即纠写。pypinyin 缺失时自动退化为仅字形匹配，不影响启动。
try:
    from functools import lru_cache
    from pypinyin import lazy_pinyin, Style
    _HAS_PINYIN = True
except ImportError:                                  # pragma: no cover
    _HAS_PINYIN = False

_PINYIN_SIM_THRESHOLD = 55   # 音节均分门槛（3字+词）。踩过线：50 分时"好学习"→"周潇齐"
                             # （52.7）误配，目标"九九级→周潇齐"=56.7，只能卡 55
_PINYIN_SIM_THRESHOLD_2 = 80 # 2字词门槛收紧：常见词组同首音节的多，防"剪辑"→"剪映"误配
_SYLLABLE_SIM_MIN = 80       # 首或尾音节相似度须达标，防止同音节堆出的误配
_SYLLABLE_SIM_FLOOR = 25     # 另一端音节的地板分：拒掉"周到的"vs"周潇济"（尾音节0分）

# ASR 高频混淆的声母组（j/q/x、平翘舌、n/l、f/h、d/t）：
# "齐/济/级"同为 j/q/x+i，识别写法互换极常见，字面逐字符比对会漏
_INITIAL_GROUPS = (
    frozenset("jqx"), frozenset("zcs"), frozenset("zhchsh"),
    frozenset("nl"), frozenset("fh"), frozenset("dt"),
)


def _split_syllable(syl: str):
    """拆 (声母, 韵母)。不依赖 pypinyin 的 INITIALS/FINALS（实测 strict=False
    下 INITIALS 对 j/q/x 系音节返回整个音节），手动按 zh/ch/sh 双字符
    前缀 + 单声母字母拆；零声母音节返回 ('', syl)。"""
    if syl[:2] in ("zh", "ch", "sh"):
        return syl[:2], syl[2:]
    if syl and syl[0] in "bpmfdtnlgkhjqxrcyzw":
        return syl[0], syl[1:]
    return "", syl


@lru_cache(maxsize=8192)
def _syllable_score(a: str, b: str) -> float:
    """单音节相似度：韵母 60% + 声母 40%（声母同混淆组记 70 分）。

    韵母为主是因为中文误听大多整音节换写法（ji/qi/zhi…），声母只做
    混淆组内的宽容；这样 "ji"vs"qi"=88 分、"ji"vs"zhou"=60 分，分得开。
    """
    if a == b:
        return 100.0
    ai, af = _split_syllable(a)
    bi, bf = _split_syllable(b)
    if ai == bi:
        ini_score = 100.0
    elif any(ai in g and bi in g for g in _INITIAL_GROUPS):
        ini_score = 70.0
    else:
        ini_score = 0.0
    return 0.6 * fuzz.ratio(af, bf) + 0.4 * ini_score


@lru_cache(maxsize=8192)
def _py_syllables(s: str):
    """无调拼音音节元组。仅对纯汉字串调用（调用方已保证），lru_cache 加速滑窗。"""
    return tuple(lazy_pinyin(s, style=Style.NORMAL, errors="ignore"))


def _is_hanzi(s: str) -> bool:
    return bool(s) and all("\u4e00" <= ch <= "\u9fff" for ch in s)


def _pinyin_hit(cand: str, w: str):
    """拼音层判定：返回相似度分（不达标返回 None）。

    评分用「逐音节相似度的平均值」而不是整串 ratio：整串 ratio 会被
    长音节的高分掩盖（"突然没"vs"偷新娘"整串 52 分擦线过关——突/偷声母
    相同、韵母 u/ou 相近，把完全不搭的后两拍裹进来了）；逐音节平均下
    target"九九级/周潇济"=52.4，误配"突然没/偷新娘"=46，分得开。

    约束防误配（等长纯汉字窗口，音节数相等可逐位对齐）：
    - 音节平均分达标：2字词 ≥80（常见词组极易同首音节，收紧），
      3字+词 ≥50
    - 首音节或尾音节 ≥ _SYLLABLE_SIM_MIN：误识别通常只错 1~2 个音节，
      至少一头咬得住
    - 另一端音节 ≥ _SYLLABLE_SIM_FLOOR：两头都不能完全跑飞，
      拒掉「周到的」vs「周潇济」这类头音节同音的普通词组
    """
    if not (_HAS_PINYIN and _is_hanzi(cand) and _is_hanzi(w) and len(cand) == len(w)):
        return None
    pc, pw = _py_syllables(cand), _py_syllables(w)
    if len(pc) != len(pw) or not pc:
        return None
    sims = [_syllable_score(a, b) for a, b in zip(pc, pw)]
    if len(w) == 2:
        # 2字词：两个音节都要 ≥80。均分会被「首音节全同」抬轿
        # （"心上/新娘"= xin 100 + shang 60 → 均 80 擦线过），但 2 字词
        # 一个音节跑飞基本就是普通词组，不该纠
        if min(sims) < _PINYIN_SIM_THRESHOLD_2:
            return None
        return sum(sims) / len(sims)
    avg = sum(sims) / len(sims)
    if avg < _PINYIN_SIM_THRESHOLD:
        return None
    head, tail = sims[0], sims[-1]
    if max(head, tail) < _SYLLABLE_SIM_MIN:
        return None
    if min(head, tail) < _SYLLABLE_SIM_FLOOR:
        return None
    return avg


def split_subtitle(text: str, max_len: int = 15):
    """把一句话断成多行字幕：先按标点切子句，贪心合并到 ≤max_len 字；
    超长的子句硬切时优先落在轻边界字（的了是这…）之后，断行更自然。
    """
    text = text.strip()
    if len(text) <= max_len:
        return [text] if text else []
    _PUNCT = "。！？!?；;，,、：:"
    clauses = re.findall(r"[^%s]+[%s]?" % (re.escape(_PUNCT), re.escape(_PUNCT)), text)
    lines, current = [], ""
    for c in [c for c in clauses if c]:
        if len(current) + len(c) <= max_len:
            current += c
            continue
        if current:
            lines.append(current)
            current = ""
        while len(c) > max_len:                  # 单个子句超长 → 硬切（带轻边界回退）
            cut_at = max_len
            zone = c[max_len // 2:max_len]
            best = -1
            for i, ch in enumerate(zone):
                if ch in _SOFT_CUT_CHARS:
                    best = max_len // 2 + i
            if best > 0:
                cut_at = best + 1
            lines.append(c[:cut_at])
            c = c[cut_at:]
        current = c
    if current:
        lines.append(current)
    return lines


def generate_srt(segments: list, offset: float = 0.0, max_len: int = 15) -> str:
    """把句级分段生成为 SRT：时间偏移校正 + 断句（每行 ≤ max_len 字）。

    segments: [{text, start, end[, char_times]}]，start/end 毫秒；offset 单位秒（可为负）。
    有 char_times（与 text 逐字对齐的出声时间）时，每行时间 = 行内首字开始
    ~ 末字结束，说完再停留 _LINE_TAIL_MS；没有则退回按字符占比均摊。
    尾部停留不会越过下一行的开始——"没人说话字幕还挂着"就是靠这两条压掉的。
    """
    off_ms = int(round(offset * 1000))
    _LINE_TAIL_MS = 400

    # 第一遍：摊开所有行（带真实/估算时间）
    entries = []   # (start_ms, speech_end_ms, line)
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = max(0, int(seg.get("start", 0) or 0) + off_ms)
        end = max(start + 200, int(seg.get("end", 0) or 0) + off_ms)   # 每条至少 200ms
        lines = split_subtitle(text, max_len)
        if not lines:
            continue
        cts = seg.get("char_times")
        if cts and len(cts) == len(text):
            pos = 0
            for line in lines:
                l0, l1 = pos, pos + len(line)
                pos = l1
                s = max(0, int(cts[l0][0]) + off_ms)
                e = max(s + 200, int(cts[l1 - 1][1]) + off_ms)
                entries.append([s, e, line])
        else:
            # 无字级时间（SenseVoice sentence_info / 兜底单条）：按字符占比均摊
            total = sum(len(l) for l in lines)
            pos, dur, consumed = start, end - start, 0
            for k, line in enumerate(lines):
                consumed += len(line)
                l_end = start + int(dur * consumed / total) if k < len(lines) - 1 else end
                if l_end <= pos:
                    l_end = pos + 200
                entries.append([pos, l_end, line])
                pos = l_end

    # 第二遍：尾部停留压到不越过下一行（字幕出声时间贴齐 + 不遮静音期）
    blocks, idx = [], 0
    for i, (s, e, line) in enumerate(entries):
        next_s = entries[i + 1][0] if i + 1 < len(entries) else None
        show_end = e + _LINE_TAIL_MS
        if next_s is not None:
            show_end = min(show_end, max(next_s - 1, s + 200))
        idx += 1
        blocks.append(f"{idx}\n{format_srt_time(s)} --> {format_srt_time(show_end)}\n{line}\n")
    return "\n".join(blocks)


# 剪映专业版导入 SRT 的兜底机制：两条字幕之间空白间隙较长时，会把前一条字幕块
# 自动延展到下一条的起始时间（无视 SRT 里写好的结束时间，2026-09-19 用户实测）。
# 解法分两层（对应用户验证过的 Subtitle Edit 重存 + 间隙填充两条路）：
_SRT_GAP_FILL_MS = 2000     # 间隙超过 2 秒就插一条隐形字幕占位，剪映没有空白可填
_ZWSP = "\u200b"            # 零宽空格：占位字幕文本不可见，预览/成片都不显示


_SRT_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)


def _parse_srt_time(h, m, s, ms) -> int:
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def _parse_srt_entries(srt_text: str) -> list:
    """把 SRT 文本解析成 [(start_ms, end_ms, [text_lines])]，容错（BOM/点毫秒/缺序号）。"""
    entries = []   # (start_ms, end_ms, [text_lines])
    cur_time, cur_text = None, []
    for raw in srt_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip("\ufeff").rstrip()
        m = _SRT_TIME_RE.search(line)
        if m:
            g = m.groups()
            cur_time = (_parse_srt_time(*g[:4]), _parse_srt_time(*g[4:]))
            cur_text = []
            continue
        if not line:
            if cur_time is not None and cur_text:
                entries.append((cur_time[0], cur_time[1], cur_text))
            cur_time, cur_text = None, []
            continue
        if cur_time is not None and not line.isdigit():   # 跳过序号行，只收文本
            cur_text.append(line)
    if cur_time is not None and cur_text:
        entries.append((cur_time[0], cur_time[1], cur_text))
    entries.sort(key=lambda x: x[0])
    return entries


def jianying_safe_srt(srt_text: str, max_gap_ms: int = _SRT_GAP_FILL_MS) -> str:
    """SRT 后处理，防剪映导入后字幕块被自动拉长。返回严格 SubRip 格式文本。

    1. 严格规范化（等价 Subtitle Edit 重存）：序号重排、时间戳统一
       HH:MM:SS,mmm、行结尾统一 CRLF、块间恰好一个空行。
    2. 长间隙填充：相邻两条字幕间隙 > max_gap_ms 时，插入一条零宽空格的
       占位字幕盖住间隙——剪映的"空白自动延展"没有空白可填，不再拉长。
       占位条目文本不可见；在时间轴上会多出空块，介意可手动删。
    """
    entries = _parse_srt_entries(srt_text)
    if not entries:
        return srt_text

    # --- 间隙填充 ---
    filled = []
    for i, (s, e, lines) in enumerate(entries):
        filled.append((s, e, lines))
        if i + 1 < len(entries):
            ns = entries[i + 1][0]
            if ns - e > max_gap_ms:
                filled.append((e + 1, ns - 1, [_ZWSP]))

    # --- 严格重排输出（CRLF、HH:MM:SS,mmm、序号连续）---
    out = []
    for idx, (s, e, lines) in enumerate(filled, 1):
        out.append(f"{idx}\r\n{format_srt_time(s)} --> {format_srt_time(e)}\r\n" + "\r\n".join(lines) + "\r\n")
    return "\r\n".join(out)


def srt_to_ass(srt_text: str) -> str:
    """把 SRT 转成 ASS（Advanced SubStation Alpha）。

    剪映专业版对 SRT 有硬编码的"长间隙自动延展"逻辑（改设置/重存都绕不开），
    而导入 ASS 时逐条 Dialogue 严格保留时间范围，不会被拉长——这是官方
    字幕导入通道下最稳的绕行方案。输出 UTF-8 无 BOM。
    """
    entries = _parse_srt_entries(srt_text)
    if not entries:
        return ""

    def _ass_time(ms: int) -> str:
        cs = (ms + 5) // 10                 # ASS 精度到厘秒，四舍五入（截断会偏差到 9ms）
        h, rem = divmod(cs, 360000)
        m, rem = divmod(rem, 6000)
        s, cs2 = divmod(rem, 100)
        return f"{h}:{m:02d}:{s:02d}.{cs2:02d}"

    header = (
        "[Script Info]\r\n"
        "; Generated by VoiceMind 字幕工作台\r\n"
        "Title: subtitles\r\n"
        "ScriptType: v4.00+\r\n"
        "PlayResX: 1920\r\n"
        "PlayResY: 1080\r\n"
        "WrapStyle: 0\r\n"
        "ScaledBorderAndShadow: yes\r\n"
        "\r\n"
        "[V4+ Styles]\r\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\r\n"
        "Style: Default,微软雅黑,70,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,80,80,60,1\r\n"
        "\r\n"
        "[Events]\r\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\r\n"
    )
    events = []
    for s, e, lines in entries:
        text = "\\N".join(lines).replace("\u200b", "")   # 占位零宽空格在 ASS 里直接去掉
        if not text.strip():
            continue
        events.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Default,,0,0,0,,{text}\r\n")
    return header + "".join(events)


# 批量任务表：task_id -> {status, total, done, current, results, folder, error}
# 后台线程执行 + 前端轮询进度（整文件夹视频可能跑几十分钟，不能占着 HTTP 请求）。
BATCH_JOBS = {}


@app.post("/batch_video")
async def batch_video(
    folder: str = Form(""),
    file: str = Form(""),
    hotword: str = Form(""),
    offset: float = Form(0.0),
    max_len: int = Form(15),
):
    """批量处理视频文件夹 / 单个文件：每个生成同名 .srt（剪映/PR 直接导入）。

    立即返回 task_id，处理在后台线程进行；用 GET /batch_video_status 轮询进度。
    file 非空 = 单文件模式（视频或音频均可，SRT 生成在文件旁边）；
    否则按 folder 批量，folder 留空 = 项目 video/ 目录。max_len = 每行字幕最大字数。
    """
    if file.strip():
        fp = os.path.abspath(file.strip())
        if not os.path.isfile(fp):
            return JSONResponse({"code": -1, "error": f"文件不存在：{fp}"}, status_code=400)
        if not fp.lower().endswith(SEARCH_EXTS):
            return JSONResponse({"code": -1, "error": "只支持视频/音频文件"}, status_code=400)
        files = [fp]
        folder = os.path.dirname(fp)
    else:
        folder = folder.strip() or os.path.join(PROJECT_ROOT, "video")
        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            return JSONResponse({"code": -1, "error": f"文件夹不存在：{folder}"}, status_code=400)
        files = sorted({
            os.path.join(folder, n) for n in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, n)) and n.lower().endswith(VIDEO_EXTS)
        })
    if not files:
        return JSONResponse({"code": -1, "error": "文件夹里没有视频文件"}, status_code=400)

    task_id = uuid.uuid4().hex[:12]
    BATCH_JOBS[task_id] = {
        "status": "running", "total": len(files), "done": 0,
        "current": "", "results": [], "folder": folder, "error": "",
        "hotword": hotword,     # 回显到状态接口：排查"热词没生效"时先看这里是否为空
    }
    print(f"[batch] task={task_id} 收到热词={hotword!r} 文件数={len(files)}", flush=True)
    threading.Thread(
        target=_batch_video_worker, args=(task_id, files, hotword, offset, max(max_len, 5)),
        daemon=True,
    ).start()
    return {"code": 0, "task_id": task_id, "total": len(files), "folder": folder}


def _batch_video_worker(task_id: str, files: list, hotword: str, offset: float, max_len: int):
    job = BATCH_JOBS[task_id]
    try:
        for fp in files:
            name = os.path.basename(fp)
            job["current"] = name
            entry = {"file": name, "srt": None, "status": "failed", "error": "", "lines": 0, "fixes": []}
            wav = None
            try:
                wav = _to_wav_16k(fp)                       # 提取音轨 + 统一 16k 单声道
                text, segments, fixes = _generate_full(wav, hotword=hotword, return_fixes=True)
                if not segments and text:
                    segments = [{"text": text, "start": 0, "end": 0}]
                srt = generate_srt(segments, offset=offset, max_len=max_len)
                srt = jianying_safe_srt(srt)   # 防剪映自动拉长：规范化 + 长间隙填充
                ass = srt_to_ass(srt)          # ASS 严格保留时间，剪映仍拉长时的绕行方案
                base = os.path.splitext(fp)[0]
                srt_path = base + ".srt"
                # UTF-8 无 BOM：剪映导入 SRT 的兼容要求（Checklist 第 6 条）；newline=''
                # 禁止 Windows 文本模式再转义（否则 \r\n 会被写成 \r\r\n）
                with open(srt_path, "w", encoding="utf-8", newline="") as f:
                    f.write(srt)
                if ass:
                    with open(base + ".ass", "w", encoding="utf-8", newline="") as f:
                        f.write(ass)
                entry.update({
                    "srt": os.path.basename(srt_path),
                    "ass": os.path.basename(base + ".ass") if ass else None,
                    "status": "success",
                    "lines": sum(1 for b in srt.replace("\r\n", "\n").split("\n\n") if b.strip()),
                    "fixes": fixes,
                })
                if fixes:
                    # 整段文本和分段会各纠一次，同一处替换出现两条 → 去重
                    seen, uniq = set(), []
                    for f in fixes:
                        key = (f["from"], f["to"])
                        if key not in seen:
                            seen.add(key)
                            uniq.append(f)
                    fixes = uniq
                    entry["fixes"] = fixes
                    print(f"[batch] {name} 热词纠错: "
                          + ", ".join(f"{f['from']}→{f['to']}({f['similarity']})" for f in fixes), flush=True)
            except Exception as e:
                entry["error"] = str(e)
            finally:
                if wav:
                    _cleanup(wav)
            job["results"].append(entry)
            job["done"] += 1
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.get("/batch_video_status")
async def batch_video_status(task_id: str = ""):
    """查询批量字幕任务进度：{done, total, current, results[]}。"""
    job = BATCH_JOBS.get(task_id)
    if not job:
        return JSONResponse({"code": -1, "error": "任务不存在"}, status_code=404)
    return {"code": 0, **job}


@app.get("/pick_folder")
def pick_folder():
    """弹出系统原生「选择文件夹」对话框，返回 {folder: 路径}。

    本地桌面场景下后端就跑在用户机器上，tkinter 对话框直接弹在桌面前台，
    免去手输路径。注意三点：
    - 用同步 def（非 async）：对话框会阻塞到用户点确定/取消，
      FastAPI 会把它丢进线程池，不卡事件循环；
    - tkinter 对象必须在同一个线程里创建和销毁，所以再套一层线程；
    - topmost 保证对话框不被浏览器窗口压在底下。
    """
    result = {"folder": ""}
    holder = {}

    def _dialog():
        import tkinter as tk
        from tkinter import filedialog
        try:
            root = tk.Tk()
            root.withdraw()                      # 不显示主窗口，只要对话框
            root.attributes("-topmost", True)
            holder["folder"] = filedialog.askdirectory(title="选择视频文件夹") or ""
            root.destroy()
        except Exception as e:                   # 无显示器/会话不支持 GUI 等
            holder["error"] = str(e)

    t = threading.Thread(target=_dialog, daemon=True)
    t.start()
    t.join()                                     # 等用户选完（请求同步等待）
    result["folder"] = holder.get("folder", "")
    if "error" in holder:
        result["error"] = holder["error"]
    return {"code": 0, **result}


@app.get("/pick_file")
def pick_file():
    """弹出系统原生「打开文件」对话框，选单个视频/音频，返回 {file: 路径}。

    单文件字幕场景：不做整个文件夹，只处理手头这一个文件。
    线程模型与 /pick_folder 相同（同步 def + 内层线程持有 tkinter）。
    """
    result = {"file": ""}
    holder = {}

    def _dialog():
        import tkinter as tk
        from tkinter import filedialog
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            exts = " ".join(SEARCH_EXTS)          # 视频 + 音频都支持
            holder["file"] = filedialog.askopenfilename(
                title="选择视频或音频文件",
                filetypes=[("视频/音频", exts), ("所有文件", "*.*")],
            ) or ""
            root.destroy()
        except Exception as e:
            holder["error"] = str(e)

    t = threading.Thread(target=_dialog, daemon=True)
    t.start()
    t.join()
    result["file"] = holder.get("file", "")
    if "error" in holder:
        result["error"] = holder["error"]
    return {"code": 0, **result}


def _to_wav_16k(src: str) -> str:
    """统一转 16k 单声道 wav（-vn 丢弃视频流），输出到系统临时目录。

    即使本来就是 wav 也过一遍 ffmpeg：视频/各类音频容器统一处理，
    且强制 16k 单声道后 VAD 时间轴更可靠。
    """
    dst = os.path.join(TEMP_DIR, f"conv_{uuid.uuid4().hex}.wav")
    ffmpeg = shutil.which("ffmpeg") or r"C:\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-vn", "-ar", "16000", "-ac", "1", dst],
        check=True, timeout=300,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return dst


def _char_times(text: str, ctc_text: str, ctc_times: list):
    """把最终文本的每个字对齐到 CTC 字级时间戳，返回 [(start_ms, end_ms), ...]（与 text 等长）。

    v2（LLM 架构）的输出 text 与 ctc_text 不完全一致：LLM 会加标点、改写个别字
    （如 ctc 的 "m二零" 被 LLM 修成 "F幺零"）。用 difflib 对齐后：
    - equal 块逐字映射；
    - replace 块按位置比例借用 CTC 时间；
    - insert（LLM 加的字，多是标点）沿用邻近字的时间。
    """
    import difflib
    n = len(text)
    times = [(None, None)] * n
    sm = difflib.SequenceMatcher(None, ctc_text, text, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                if i1 + k < len(ctc_times):
                    times[j1 + k] = ctc_times[i1 + k]
        elif tag == "replace" and i2 > i1:
            span = i2 - i1
            for k in range(j2 - j1):
                idx = i1 + min(span - 1, int((k + 0.5) * span / max(1, j2 - j1)))
                if idx < len(ctc_times):
                    times[j1 + k] = ctc_times[idx]
    # 前向填补（LLM 插入的字）：沿用前一个字的 end
    last = None
    for i in range(n):
        if times[i][0] is None:
            if last is not None:
                times[i] = (last, last)
        else:
            last = times[i][1]
    # 后向兜底（开头就被插入的罕见情况）：沿用第一个已知 start
    first = next((t[0] for t in times if t[0] is not None), 0)
    for i in range(n):
        if times[i][0] is None:
            times[i] = (first, first)
    return times


def _segments_from_char_ts(item: dict) -> list:
    """v2 模型没有 sentence_info：用字级时间戳与最终文本对齐，切出句级分段。

    item["timestamps"] 是 CTC 对齐的 token 时间 [{token, start_time, end_time}]（秒），
    token 可能是多字符（如英文词），内部按字符均分时间。

    切分两刀：
    1. 句末标点（。！？；）——常规断句；
    2. 句内长停顿（相邻字间隔 ≥ _PAUSE_SPLIT_MS）——长句经常横跨好几段
       静音（"六、学生会的…吧"一句跨 15 秒），按字级时间硬切，保证每段
       字幕只覆盖真正在说话的区间。
    每段附带 char_times（与 text 逐字对齐的出声时间），供 generate_srt
    精确到行；热词校正是等长替换，不会破坏对齐。
    """
    toks = item.get("timestamps") or []
    if not toks:
        return []
    chars, times = [], []
    for tk in toks:
        t0 = float(tk.get("start_time", 0) or 0) * 1000.0
        t1 = float(tk.get("end_time", 0) or 0) * 1000.0
        s = str(tk.get("token", ""))
        m = max(1, len(s))
        for i, ch in enumerate(s):
            chars.append(ch)
            times.append((int(t0 + (t1 - t0) * i / m), int(t0 + (t1 - t0) * (i + 1) / m)))
    ctc_text = "".join(chars)
    text = _clean_tags(item.get("text"))
    if not text or not ctc_text:
        return []
    ct = _char_times(text, ctc_text, times)

    # 第一刀：句末标点
    sent_spans, start_idx = [], 0
    for i, ch in enumerate(text):
        if ch in "。！？!?；;":
            if text[start_idx:i + 1].strip():
                sent_spans.append((start_idx, i + 1))
            start_idx = i + 1
    if start_idx < len(text) and text[start_idx:].strip():
        sent_spans.append((start_idx, len(text)))

    # 第二刀：句内长停顿（字间空隙 ≥ 1 秒）
    segments = []
    for a, b in sent_spans:
        cuts = [a]
        for i in range(a + 1, b):
            if ct[i][0] - ct[i - 1][1] >= _PAUSE_SPLIT_MS:
                cuts.append(i)
        cuts.append(b)
        for s, e in zip(cuts, cuts[1:]):
            raw = text[s:e]
            lead = len(raw) - len(raw.lstrip())     # 去掉首尾空格并同步收紧时间窗
            trail = len(raw) - len(raw.rstrip())
            s2, e2 = s + lead, e - trail
            if s2 >= e2:
                continue
            piece = {
                "text": text[s2:e2],
                "start": ct[s2][0],
                "end": ct[e2 - 1][1],
                "char_times": ct[s2:e2],
            }
            # 行首是标点（停顿切分把逗号切到了下一行开头）→ 并回上一行：
            # 只拼文本和字时间，不延长上一行的时间窗（标点无声，尾巴仍被钳制）
            if segments and re.match(r"^[，、,：:；;）)」》”]", piece["text"]):
                prev = segments[-1]
                prev["text"] += piece["text"]
                prev["end"] = piece["end"]
                prev["char_times"] = prev["char_times"] + piece["char_times"]
            else:
                segments.append(piece)
    return segments


def _generate_full(path: str, hotword: str = "", return_fixes: bool = False):
    """识别一次，同时返回（整段文本, 句子级分段）。

    时间戳来源（按优先级）：
    1. sentence_info（SenseVoice 系，VAD 粒度，毫秒）；
    2. 字级 timestamps + 文本对齐（v2/LLM 系，精度到字，按标点切句）。
    hotword：热词写法校正（见 _apply_hotwords），作用于整段文本与每个分段。
    return_fixes：为 True 时额外返回纠错明细 [{from,to,similarity}]（批量任务
    回显到前端/日志用，方便判断「热词没生效」到底卡在前端传参还是后端匹配）。
    """
    res = get_base_model().generate(input=path, sentence_timestamp=True, batch_size_s=300)
    text = _clean_tags(res[0].get("text")) if res else ""
    segments = []
    for item in res:
        for sent in item.get("sentence_info") or []:
            seg_text = _clean_tags(sent.get("text"))
            if seg_text:
                segments.append({
                    "text": seg_text,
                    "start": int(sent.get("start", 0) or 0),   # 毫秒
                    "end": int(sent.get("end", 0) or 0),       # 毫秒
                })
        if not segments:
            # v2：sentence_info 为空，走字级时间戳对齐
            segments = _segments_from_char_ts(item)
    fixes = []
    if hotword.strip():
        text, fixes = _apply_hotwords(text, hotword)
        for seg in segments:
            seg["text"], seg_fixes = _apply_hotwords(seg["text"], hotword)
            fixes.extend(seg_fixes)
    if return_fixes:
        return text, segments, fixes
    return text, segments


def _recognize_segments(path: str) -> list:
    """识别并返回句子级分段 [{text, start, end}]，时间单位毫秒。

    关键点：FunASR 的句子级时间戳需要显式传 sentence_timestamp=True
    （结果放在 sentence_info 里），不存在 output_timestamp 这个参数；
    SenseVoice 不预测逐字时间戳，分段粒度取决于 VAD。
    """
    text, segments = _generate_full(path)
    # 兜底：拿不到分段就整段一条（时间未知，置 0）
    if not segments and text:
        segments = [{"text": text, "start": 0, "end": 0}]
    return segments


def _parse_keywords(keywords: str):
    """逗号分隔的关键词串 -> 去重后的列表；空则返回 None（由调用方报错）。"""
    seen, result = set(), []
    for kw in (keywords or "").replace("，", ",").split(","):
        kw = kw.strip()
        if kw and kw not in seen:
            seen.add(kw)
            result.append(kw)
    return result or None


def _match_segment(text: str, keyword_list: list, threshold: int):
    """对一个分段做精确+模糊匹配，返回命中的关键词信息列表（未命中为空列表）。"""
    low = text.lower()
    hits = []
    for kw in keyword_list:
        if kw.lower() in low:
            hits.append({"keyword": kw, "match_type": "exact", "similarity": 100.0})
        else:
            sim = fuzz.partial_ratio(kw.lower(), low)
            if sim >= threshold:
                hits.append({"keyword": kw, "match_type": "fuzzy", "similarity": round(sim, 1)})
    return hits


@app.post("/search_keywords")
async def search_keywords(
    file: UploadFile = File(...),
    keywords: str = Form(""),
    fuzzy_threshold: int = Form(75),
):
    """
    识别音频/视频并检索关键词，返回每个匹配片段的时间点（毫秒）与内容。

    - keywords：逗号分隔，如 "用户体验,UX,上线时间"
    - fuzzy_threshold：模糊匹配阈值 50~100，越高越严格（默认 75）。
      注意：partial_ratio 下短关键词错 1 个字扣分很重——4 字词错 1 字
      只剩 75 分（如"用户体验"vs"用户体检"），阈值 80 会漏掉这类错别字。
    """
    keyword_list = _parse_keywords(keywords)
    if not keyword_list:
        return JSONResponse({"code": -1, "error": "请至少指定一个关键词"}, status_code=400)
    threshold = max(50, min(100, int(fuzzy_threshold)))

    temp_filename = _save_upload(file)
    conv_file = None
    try:
        src = _to_wav_16k(temp_filename)
        conv_file = src
        segments = _recognize_segments(src)

        matches = []
        for seg in segments:
            hits = _match_segment(seg["text"], keyword_list, threshold)
            if hits:
                matches.append({
                    "keywords": [h["keyword"] for h in hits],
                    "hits": hits,
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                })

        return {
            "code": 0,
            "filename": file.filename,
            "keywords": keyword_list,
            "fuzzy_threshold": threshold,
            "total_segments": len(segments),
            "total_matches": len(matches),
            "matches": matches,
        }
    except Exception as e:
        return JSONResponse({"code": -1, "error": str(e)}, status_code=500)
    finally:
        _cleanup(temp_filename)
        if conv_file:
            _cleanup(conv_file)


@app.post("/batch_search")
async def batch_search(
    folder: str = Form(""),
    keywords: str = Form(""),
    fuzzy_threshold: int = Form(75),
):
    """
    批量检索：扫描文件夹（留空 = 项目 audio/ 与 video/ 目录）下所有音视频，
    逐个识别并搜索关键词，返回按文件分组的结果。耗时与文件总时长成正比。
    """
    keyword_list = _parse_keywords(keywords)
    if not keyword_list:
        return JSONResponse({"code": -1, "error": "请至少指定一个关键词"}, status_code=400)
    threshold = max(50, min(100, int(fuzzy_threshold)))

    # 留空时默认扫项目素材目录；显式传路径时必须是存在的目录
    if folder.strip():
        roots = [os.path.abspath(folder.strip())]
        if not os.path.isdir(roots[0]):
            return JSONResponse({"code": -1, "error": f"文件夹不存在：{roots[0]}"}, status_code=400)
    else:
        roots = [os.path.join(PROJECT_ROOT, "audio"), os.path.join(PROJECT_ROOT, "video")]
        roots = [r for r in roots if os.path.isdir(r)]
        if not roots:
            return JSONResponse({"code": -1, "error": "未指定文件夹，且项目 audio/video 目录不存在"}, status_code=400)

    files = []
    for root in roots:
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isfile(path) and name.lower().endswith(SEARCH_EXTS):
                files.append(path)
    files = sorted(set(files))

    results, failed = [], []
    for filepath in files:
        wav_path = None
        try:
            wav_path = _to_wav_16k(filepath)
            segments = _recognize_segments(wav_path)
            matches = []
            for seg in segments:
                hits = _match_segment(seg["text"], keyword_list, threshold)
                if hits:
                    matches.append({
                        "keywords": [h["keyword"] for h in hits],
                        "hits": hits,
                        "text": seg["text"],
                        "start": seg["start"],
                        "end": seg["end"],
                    })
            if matches:
                results.append({"file": os.path.basename(filepath), "path": filepath, "matches": matches})
        except Exception as e:
            failed.append({"file": os.path.basename(filepath), "error": str(e)})
        finally:
            if wav_path:
                _cleanup(wav_path)

    return {
        "code": 0,
        "keywords": keyword_list,
        "fuzzy_threshold": threshold,
        "scanned_dirs": roots,
        "total_files": len(files),
        "matched_files": len(results),
        "failed_files": failed,
        "results": results,
    }


# ================================================================
# 语音智能体：RAG 知识库 + Agent 问答
# ================================================================
# 这两个模块本身只依赖标准库，重依赖（chromadb / sentence-transformers / openai）
# 都在模块内部用到时才加载，所以放在这里 import 不会拖慢服务启动。
from rag_module import get_rag_engine, is_ready as rag_is_ready          # noqa: E402
from agent_module import (                                                # noqa: E402
    get_agent,
    reload_agent,
    config_status as llm_status,
    CONFIG_PATH as LLM_CONFIG_PATH,     # llm_config.json 的位置，保存 Key 接口要用
)


# ---- 会话记忆：让"它怎么用？"这类追问能接上上下文 ----
# 结构 {session_id: [{"role": "user"/"assistant", "content": "..."}]}
# 只放内存，服务重启即清空（问答场景够用，也避免用户隐私文本落盘）
QA_SESSIONS = {}
_QA_MAX_SESSIONS = 200      # 最多保留 200 个会话，防止长跑内存涨
_QA_MAX_TURNS = 8           # 每个会话最多留 8 轮，超了丢最旧的
_QA_LOCK = threading.Lock()


def _qa_history(session_id: str) -> list:
    """取某个会话的历史消息（副本）。"""
    if not session_id:
        return []
    with _QA_LOCK:
        return list(QA_SESSIONS.get(session_id, []))


def _qa_push(session_id: str, role: str, content: str):
    """往会话里追加一条消息（用户问或助手答）。"""
    if not session_id or not content:
        return
    with _QA_LOCK:
        # 会话数太多时，丢掉最早创建的那个（dict 保序）
        if session_id not in QA_SESSIONS and len(QA_SESSIONS) >= _QA_MAX_SESSIONS:
            QA_SESSIONS.pop(next(iter(QA_SESSIONS)), None)
        hist = QA_SESSIONS.setdefault(session_id, [])
        hist.append({"role": role, "content": content})
        if len(hist) > _QA_MAX_TURNS * 2:
            del hist[:len(hist) - _QA_MAX_TURNS * 2]


def _qa_pipeline(question: str, session_id: str = "", top_k: int = 0, hotword: str = "") -> dict:
    """问答主链路：检索 → 生成。语音和文字两个入口共用这一段，避免逻辑分叉。

    返回 {answer, used_llm, sources, hits, error, rag_ms, llm_ms}
    """
    t_rag = time.time()
    rag = get_rag_engine()                       # 首次调用会加载向量模型（几秒）
    cfg_topk = 0
    try:
        cfg_topk = int(get_agent().config.get("top_k", 3) or 3)
    except Exception:
        cfg_topk = 3
    hits = rag.search(question, top_k=top_k or cfg_topk)
    rag_ms = int((time.time() - t_rag) * 1000)

    t_llm = time.time()
    result = get_agent().answer(question, hits, history=_qa_history(session_id))
    llm_ms = int((time.time() - t_llm) * 1000)

    # 记住这一轮（下次追问时带上）
    _qa_push(session_id, "user", question)
    _qa_push(session_id, "assistant", result.get("answer", ""))

    return {
        "answer": result.get("answer", ""),
        "used_llm": result.get("used_llm", False),
        "sources": result.get("sources", []),
        # 命中片段只回显前 160 字，页面展示用；全文留在库里
        "hits": [
            {"source": h.get("source"), "score": h.get("score"),
             "text": (h.get("text") or "")[:160]}
            for h in hits
        ],
        "error": result.get("error", ""),
        "rag_ms": rag_ms,
        "llm_ms": llm_ms,
    }


@app.post("/upload_knowledge")
async def upload_knowledge(file: UploadFile = File(...), source_name: str = Form("")):
    """上传知识文档（docx / pdf / md / txt 等），自动切块 + 向量化 + 入库。

    解析规则见 rag_module.read_file：Word 用 python-docx（含表格），
    PDF 用 pypdf 逐页抽文本，其余按纯文本读（utf-8 失败退 gbk）。
    临时文件保留**原始扩展名**并按原始文件名记来源，
    否则临时文件名是一串 uuid，知识库里所有来源都会显示成乱码串，没法用。
    """
    name = (source_name or file.filename or "").strip()
    ext = os.path.splitext(name)[1].lower()
    allowed = (".docx", ".doc", ".pdf", ".md", ".markdown",
               ".txt", ".csv", ".json", ".log", ".srt")
    if ext and ext not in allowed:
        return JSONResponse(
            {"code": -1, "error": f"不支持的文件格式 {ext}，目前支持 Word / PDF / Markdown / TXT"},
            status_code=400,
        )

    temp_path = _save_upload(file)      # 保留原始扩展名存到系统临时目录
    try:
        info = get_rag_engine().add_file(temp_path, source_name=name)
        if info.get("error"):
            return JSONResponse({"code": -1, "error": info["error"]}, status_code=400)
        print(f"[kb] 入库 {name}：{info['chunks']} 块 / {info['chars']} 字", flush=True)
        return {"code": 0, "filename": name, **info}
    except Exception as e:
        return JSONResponse({"code": -1, "error": f"入库失败：{e}"}, status_code=500)
    finally:
        _cleanup(temp_path)


@app.post("/knowledge_text")
async def knowledge_text(text: str = Form(...), source_name: str = Form("手动输入")):
    """直接粘贴一段文字入库（不想存成文件时用）。"""
    if not text.strip():
        return JSONResponse({"code": -1, "error": "内容为空"}, status_code=400)
    try:
        info = get_rag_engine().add_text(text, source_name=source_name.strip() or "手动输入")
        return {"code": 0, "filename": source_name, **info}
    except Exception as e:
        return JSONResponse({"code": -1, "error": f"入库失败：{e}"}, status_code=500)


@app.get("/knowledge_list")
async def knowledge_list():
    """查看知识库现状：有哪些文档、各多少块、共多少块，以及引擎是否已加载。"""
    if not rag_is_ready():
        # 还没用过知识库：不主动加载模型（加载要几秒），直接如实返回空
        return {"code": 0, "ready": False, "total_chunks": 0, "sources": []}
    rag = get_rag_engine()
    return {
        "code": 0,
        "ready": True,
        "total_chunks": rag.count(),
        "sources": rag.list_sources(),
    }


@app.post("/knowledge_delete")
async def knowledge_delete(source: str = Form(...)):
    """按文档名删除知识（重新上传同名文件会自动覆盖，一般用不到这个）。"""
    if not source.strip():
        return JSONResponse({"code": -1, "error": "缺少 source"}, status_code=400)
    get_rag_engine().delete_source(source.strip())
    return {"code": 0, "message": f"已删除 {source}"}


@app.post("/knowledge_reset")
async def knowledge_reset():
    """清空整个知识库（危险操作，前端要二次确认）。"""
    get_rag_engine().reset()
    print("[kb] 知识库已清空", flush=True)
    return {"code": 0, "message": "知识库已清空"}


@app.post("/qa_ask")
async def qa_ask(
    question: str = Form(...),
    session_id: str = Form(""),
    top_k: int = Form(0),
):
    """文字提问（前端输入框、或语音识别完的文本都走这里）。"""
    question = question.strip()
    if not question:
        return JSONResponse({"code": -1, "error": "问题为空"}, status_code=400)
    try:
        res = _qa_pipeline(question, session_id=session_id, top_k=top_k)
        return {"code": 0, "question": question, **res}
    except Exception as e:
        return JSONResponse({"code": -1, "error": f"问答失败：{e}"}, status_code=500)


@app.post("/voice_qa")
async def voice_qa(
    file: UploadFile = File(...),
    hotword: str = Form(""),
    session_id: str = Form(""),
    top_k: int = Form(0),
):
    """语音问答：上传一段音频/视频 → 识别成问题 → 检索知识库 → 大模型作答。

    完整链路：ffmpeg 转 16k wav → FunASR 识别（复用 _generate_full，自动享受
    热词校正与字级时间戳）→ RAG 检索 → Agent 生成 → 返回文字答案。

    与说明书实现的差异：
    1. 不直接用 model.generate(input=..., language="zh")，而是复用项目已有的
       _generate_full()。因为默认模型是 Fun-ASR-Nano v2（LLM 架构），参数集
       和 SenseVoice 不同，直接调用可能不认 language 参数；复用现成函数还能
       顺带拿到热词校正，答人名不会答错。
    2. 上传文件先经 ffmpeg 转 16k 单声道 wav 再识别（说明书里直接把上传文件
       写成 .wav，若用户传的是 mp3/m4a，模型解码会失败）。
    """
    temp_path = _save_upload(file)
    conv_path = None
    t0 = time.time()
    try:
        conv_path = _to_wav_16k(temp_path)          # 统一转码，兼容任意音视频格式
        question, _segments, fixes = _generate_full(conv_path, hotword=hotword, return_fixes=True)
        asr_ms = int((time.time() - t0) * 1000)

        question = (question or "").strip()
        if not question:
            return {
                "code": 0, "question": "", "answer": "没有识别到语音内容，请靠近麦克风或确认音频有声音。",
                "used_llm": False, "sources": [], "hits": [], "error": "",
                "asr_ms": asr_ms, "rag_ms": 0, "llm_ms": 0, "fixes": [],
            }

        res = _qa_pipeline(question, session_id=session_id, top_k=top_k, hotword=hotword)
        return {"code": 0, "question": question, "fixes": fixes, "asr_ms": asr_ms, **res}
    except subprocess.CalledProcessError as e:
        return JSONResponse(
            {"code": -1, "error": "音频转码失败：文件可能损坏，或 ffmpeg 不在 PATH 中"},
            status_code=500,
        )
    except Exception as e:
        return JSONResponse({"code": -1, "error": f"语音问答失败：{e}"}, status_code=500)
    finally:
        _cleanup(temp_path)
        if conv_path:
            _cleanup(conv_path)


@app.get("/qa_config")
async def qa_config_get():
    """查看大模型配置状态（Key 只回显后 4 位，不回传明文）。"""
    return {"code": 0, **llm_status()}


@app.post("/qa_config")
async def qa_config_set(
    api_key: str = Form(""),
    model: str = Form(""),
    base_url: str = Form(""),
):
    """在前端直接保存 API Key / 模型名（写入 code/llm_config.json 并热重载）。

    这样不用为了填一个 Key 去手动改文件、再重启服务。
    """
    cfg = {}
    if os.path.exists(LLM_CONFIG_PATH):
        try:
            with open(LLM_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
        except Exception:
            cfg = {}
    if api_key.strip():
        cfg["api_key"] = api_key.strip()
    if model.strip():
        cfg["model"] = model.strip()
    if base_url.strip():
        cfg["base_url"] = base_url.strip()
    cfg.setdefault("provider", "deepseek")
    cfg.setdefault("temperature", 0.2)
    cfg.setdefault("max_tokens", 800)
    cfg.setdefault("timeout", 60)
    cfg.setdefault("history_turns", 3)
    cfg.setdefault("top_k", 3)

    with open(LLM_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    reload_agent()          # 重新读配置，不用重启服务
    print("[agent] 配置已更新并热重载", flush=True)
    return {"code": 0, **llm_status()}


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("VoiceMind 服务启动中（模型按需加载，端口先就绪）...")
    print("首次识别/检索时会加载模型，约 30~90 秒")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
