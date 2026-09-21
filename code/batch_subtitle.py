#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_subtitle.py —— 批量给视频/音频生成 SRT 字幕

用法示例：
    python batch_subtitle.py                          # 默认扫描 ../audio 与 ../video，字幕输出到 ../subtitle
    python batch_subtitle.py D:/videos                # 处理指定目录
    python batch_subtitle.py D:/a D:/b -r             # 同时处理多个目录（可递归）
    python batch_subtitle.py --side-by-side           # 字幕与源文件同目录（默认集中到 ../subtitle）
    python batch_subtitle.py --outdir D:/subs         # 指定字幕输出目录
    python batch_subtitle.py --force                  # 覆盖已存在的 .srt
    python batch_subtitle.py --language zh            # 指定中文（默认 auto 自动判定）
    python batch_subtitle.py --list                   # 只列出待处理文件，不识别

设计说明（为什么和"常见写法"不一样）：
1. 不使用 punc_model 标点模型。SenseVoiceSmall 自身的富文本输出已经带标点，
   再叠一层 ct-punc 会得到重复/错乱标点（实测出现"，。"连排），反而更差。
2. FunASR 的句子级时间戳需要显式传 sentence_timestamp=True 才会生成
   （见 funasr/auto/auto_model.py 中 `elif kwargs.get("sentence_timestamp", False)`），
   否则结果里只有 key/text 两个字段，拿不到任何时间信息，
   最终会生成一条 00:00:00,000 --> 00:00:00,000 的废字幕。
3. SenseVoice 不预测逐字时间戳，所以 sentence_info 的粒度取决于 VAD——
   一段连续朗读只会得到 1 个覆盖全片的片段。因此这里在 VAD 片段内部
   再按标点切句，并按"非标点字数"比例分配时间，才能得到可用的字幕。
4. 时间戳单位是【毫秒】。按"秒"解释会让整条时间轴缩小 1000 倍。
5. --language 默认 auto：实测强制 zh 并不能改变 SenseVoice 的语种判定
   （输出文本里依然带 <|en|> 标记），却会让非中文内容按中文规则切句。
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave

# ---------------------------------------------------------------- 常量配置

# 项目目录结构：<项目根>/code/batch_subtitle.py、<项目根>/audio、<项目根>/video、<项目根>/subtitle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # .../code
PROJECT_ROOT = os.path.dirname(BASE_DIR)                  # 项目根目录
DEFAULT_INPUT_DIRS = [
    os.path.join(PROJECT_ROOT, "audio"),
    os.path.join(PROJECT_ROOT, "video"),
]
DEFAULT_OUTDIR = os.path.join(PROJECT_ROOT, "subtitle")

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm", ".m4v", ".ts")
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma", ".opus")
ALL_EXTS = VIDEO_EXTS + AUDIO_EXTS

FFMPEG_FALLBACKS = (
    r"C:\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
)

# 句末标点：优先在这里切分
SENT_END_CHARS = "。！？!?…"
# 句中停顿标点：句子太长时二次切分。
# 刻意不含顿号"、"——顿号是列举内部的停顿（如"桥梁纽带、宣传公寓规定"），
# 在顿号处断行会切出大量 5~7 字的碎片字幕。
CLAUSE_CHARS = "，；：,;:"
# 全部标点（计算"有效字数"权重时排除）
ALL_PUNC = SENT_END_CHARS + CLAUSE_CHARS + "、“”‘’\"'（）()《》〈〉【】—－-…~·"
# 可以作为切分点的字符：标点 + 空格（英文必须靠空格断词，不能按字符硬切）。
# 刻意不含 ASCII 的 ' 和 " —— 英文里它们出现在词内部（I'm / I've / don't），
# 在此断开会把缩写词切成两半，中文引号（“”‘’）则不受影响。
CUT_CHARS = SENT_END_CHARS + CLAUSE_CHARS + "、“”‘’（）()《》〈〉【】—－-…~· "


def is_cjk_dominant(text: str) -> bool:
    """粗略判断文本是否以中文为主，用于决定单行字幕长度。

    中文一个字信息量大，20 字左右一行合适；英文按字符算同样长度只有 3~4 个词，
    会切得过于零碎，因此英文放宽到约 42 字符（行业上英文字幕常见 42 字符/行）。
    """
    if not text:
        return False
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk >= max(1, len(text) * 0.3)


# ---------------------------------------------------------------- 工具函数

def find_ffmpeg() -> str:
    """定位 ffmpeg：先查 PATH，再查常见安装位置。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for cand in FFMPEG_FALLBACKS:
        if os.path.isfile(cand):
            return cand
    raise RuntimeError(
        "未找到 ffmpeg。请确认已安装并加入 PATH，或修改脚本里的 FFMPEG_FALLBACKS。"
    )


def format_srt_time(ms: float) -> str:
    """毫秒 -> SRT 时间格式 00:00:00,000

    注意：FunASR 返回的 start/end 单位是【毫秒】。
    若按"秒"来解释，时间轴会整体缩小 1000 倍（全部变成 00:00:00,000）。
    """
    total_ms = max(0, int(round(ms)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def normalize_to_wav(src: str, ffmpeg: str, temp_dir: str) -> str:
    """用 ffmpeg 统一转成 16kHz 单声道 WAV。

    为什么所有文件都过一遍 ffmpeg（哪怕本来已经是 wav）：
    - 视频/各类音频容器都能统一处理，避免"个别文件莫名解码失败"；
    - 强制 16k 单声道，正是模型期望的输入，时间轴也更可靠。
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="batchsub_", dir=temp_dir)
    os.close(fd)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", src,
        "-vn",                # 丢弃视频流
        "-ar", "16000", "-ac", "1",
        "-f", "wav",
        wav_path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        try:
            os.remove(wav_path)
        except OSError:
            pass
        err = proc.stderr.decode("utf-8", errors="ignore").strip().splitlines()
        raise RuntimeError("ffmpeg 提取音频失败: " + (err[-1] if err else "未知错误"))
    return wav_path


def wav_duration_ms(wav_path: str) -> float:
    """读取标准 WAV 的时长（毫秒）。"""
    with wave.open(wav_path, "rb") as w:
        return w.getnframes() / float(w.getframerate()) * 1000.0


def clean_text(text: str) -> str:
    """去掉 SenseVoice 输出里的富文本标记，如 <|zh|><|NEUTRAL|><|Speech|><|woitn|>"""
    return re.sub(r"<\|[^|]*\|>", "", str(text or "")).strip()


def _best_cut(chunk: str, max_len: int) -> int:
    """在 max_len 附近找一个标点/空格作为切点，避免把词语拦腰截断。

    中文例：「并传达停水停电通知、桥梁纽带、宣传公寓规定、组织文明评比反馈学生建议。」
    若直接按第 20 个字硬切，会断成「宣传公寓规 / 定、组织…」；
    在附近的顿号处断开则得到「…宣传公寓规定、/ 组织文明评比…」，可读性好得多。

    英文例：无标点的 "Hello everyone welcome back to our podcast …"，
    切点必须落在空格上，否则会切出 "welco / melcom" 这种断词。
    """
    lo = max(1, int(max_len * 0.6))
    hi = min(len(chunk) - 1, int(max_len * 1.15))
    best = None
    for i in range(lo, hi + 1):
        if chunk[i - 1] in CUT_CHARS:          # 第 i-1 位是标点/空格 -> 可在它之后切
            if best is None or abs(i - max_len) < abs(best - max_len):
                best = i
    return best if best is not None else max_len


def split_text(text: str, max_len: int = 20) -> list:
    """把一段文本切成适合做字幕的短行。

    三级策略：
      1. 先按句末标点（。！？）切；
      2. 仍然过长的片段，再按逗号类停顿标点（，；：）切；
      3. 还没有标点可切的超长片段（典型场景：英文转写无标点），
         在附近的标点或空格处断开，绝不把单词拦腰截断。
    标点跟随前一行，不单独成行。
    非中文内容自动放宽行长（英文约 42 字符），否则一行只有 3~4 个单词。
    """
    text = clean_text(text)
    if not text:
        return []

    # 中文 20 字 / 英文约 42 字符，贴合字幕行业惯例
    limit = max_len if is_cjk_dominant(text) else int(max_len * 2.1)

    # 第 1 级：按句末标点切，标点保留在句尾。
    # 英文句号要求其后必须跟空白，否则会把小数（3.14）、缩写（Mr.）切开。
    chunks = []
    for raw in re.findall(r"[^" + re.escape(SENT_END_CHARS) + r"]+[" + re.escape(SENT_END_CHARS) + r"]*", text):
        chunks.extend(p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip())

    # 第 2 级：过长的按停顿标点再切
    stage2 = []
    for c in chunks:
        if len(c) <= limit:
            stage2.append(c)
            continue
        parts = re.findall(r"[^" + re.escape(CLAUSE_CHARS) + r"]+[" + re.escape(CLAUSE_CHARS) + r"]*", c)
        stage2.extend([p.strip() for p in parts if p.strip()])

    # 第 3 级：仍然过长的，切在附近标点/空格处（保留可读性，不硬切字符）
    lines = []
    for c in stage2:
        while len(c) > limit * 1.4:
            cut = _best_cut(c, limit)
            piece = c[:cut].strip()
            if piece:
                lines.append(piece)
            c = c[cut:].lstrip()
        if c.strip():
            lines.append(c.strip())

    # 合并"过短且没有句末标点"的碎片行，避免出现 1~2 个字的字幕
    merged = []
    for line in lines:
        if merged and len(line) <= 3 and line[-1] not in SENT_END_CHARS:
            prev = merged[-1]
            # 英文合并要补空格，否则会粘成一个词（"have" + "a" -> "havea"）
            need_space = (prev and prev[-1].isascii() and prev[-1].isalnum()
                          and line[0].isascii() and line[0].isalnum())
            merged[-1] = prev + (" " if need_space else "") + line
        else:
            merged.append(line)
    return merged


def time_weights(lines: list) -> list:
    """按"非标点有效字数"给每行分配权重，用于按比例切分时间。"""
    weights = []
    for line in lines:
        effective = len(re.sub(r"[" + re.escape(ALL_PUNC) + r"\s]", "", line))
        weights.append(max(1, effective))
    return weights


def build_srt_entries(res: list, wav_path: str) -> list:
    """把 FunASR 结果整理成 SRT 条目列表 [(start_ms, end_ms, text), ...]。"""
    if not res:
        return []

    r0 = res[0]
    fallback_duration = wav_duration_ms(wav_path)
    sentences = r0.get("sentence_info") or []

    # 拿不到分段：整段文本 + 音频总时长兜底
    if not sentences:
        text = clean_text(r0.get("text", ""))
        if not text:
            return []
        sentences = [{"start": 0, "end": fallback_duration, "text": text, "sentence": text}]

    entries = []
    for seg in sentences:
        seg_text = clean_text(seg.get("text") or seg.get("sentence") or "")
        if not seg_text:
            continue

        start = float(seg.get("start", 0) or 0)
        end = float(seg.get("end", 0) or 0)
        if end <= start:
            end = start + max(1000.0, len(seg_text) * 200.0)

        lines = split_text(seg_text)
        if not lines:
            continue
        if len(lines) == 1:
            entries.append((start, end, lines[0]))
            continue

        # 一行字幕对应一段时间：按有效字数比例分配该片段的时长
        weights = time_weights(lines)
        total_w = sum(weights)
        span = end - start
        cursor = start
        for line, w in zip(lines, weights):
            piece = span * w / total_w
            entries.append((cursor, cursor + piece, line))
            cursor += piece

    # 保证时间单调不减，且每行至少有 300ms 显示时间
    cleaned = []
    for i, (s, e, t) in enumerate(entries):
        if cleaned:
            s = max(s, cleaned[-1][1])
        if e <= s:
            e = s + 300.0
        cleaned.append((s, e, t))
    return cleaned


def write_srt(entries: list, srt_path: str) -> None:
    """写出 SRT。用 utf-8-sig（带 BOM），Windows 记事本打开不乱码。"""
    out = []
    for i, (start, end, text) in enumerate(entries, 1):
        out.append(str(i))
        out.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        out.append(text)
        out.append("")
    with open(srt_path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write("\n".join(out))


# ---------------------------------------------------------------- 主流程

def collect_files(root: str, recursive: bool) -> list:
    """收集音视频文件。扩展名大小写不敏感；跳过自己产生的临时文件。"""
    found = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                found.append(os.path.join(dirpath, name))
    else:
        with os.scandir(root) as it:
            for entry in it:
                if entry.is_file():
                    found.append(entry.path)

    result = []
    for path in found:
        name = os.path.basename(path)
        # 跳过本脚本自己生成的临时音频，否则重复运行会把它们当成素材
        if name.startswith("batchsub_") or name.endswith(".temp.wav"):
            continue
        if name.lower().endswith(ALL_EXTS):
            result.append(path)
    return sorted(set(result))


def process_file(model, src: str, ffmpeg: str, temp_dir: str,
                 args, srt_path: str) -> bool:
    """处理单个文件：转码 -> 识别 -> 写 SRT。"""
    print(f"\n{'=' * 68}")
    print(f"▶ {src}")
    print(f"{'=' * 68}")
    t0 = time.time()
    wav_path = None
    try:
        print("  [1/2] 提取音频（16kHz 单声道）...")
        wav_path = normalize_to_wav(src, ffmpeg, temp_dir)

        print("  [2/2] 识别中...")
        res = model.generate(
            input=wav_path,
            language=args.language,          # 默认 auto：由 SenseVoice 自行判定语种
            use_itn=True,                    # 输出数字/标点，字幕更可读
            sentence_timestamp=True,         # 关键：不加这个拿不到任何时间戳
            batch_size_s=300,                # 长音频分批，显著提速
        )

        entries = build_srt_entries(res, wav_path)
        if not entries:
            print("  ⚠️ 未识别到内容，跳过")
            return False

        write_srt(entries, srt_path)
        dur = wav_duration_ms(wav_path) / 1000.0
        elapsed = time.time() - t0
        print(f"  ✅ 已生成 {srt_path}")
        print(f"     {len(entries)} 条字幕 | 音频 {dur:.1f}s | 耗时 {elapsed:.1f}s"
              f" | 约 {elapsed / dur:.3f}x 音频时长")
        return True

    except Exception as e:
        print(f"  ❌ 处理失败: {type(e).__name__}: {e}")
        return False
    finally:
        # 清理临时音频；删除失败不能影响主流程
        if wav_path and not args.keep_temp:
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except OSError as e:
                print(f"     （临时文件清理失败，可手动删除：{wav_path} — {e}）")


def resolve_outdir(args) -> str:
    """决定字幕输出目录。

    --outdir 优先；--side-by-side 返回 None（表示与源文件同目录）；
    否则默认集中输出到 ../subtitle，避免字幕散落在音频/视频目录里。
    """
    if args.outdir:
        return os.path.abspath(args.outdir)
    if args.side_by_side:
        return None
    return DEFAULT_OUTDIR


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量把视频/音频转成 SRT 字幕（FunASR SenseVoiceSmall）")
    parser.add_argument("dirs", nargs="*", default=None,
                        help="待处理目录（可多个），默认为 ../audio 与 ../video")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归处理子目录")
    parser.add_argument("-f", "--force", action="store_true",
                        help="覆盖已存在的 .srt（默认跳过，避免白跑）")
    parser.add_argument("--outdir", default=None,
                        help="字幕输出目录（默认集中到 ../subtitle）")
    parser.add_argument("--side-by-side", action="store_true",
                        help="字幕与源文件放同一目录（默认集中到 subtitle/）")
    parser.add_argument("--language", default="auto",
                        help="识别语言：auto / zh / en / yue / ja / ko（默认 auto）")
    parser.add_argument("--max-line", type=int, default=20,
                        help="单行字幕最大字数（默认 20）")
    parser.add_argument("--keep-temp", action="store_true", help="保留中间 wav，便于排查")
    parser.add_argument("--list", action="store_true", help="只列出待处理文件，不识别")
    args = parser.parse_args()

    # 默认扫 audio/ + video/；显式传目录时不存在的目录视为错误
    if args.dirs:
        roots = [os.path.abspath(d) for d in args.dirs]
        missing = [r for r in roots if not os.path.isdir(r)]
        if missing:
            for m in missing:
                print(f"❌ 目录不存在：{m}")
            return 2
    else:
        roots = [r for r in DEFAULT_INPUT_DIRS if os.path.isdir(r)]
        if not roots:
            print("❌ 默认素材目录不存在：")
            for r in DEFAULT_INPUT_DIRS:
                print(f"  - {r}")
            return 2

    files = []
    for root in roots:
        files.extend(collect_files(root, args.recursive))
    files = sorted(set(files))

    if not files:
        print("在这些目录下没有找到视频/音频文件：")
        for r in roots:
            print(f"  - {r}")
        print(f"支持的扩展名：{', '.join(ALL_EXTS)}")
        return 0

    print("待处理目录：")
    for r in roots:
        print(f"  - {r}")
    print(f"找到 {len(files)} 个文件：")
    for f in files:
        print(f"  - {f}")
    if args.list:
        return 0

    outdir = resolve_outdir(args)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        print(f"字幕输出目录：{outdir}")

    # 先挑出真正要跑的文件，全部已存在就提前退出，省掉模型加载时间
    todo, skipped, seen = [], [], {}
    for src in files:
        if outdir:
            srt_path = os.path.join(outdir,
                                    os.path.splitext(os.path.basename(src))[0] + ".srt")
        else:
            srt_path = os.path.splitext(src)[0] + ".srt"

        # audio/ 与 video/ 下的同名文件会撞到同一个字幕文件，提前拦截避免静默覆盖
        if srt_path in seen:
            print(f"⚠️ 字幕名冲突，已跳过：{src}")
            print(f"     （它与 {seen[srt_path]} 都会生成 {srt_path}）")
            continue
        seen[srt_path] = src

        if os.path.exists(srt_path) and not args.force:
            skipped.append(srt_path)
        else:
            todo.append((src, srt_path))

    if skipped:
        print(f"\n已存在字幕、自动跳过 {len(skipped)} 个（加 --force 可覆盖）：")
        for s in skipped:
            print(f"  - {s}")
    if not todo:
        print("\n没有需要处理的文件。")
        return 0

    try:
        ffmpeg = find_ffmpeg()
    except RuntimeError as e:
        print(f"❌ {e}")
        return 2

    print("\n正在加载语音识别模型...")
    from funasr import AutoModel  # 延迟导入：--list 模式无需加载 torch

    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        # 刻意不配 punc_model：SenseVoice 自身已输出标点，
        # 叠加 ct-punc 会产生重复标点（实测出现"，。"连排）
        disable_update=True,
    )
    print("模型加载完成！")

    temp_dir = tempfile.mkdtemp(prefix="batchsub_")
    ok, failed = 0, []
    t_all = time.time()
    try:
        for src, srt_path in todo:
            if process_file(model, src, ffmpeg, temp_dir, args, srt_path):
                ok += 1
            else:
                failed.append(src)
    finally:
        # 兜底清掉临时目录（norm 失败时可能在里面留下半成品）
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)

    total = time.time() - t_all
    print(f"\n{'=' * 68}")
    print(f"处理完成：成功 {ok}/{len(todo)} | 总耗时 {total:.1f}s")
    if failed:
        print("失败列表：")
        for f in failed:
            print(f"  - {f}")
    print(f"{'=' * 68}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
