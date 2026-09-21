# -*- coding: utf-8 -*-
"""
RAG（检索增强生成）知识库模块 —— 语音智能体的"记忆"

【这个文件是干什么的】
把用户上传的文档拆成小块 → 用 embedding 模型把每块变成一串数字（向量）→ 存进本地
向量数据库 → 用户提问时把问题也变成向量 → 找出最相似的几块，喂给大模型当"参考资料"。
这就是 RAG（Retrieval-Augmented Generation，检索增强生成）的全部含义：
先检索、再生成，让大模型回答基于你的文档，而不是凭空编。

【为什么不把整篇文档直接丢给大模型】
1. 长度限制：文档几千上万字，模型一次吃不下；
2. 成本：按字数计费，每次问答都塞全文会非常贵；
3. 准确度：无关内容会干扰模型，检索出最相关的几段反而答得更准。

【设计约定（和 app.py 保持一致）】
- 重依赖（sentence_transformers / chromadb）不在模块顶层 import，改为首次使用时
  加载。否则 app.py 启动会凭空慢十几秒，端口迟迟不就绪。
- 向量库落在项目根目录 rag_db/，不进系统临时目录，重启服务数据还在。
"""

import hashlib
import os
import re
import threading

# ---------------------------------------------------------------- 路径配置
# 本文件在 <项目根>/code/rag_module.py，所以往上一级就是项目根
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
RAG_DB_DIR = os.path.join(PROJECT_ROOT, "rag_db")      # 向量库持久化目录

# ---------------------------------------------------------------- 模型配置
# bge-small-zh-v1.5 是智源研究院开源的轻量中文向量模型（约 95MB）。
# 选它的理由：中文检索效果在"小模型"里最好、CPU 也能跑、体积小下载快。
# 对比：text2vec-base-chinese 效果略弱；bge-large-zh 效果好但 1.3GB 且慢 3 倍。
EMB_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# bge 系列官方建议：给"被检索的问题"加一段指令前缀，检索命中率会明显提升
# （文档侧不加前缀，只有问题侧加）。这是官方说明里的用法，不是玄学。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# HuggingFace 在国内直连基本下不动，默认走 hf-mirror 镜像；
# 若你已配置代理，可用环境变量 HF_ENDPOINT 覆盖（setdefault 不覆盖已有值）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------- 分块参数
# 为什么不是说明书里的"每 500 字硬切一刀"：
# 硬切会把一句话拦腰截断（比如把"这个产品怎么用？"切成了"这个产品怎"/"么用？"），
# 检索出来的碎片语义不完整，大模型看着也费劲。改成"先按段落、再按句子"切：
# - 优先在空行（段落）处断开，段落本身就是语义单元；
# - 段落太长时在句号/问号/感叹号处断，不切碎句子；
# - 相邻块之间留 80 字重叠，避免答案正好卡在两块交界处被丢掉。
CHUNK_SIZE = 400        # 每块目标字数
CHUNK_OVERLAP = 80      # 相邻块重叠字数

# 相似度下限：低于这个分数的片段视为"跟问题无关"，直接丢掉。
# 为什么必须有这道闸门：向量的相似度不会归零——随便问一句"今天天气怎么样"，
# 也能在知识库里找到 0.3 分左右的片段。如果不拦，这些垃圾片段会一起喂给大模型，
# 导致它硬编出一个答案。实测（bge-small-zh）：真正相关的问题得分 0.45~0.75，
# 无关问题普遍 ≤0.35，所以在 0.35 附近设闸门，区分度刚好。
# 可用环境变量 RAG_MIN_SCORE 覆盖，比如文档很短、得分普遍偏低时调到 0.3。
MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.35"))

# ---------------------------------------------------------------- 单例与锁
_engine = None
_engine_lock = threading.Lock()   # 防止两个请求同时初始化，把模型加载两遍


class RAGEngine:
    """RAG 引擎：负责知识入库与检索。"""

    def __init__(self, persist_dir: str = RAG_DB_DIR):
        # ---- 懒加载：只有真正用到时才 import 重依赖 ----
        import chromadb                                    # 向量数据库
        from sentence_transformers import SentenceTransformer   # 文本转向量

        os.makedirs(persist_dir, exist_ok=True)
        # PersistentClient = 数据写磁盘，重启服务不丢；对应内存版是 Client()
        self.client = chromadb.PersistentClient(path=persist_dir)
        # collection 相当于数据库里的"一张表"，名字叫 knowledge
        # 注意：cosine 余弦距离是文本检索的通用选择（只看方向不看长度）
        self.collection = self.client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        # 首次加载会从镜像下载模型（约 95MB），之后走本地缓存秒开
        self.embedder = SentenceTransformer(EMB_MODEL_NAME)
        self._lock = threading.Lock()   # chromadb 的写操作加锁，避免并发写坏索引

    # ------------------------------------------------------------ 内部工具
    def _embed(self, texts: list, is_query: bool = False) -> list:
        """把文本列表转成向量列表。

        is_query=True 时给每条文本加检索指令前缀（见 QUERY_INSTRUCTION 说明）。
        normalize_embeddings=True 把向量长度归一化，这样余弦相似度计算更稳定，
        是 bge 模型官方推荐的用法。
        """
        if is_query:
            texts = [QUERY_INSTRUCTION + t for t in texts]
        vecs = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,   # 关掉进度条，否则日志里全是刷屏的进度条
        )
        return [v.tolist() for v in vecs]

    @staticmethod
    def _split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
        """把长文切成带重叠的小块。返回字符串列表。

        切分顺序：标题 → 段落 → 句子 → 兜底硬切，越靠前越优先。
        为什么先按标题切：文档里的 ## 小标题天然就是主题边界，按它切出来的块
        "一个块聊一件事"，检索时能精准命中（比如问"字幕被拉长怎么办"，
        直接命中"常见问题"那一节，而不会把整章塞进同一块导致关键词被稀释）。
        """
        text = (text or "").replace("\r\n", "\n").strip()
        if not text:
            return []

        # ---------- 第一层：按 markdown 标题切"章节" ----------
        # 标题行归到它下面的内容里（一起切走，标题对检索有提示作用）
        sections, cur = [], []
        for line in text.split("\n"):
            if re.match(r"^\s*#{1,6}\s", line) and cur:
                sections.append("\n".join(cur))
                cur = [line]
            else:
                cur.append(line)
        if cur:
            sections.append("\n".join(cur))
        sections = [s.strip() for s in sections if s.strip()]

        _SENT_END = "。！？!?；;\n"

        def to_units(block: str, limit: int) -> list:
            """把一个章节拆成不超过 limit 字的'原子单元'（段落 → 句子）。"""
            units = []
            for para in [p.strip() for p in block.split("\n\n") if p.strip()]:
                if len(para) <= limit:
                    units.append(para)
                    continue
                buf = ""
                for ch in para:
                    buf += ch
                    if ch in _SENT_END:
                        if buf.strip():
                            units.append(buf.strip())
                        buf = ""
                if buf.strip():
                    units.append(buf.strip())     # 末句没有标点也收
            return units

        # ---------- 第二层：把单元贪心合并成 ≈size 的块（章节之间不跨） ----------
        chunks = []
        for sec in sections:
            units = to_units(sec, size)
            cur_chunk = ""
            for unit in units:
                if not cur_chunk:
                    cur_chunk = unit
                elif len(cur_chunk) + len(unit) + 1 <= size:
                    cur_chunk += "\n" + unit
                else:
                    chunks.append(cur_chunk)
                    # 重叠：把上一块结尾 overlap 字垫到新块开头，
                    # 防止答案正好卡在两块交界处被漏掉
                    tail = cur_chunk[-overlap:] if overlap and len(cur_chunk) > overlap else ""
                    cur_chunk = (tail + "\n" + unit) if tail else unit
            if cur_chunk.strip():
                chunks.append(cur_chunk)

        # ---------- 第三层：兜底硬切（单个单元就超长，比如整段没有标点的口述稿） ----------
        final = []
        for c in chunks:
            if len(c) <= size * 2:
                final.append(c)
            else:
                for i in range(0, len(c), size):
                    final.append(c[i:i + size])
        return [c.strip() for c in final if c.strip()]

    # ------------------------------------------------------------ 文档解析
    # 多格式支持：Word / PDF / Markdown / TXT。解析器按需 import（与整个模块
    # 的懒加载约定一致：没上传过 Word/PDF 就不会加载 python-docx / pypdf）。
    _TEXT_EXTS = (".txt", ".md", ".markdown", ".csv", ".json", ".log", ".srt")

    def _read_text(self, path: str) -> str:
        """读纯文本：先试 utf-8，失败再试 gbk（Windows 记事本默认存 GBK，很常见）。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="gbk", errors="ignore") as f:
                return f.read()

    @staticmethod
    def _read_docx(path: str) -> str:
        """读 Word 文档（.docx）。表格内容也收进来——说明书类文档常把
        关键参数放在表格里，只读段落会漏掉。注意：python-docx 读不了
        老版二进制 .doc，报错时由调用方提示"另存为 .docx"。"""
        from docx import Document
        doc = Document(path)
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))   # 一行表格拼成一行文字
        return "\n".join(parts)

    @staticmethod
    def _read_pdf(path: str) -> str:
        """读 PDF 文本。扫描版 PDF（图片型）extract_text 会返回空，
        由 add_file 统一给出"内容为空"的提示。"""
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                pages.append(t)
        return "\n".join(pages)

    @staticmethod
    def _read_doc(path: str) -> str:
        """读老版二进制 .doc（Word 97-2003）。纯 Python 解析，不依赖 Word/WPS。

        原理：.doc 是 OLE 复合文档（类似一个老式小文件系统），正文存在名为
        WordDocument 的流里，流开头有一张"档案卡"（FIB）记录正文在哪。
        1. 路线一（绝大多数 .doc 都走这条）：分片表（piece table）在 0Table /
           1Table 流里，把正文按"片"取出——压缩片是 8 位单字节（西文），
           非压缩片是 16 位 Unicode（中文在这类片里）；
        2. 路线二（极老格式没有分片表）：直接按 FIB 里的 fcMin..fcMac 区间读。
        两个条件以 FIB 里的标志位为准，都是公开的文档格式规范，不是猜的。
        解不出来就抛错，提示转 .docx——比硬解出乱码塞进知识库要好。
        """
        import struct
        import olefile
        try:
            ole = olefile.OleFileIO(path)
        except Exception as e:
            raise ValueError("这不是有效的 Word 文档（文件损坏或已加密）") from e
        try:
            wd = ole.openstream("WordDocument").read()
            # 0xA5EC 是 .doc 魔数，流开头必须是它，否则根本不是 .doc
            if len(wd) < 0x200 or struct.unpack_from("<H", wd, 0)[0] != 0xA5EC:
                raise ValueError("不是标准的 .doc 文件")
            # 标志位在 0x0A：0x0200 选 0Table/1Table，0x1000 表示正文是 16 位 Unicode
            flags = struct.unpack_from("<H", wd, 0x0A)[0]
            fcMin, fcMac = struct.unpack_from("<II", wd, 0x18)
            ccpText = struct.unpack_from("<i", wd, 0x4C)[0]   # 正文总字数（不含脚注等）

            text = ""
            # ---------- 路线一：分片表 ----------
            # fcClx/lcbClx 指向分片表在 Table 流里的位置（格式规范固定偏移）
            fcClx, lcbClx = struct.unpack_from("<II", wd, 0x1A2)
            if lcbClx:
                tbl = ole.openstream("1Table" if flags & 0x0200 else "0Table").read()
                clx = tbl[fcClx:fcClx + lcbClx]
                pos, parts, total = 0, [], 0
                while pos < len(clx):
                    tag = clx[pos]
                    if tag == 1:                    # 修订记录块，与正文无关，跳过
                        pos += 3 + struct.unpack_from("<H", clx, pos + 1)[0]
                    elif tag == 2:                  # 正文分片表
                        lcb = struct.unpack_from("<I", clx, pos + 1)[0]
                        plc = clx[pos + 5:pos + 5 + lcb]
                        n = (lcb - 4) // 12         # n+1 个字符位置 + n 个片描述
                        cps = struct.unpack_from("<%dI" % (n + 1), plc, 0)
                        base = 4 * (n + 1)
                        for i in range(n):
                            fc_raw = struct.unpack_from("<I", plc, base + 8 * i + 2)[0]
                            cch = cps[i + 1] - cps[i]
                            if fc_raw & 0x40000000:  # 压缩片：8 位单字节
                                off = (fc_raw & 0x3FFFFFFF) // 2
                                seg = wd[off:off + cch].decode("cp1252", "replace")
                            else:                    # 非压缩片：16 位 Unicode（中文在这）
                                off = fc_raw & 0x3FFFFFFF
                                seg = wd[off:off + 2 * cch].decode("utf-16-le", "replace")
                            if 0 < ccpText:
                                seg = seg[:max(0, ccpText - total)]
                            parts.append(seg)
                            total += len(seg)
                            if 0 < ccpText <= total:
                                break
                        break
                    else:
                        break
                text = "".join(parts)

            # ---------- 路线二：无分片表的老格式，直接读正文区间 ----------
            if not text and 0 < fcMin <= fcMac <= len(wd):
                raw = wd[fcMin:fcMac]
                if flags & 0x1000:
                    text = raw.decode("utf-16-le", "replace")
                else:
                    text = raw.decode("gbk", "replace")   # 中文旧文档按 GBK 解

            # 清理：表格单元格分隔符(0x07)转竖线、回车转换行、去掉控制字符
            text = (text or "").replace("\x07", " | ").replace("\r", "\n").replace("\x0b", "\n")
            return "".join(ch for ch in text if ch >= " " or ch in "\n\t")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError("老版 .doc 解析失败，请用 Word 另存为 .docx 后重新上传") from e
        finally:
            try:
                ole.close()
            except Exception:
                pass

    def read_file(self, path: str) -> str:
        """根据扩展名自动选择解析方式，返回全文文本。"""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self._read_pdf(path)
        if ext == ".docx":
            return self._read_docx(path)
        if ext == ".doc":
            return self._read_doc(path)   # 老版二进制格式，纯 Python 解析
        if ext in self._TEXT_EXTS:
            return self._read_text(path)
        raise ValueError(f"不支持的文件格式: {ext}")

    # ------------------------------------------------------------ 写入
    def add_file(self, file_path: str, source_name: str = "") -> dict:
        """把一个文件读进来、切块、向量化、入库。

        file_path：磁盘上的真实路径（可能是服务端保存的临时文件）
        source_name：展示给用户看的原始文件名（临时文件名是一串 uuid，不能拿来展示）

        支持格式：docx / doc / pdf / txt / md / markdown / csv / json / log / srt
        重复上传同一文件名时会先删掉旧的同名知识，避免同一份文档在库里存两份、
        检索出来两条一模一样的结果。
        """
        try:
            content = self.read_file(file_path)
        except ValueError:
            raise   # "不支持的格式 / 请转 .docx" 这类提示原样上抛给接口层
        content = content.strip()
        if not content:
            return {"chunks": 0, "chars": 0, "error": "文件内容为空"}

        name = source_name or os.path.basename(file_path)
        pieces = self._split_text(content)
        if not pieces:
            return {"chunks": 0, "chars": 0, "error": "切分后没有有效内容"}

        # 文档内容指纹：同一个文件内容没变时，可以据此判断
        digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]

        with self._lock:
            # 同名的旧知识先清掉（重新上传 = 覆盖更新）
            try:
                self.collection.delete(where={"source": name})
            except Exception:
                pass   # 首次上传时集合为空，delete 可能报错，忽略即可

            ids = [f"{name}#{digest}#{i}" for i in range(len(pieces))]
            embeddings = self._embed(pieces)
            metadatas = [
                {"source": name, "chunk": i, "digest": digest, "chars": len(p)}
                for i, p in enumerate(pieces)
            ]
            # 分批写：一次性写几万条会爆内存，500 条一批足够稳
            batch = 500
            for s in range(0, len(ids), batch):
                self.collection.add(
                    ids=ids[s:s + batch],
                    embeddings=embeddings[s:s + batch],
                    documents=pieces[s:s + batch],
                    metadatas=metadatas[s:s + batch],
                )

        return {"chunks": len(pieces), "chars": len(content), "digest": digest}

    def add_text(self, text: str, source_name: str) -> dict:
        """直接把一段文字当知识入库（前端手输知识用，不需要先存成文件）。"""
        tmp = os.path.join(RAG_DB_DIR, "_inline_input.txt")
        os.makedirs(RAG_DB_DIR, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text or "")
        try:
            return self.add_file(tmp, source_name=source_name)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    # ------------------------------------------------------------ 检索
    def search(self, query: str, top_k: int = 3) -> list:
        """检索与问题最相关的 top_k 个知识块。

        返回 [{text, source, score, chunk}]，score 是相似度（0~1，越大越像）。
        chromadb 返回的是"距离"（越小越像），这里换算成相似度给前端展示更直观。
        """
        query = (query or "").strip()
        if not query or self.count() == 0:
            return []

        res = self.collection.query(
            query_embeddings=self._embed([query], is_query=True),
            n_results=max(1, min(top_k, 20)),      # 限制范围，防止前端传个 1000
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 1.0
            # 余弦距离 0（完全一样）~ 2（完全相反）；换算成 1~0 的相似度
            score = round(max(0.0, 1.0 - float(dist)), 4)
            # 相关性闸门：低于 MIN_SCORE 的片段直接丢，宁可回答"没找到"，
            # 也不要把无关内容喂给大模型让它硬编（见 MIN_SCORE 的说明）
            if score < MIN_SCORE:
                continue
            out.append({
                "text": doc,
                "source": (meta or {}).get("source", "未知来源"),
                "chunk": (meta or {}).get("chunk", 0),
                "score": score,
            })
        return out

    # ------------------------------------------------------------ 管理
    def count(self) -> int:
        """库里一共有多少个知识块。"""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def list_sources(self) -> list:
        """列出知识库里所有文档名 + 各自块数（前端"知识库现状"用）。"""
        try:
            data = self.collection.get(include=["metadatas"])
        except Exception:
            return []
        counter = {}
        for meta in data.get("metadatas") or []:
            name = (meta or {}).get("source", "未知来源")
            counter[name] = counter.get(name, 0) + 1
        return [{"source": k, "chunks": v} for k, v in sorted(counter.items())]

    def delete_source(self, source_name: str) -> int:
        """按文档名删除知识，返回删除前的块数。"""
        with self._lock:
            before = len([s for s in self.list_sources() if s["source"] == source_name])
            try:
                self.collection.delete(where={"source": source_name})
            except Exception:
                pass
            return before

    def reset(self):
        """清空知识库（危险操作，接口层要二次确认）。"""
        with self._lock:
            try:
                self.client.delete_collection("knowledge")
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name="knowledge", metadata={"hnsw:space": "cosine"}
            )


def get_rag_engine() -> RAGEngine:
    """获取全局 RAG 引擎（单例）。

    首次调用会加载 embedding 模型（约 3~10 秒，含首次下载 95MB），
    之后所有请求共用同一个实例。
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:       # 双重检查：锁内再判一次，防止重复创建
                _engine = RAGEngine()
    return _engine


def is_ready() -> bool:
    """引擎是否已加载（不触发加载）。/health 探查用。"""
    return _engine is not None
