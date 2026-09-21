# VoiceMind 声智助手

> 基于语音识别与 RAG 的本地语音智能助手，全流程本地运行，数据不出本机。

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![FunASR](https://img.shields.io/badge/FunASR-1.4.14-green)](https://github.com/alibaba-damo-academy/FunASR)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 项目简介

VoiceMind 是一个**桌面级语音智能助手**，集成了语音识别、说话人分离、实时字幕、视频字幕生成、RAG 知识问答等能力，支持音频、视频、Word、PDF 等多种格式，一键启动，双击即用。

**核心价值**：
- **本地运行**：所有模型和数据都在本地，无需联网，隐私安全
- **多场景覆盖**：会议记录、视频字幕、课堂笔记、知识问答
- **工业级精度**：AISHELL-1 标准集 CER **2.80%**，真实会议场景 CER **7.11%**
- **高性能推理**：RTF **0.032**（约 31 倍速），1 小时录音约 2 分钟出稿

---

## 核心能力

| 功能 | 技术实现 | 性能指标 |
|------|---------|---------|
| **语音识别** | Fun-ASR-Nano v2 + SenseVoiceSmall | CER 2.80%（AISHELL-1） |
| **说话人分离** | CAM++ 声纹聚类 | 单人防误分、双人对话身份一致 |
| **实时字幕** | Paraformer-streaming + WebSocket | 首字延迟 1.83s |
| **字幕生成** | FunASR + ffmpeg + 智能断句 | 字级时间戳、SRT/ASS 双输出 |
| **关键词检索** | 向量检索 + 拼音级纠错 | 支持热词 |
| **知识问答** | BGE-small-zh + ChromaDB + DeepSeek | 3 轮会话记忆、无 Key 降级 |

---

## 性能数据

完整测试报告见 [docs/性能测试报告.md](docs/性能测试报告.md)。

### 识别准确率

| 测试集 | 模型 | CER | 整句正确率 |
|--------|------|-----|-----------|
| AISHELL-1 dev（1200条） | 基座 | 3.97% | 59.67% |
| AISHELL-1 dev（1200条） | **v2 微调** | **2.80%** | **63.75%** |
| 真实会议（1500条） | SenseVoiceSmall | 10.30% | 23.87% |
| 真实会议（1500条） | **v2 微调** | **7.11%** | **30.73%** |

**微调收益**：CER 相对下降 **29.5%**，产品场景 CER 相对改善 **31%**。

### 处理速度

| 音频 | 时长 | 处理时间 | RTF |
|------|------|---------|-----|
| 短音频 | 4.1s | 0.18s | 0.044 |
| 长音频 | 43.8s | 0.63s | 0.014 |
| **平均** | — | — | **0.032** |

### 鲁棒性（80条 × 5种条件）

| 条件 | CER |
|------|-----|
| 干净音频 | 2.01% |
| 慢速 | 2.01% |
| 快速 | 1.75% |
| 变调 | 2.01% |
| 加噪 | 4.81% |

---

## 技术架构

```text
┌─────────────────────────────────────────────────────────────┐
│                       用户浏览器                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │ 普通识别 │ │ 说话人分离│ │ 实时录音 │ │ 知识问答 │        │
│  │  (HTTP)  │ │  (HTTP)  │ │(WebSocket)│ │  (HTTP)  │        │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘        │
└───────┼────────────┼────────────┼────────────┼───────────────┘
        │            │            │            │
        ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                FastAPI 后端 (app.py, 端口8000)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │  /asr    │ │ /asr_spk │ │ /search  │ │/voice_qa │        │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘        │
│       └────────────┼────────────┼────────────┘              │
│                    ▼            ▼                            │
│        ┌─────────────────┐ ┌──────────────┐                 │
│        │   FunASR 模型   │ │   RAG 模块   │                 │
│        │   SenseVoice    │ │  BGE+Chroma  │                 │
│        └─────────────────┘ └──────────────┘                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           WebSocket 服务 (ws_server.py, 端口8001)             │
│              Paraformer-streaming 流式识别                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| **语音识别** | FunASR 1.4.14、Fun-ASR-Nano v2、SenseVoiceSmall、Paraformer-streaming |
| **说话人分离** | CAM++ |
| **标点恢复** | CT-Transformer |
| **VAD** | FSMN-VAD |
| **RAG** | BGE-small-zh-v1.5、ChromaDB |
| **LLM** | DeepSeek（OpenAI 兼容接口，可接通义千问等） |
| **后端** | FastAPI、Uvicorn、WebSocket |
| **前端** | HTML/CSS/JS、AudioWorklet |
| **音频处理** | ffmpeg、torchaudio、soundfile |
| **部署** | Windows 一键启动器（Job Object + PID 管理） |

---

## 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 |
| Python | 3.10 |
| 内存 | 16GB 推荐 |
| GPU | 可选，NVIDIA 显卡可加速 |
| ffmpeg | 必须，用于音频格式转换 |

### 安装步骤

```powershell
# 1. 克隆仓库
git clone https://github.com/wahahaha123456/VoiceMind.git
cd VoiceMind

# 2. 创建虚拟环境
python -m venv funasr_env
funasr_env\Scripts\activate

# 3. 安装依赖
pip install funasr modelscope torch torchaudio fastapi uvicorn websockets chromadb sentence-transformers python-docx pypdf

# 4. 安装 ffmpeg
# 下载 https://www.gyan.dev/ffmpeg/builds/
# 解压后把 bin 目录添加到系统 PATH

# 5. 配置 API Key（可选，用于知识问答）
copy code\llm_config.example.json code\llm_config.json
# 编辑 code\llm_config.json，填入你的 DeepSeek 或通义千问 API Key

# 6. 启动
# 双击 启动VoiceMind.bat
# 浏览器会自动打开 http://127.0.0.1:8000/index.html
```

---

## 使用说明

- **普通识别**：上传音频或视频文件，点击「开始识别」，自动生成带时间戳的文本。
- **说话人分离**：上传多人对话音频，系统自动区分不同说话人，输出带角色标签的文本。
- **实时录音**：点击「实时录音」，允许麦克风权限，边说边出字幕。
- **视频字幕**：把视频放到 `video/` 文件夹，点击「批量生成字幕」，自动生成同名 SRT 文件。
- **知识问答**：上传 Word/PDF/Markdown/TXT 文档，用语音提问，系统基于知识库回答。

---

## 项目结构

```text
VoiceMind/
├── code/                    # 后端 Python 代码
│   ├── app.py              # FastAPI 主服务
│   ├── ws_server.py        # WebSocket 流式识别
│   ├── rag_module.py       # RAG 检索模块
│   ├── agent_module.py     # LLM Agent 模块
│   ├── launcher.py         # 一键启动器
│   ├── stop.py             # 停止脚本
│   └── ...
├── web/                     # 前端
│   ├── index.html          # 主页面
│   └── audio-processor.js  # 音频采集
├── finetune/                # 微调脚本
│   ├── fun_asr_nano/
│   │   ├── finetune_local.bat
│   │   ├── lora_local.bat
│   │   └── decode_local.bat
│   └── 微调说明.md          # 微调流程与数据说明
├── docs/                    # 文档
│   ├── 性能测试报告.md
│   └── 示例知识库_产品说明书.md
├── audio/                   # 音频文件（不上传）
├── video/                   # 视频文件（不上传）
├── subtitle/                # 字幕输出（不上传）
├── logs/                    # 日志（不上传）
├── 启动VoiceMind.bat        # 一键启动
├── 停止VoiceMind.bat        # 一键停止
├── .gitignore
├── README.md
└── LICENSE
```

---

## 性能测试报告

完整的性能测试数据见 [docs/性能测试报告.md](docs/性能测试报告.md)，包含：

- 识别准确率（CER）：AISHELL-1 + 真实会议场景
- 处理速度（RTF）：短音频 + 长音频
- 实时延迟：首字响应 1.83s
- 说话人分离：单人防误分 + 双人对话验证
- 鲁棒性：80条 × 5种条件
- 代码规模：9378 行

---

## 开发历程

| 阶段 | 内容 |
|------|------|
| v1.0 | 基础语音识别，支持音频转文字 |
| v2.0 | 集成说话人分离、实时录音、关键词检索 |
| v3.0 | 接入 RAG + LLM，实现知识问答 |
| v4.0 | 桌面化打包，一键启动 |
| v5.0 | 微调 Fun-ASR-Nano，CER 2.80% |

---

## 许可证

MIT License，详见 [LICENSE](LICENSE)。

## 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 阿里达摩院语音识别工具包
- [ModelScope](https://modelscope.cn/) - 模型托管平台
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [BGE](https://huggingface.co/BAAI/bge-small-zh-v1.5) - 中文 Embedding 模型

## 联系方式

- GitHub: [@wahahaha123456](https://github.com/wahahaha123456)
- 项目地址: https://github.com/wahahaha123456/VoiceMind
