# VoiceMind 声智助手

基于语音识别与 RAG 的本地语音智能助手：实时转录、说话人分离、字幕批量生成、知识库问答，全流程在本机运行，数据不出本机。

## 功能特性

- **语音识别**：上传音频/视频自动转写，支持中英日韩，默认使用本地微调的 Fun-ASR-Nano v2，SenseVoiceSmall 作为兜底
- **说话人分离**：基于 CAM++ 区分多人对话，输出带角色标签的分段文稿
- **实时录音**：边说边出字幕，基于 paraformer-zh-streaming + WebSocket 流式识别
- **字幕工作台**：批量或单个音视频生成同名 SRT + ASS 字幕，出声对齐、智能断句、热词拼音级纠错，剪映/PR 导入即用
- **关键词检索**：在录音/视频中搜索关键词并定位时间点，支持精确与模糊匹配
- **知识库问答**：上传 Word / PDF / Markdown / TXT 建成知识库，用语音或文字提问，检索后由大模型基于文档作答

## 技术栈

| 层级 | 技术 |
|------|------|
| 语音识别 | FunASR、Fun-ASR-Nano（微调）、SenseVoiceSmall（兜底） |
| 流式识别 | paraformer-zh-streaming + WebSocket |
| 语音处理 | fsmn-vad（端点检测）、ct-punc（标点恢复）、CAM++（说话人） |
| RAG | BAAI/bge-small-zh-v1.5（向量化）、ChromaDB（向量库） |
| LLM | DeepSeek（OpenAI 兼容接口，可换其他兼容服务） |
| 后端 | FastAPI（:8000）、WebSocket（:8001） |
| 前端 | 原生 HTML/CSS/JS、AudioWorklet |

## 快速开始

### 1. 环境要求

- Python 3.10（本项目使用虚拟环境 `funasr_env`）
- ffmpeg（音视频转码，需在 PATH 中）
- 内存 16GB 推荐；有 NVIDIA 显卡更快（本项目在 RTX 4060 Laptop 8GB 上验证）

### 2. 安装依赖

```bash
python -m venv funasr_env
funasr_env\Scripts\activate
pip install funasr modelscope torch torchaudio fastapi uvicorn websockets ^
            chromadb sentence-transformers openai rapidfuzz pypinyin ^
            python-docx pypdf olefile markdown numpy
```

> 提示：`numpy` 请锁在 1.26.x（2.x 与 funasr 部分组件不兼容）。

### 3. 配置 API Key

复制 `code/llm_config.example.json` 为 `code/llm_config.json`，填入自己的 API Key：

```bash
copy code\llm_config.example.json code\llm_config.json
```

也可以启动后在前端「语音问答」页底部直接填写，服务会热重载，无需重启。
**未填 Key 时系统自动降级**：只做检索并返回知识库原文，不会报错。

### 4. 启动

双击项目根目录的 `启动VoiceMind.bat`，浏览器会自动打开 `http://127.0.0.1:8000/index.html`。
停止服务：双击 `停止VoiceMind.bat`。

首次使用某个模型时会自动从 ModelScope / HuggingFace 镜像下载（几百 MB 到 1GB+），需要等待。

### 5. 关于 ASR 模型

`code/app.py` 中 `_ASR_MODEL_V2_DIR` 指向本地微调模型目录，该目录**不包含在仓库中**。
如果没有自己的微调模型，把 `ASR_MODEL_NAME` 改为 `iic/SenseVoiceSmall`（或注释掉 v2 分支）即可直接用通用模型跑起来。

## 项目结构

```
VoiceMind/
├── code/                    # 后端脚本
│   ├── app.py               # FastAPI 后端（识别 / 字幕 / RAG / Agent 接口）
│   ├── ws_server.py         # WebSocket 流式识别服务（:8001）
│   ├── rag_module.py        # RAG 模块：多格式解析、分块、向量检索
│   ├── agent_module.py      # LLM Agent：检索增强问答、会话记忆、降级
│   ├── batch_subtitle.py    # 批量字幕命令行工具
│   ├── launcher.py          # 一键启动器（无窗口常驻）
│   ├── stop.py              # 停止脚本
│   └── llm_config.example.json
├── web/                     # 前端
│   ├── index.html
│   └── audio-processor.js
├── docs/                    # 文档
├── finetune/                # 微调脚本与说明
├── 启动VoiceMind.bat
├── 停止VoiceMind.bat
└── README.md
```

## 使用说明

- **普通识别**：上传音频/视频，点击「开始识别」，结果可导出 TXT / SRT
- **说话人分离**：选择该模式后上传，输出带「说话人 1/2/3」标签的分段
- **关键词检索**：输入关键词（可加热词），返回每个命中的时间点与上下文
- **实时录音**：点击开始，授权麦克风，字幕逐句流出（仅限 127.0.0.1 / localhost）
- **字幕工作台**：选文件夹批量、或选单个文件单独生成，输出同名 `.srt` + `.ass`
- **语音问答**：先上传文档建知识库，再录音/传音频/打字提问

## 常见问题

- **字幕导入剪映被自动拉长**：用生成的 `.ass` 文件（SRT 有间隙时剪映会兜底填充）
- **热词不生效**：热词已做拼音级校正，人名同音字（如「周潇济/周潇齐」）也能纠；仍不生效请确认模型是否被正确加载
- **服务起不来**：查看 `logs/launcher.log`、`logs/app.log`

## 许可证

MIT License

## 致谢

- [FunASR](https://github.com/modelscope/FunASR) — 阿里达摩院语音识别工具包
- [ModelScope](https://modelscope.cn/) — 模型托管平台
- [ChromaDB](https://www.trychroma.com/) — 向量数据库
- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) — 中文向量模型
