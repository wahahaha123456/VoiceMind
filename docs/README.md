# docs —— 说明文档

## 文档索引

| 文档 | 位置 | 内容 |
|------|------|------|
| 项目结构说明 | [`../项目结构说明.md`](../项目结构说明.md) | 目录用途、启动命令、接口清单、注意事项 |

## 本目录约定

- 项目级的说明、设计决策、排错笔记都放这里，文件名用中文或英文均可。
- 详细的启动与接口说明统一维护在根目录的 `项目结构说明.md`，避免两处内容不一致。

## 一句话速查

```powershell
cd C:\Users\iiiis\Desktop\FunASR

# 1) Web 服务：页面 + 上传识别 + 说话人分离 + 导出  → http://127.0.0.1:8000/index.html
funasr_env\Scripts\python code\app.py

# 2) 流式识别：实时录音字幕  → ws://127.0.0.1:8001
funasr_env\Scripts\python code\ws_server.py

# 3) 批量字幕：扫描 audio/ + video/，输出到 subtitle/
funasr_env\Scripts\python code\batch_subtitle.py
```

> 两个服务需要在**两个终端**里各开一个，Web 界面才能同时支持文件上传和实时录音。
