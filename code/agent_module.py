# -*- coding: utf-8 -*-
"""
Agent（智能体）模块 —— 语音智能体的"大脑"

【这个文件是干什么的】
拿 RAG 检索出来的知识块 + 用户的问题，拼成一段提示词（prompt）发给大模型，
让大模型基于资料回答。它比"直接问大模型"多了三道约束：
1. 只根据给定资料答，资料里没有就说不知道（防止胡编）；
2. 回答里标注来源文档名（可追溯）；
3. 保留最近几轮对话（让"它怎么用？"这种省略主语的追问也能答对）。

【为什么用 OpenAI 兼容接口而不是各家 SDK】
DeepSeek、通义千问、智谱等国内厂商都提供了"OpenAI 兼容接口"——
意思是它们的调用格式和 OpenAI 一模一样，只是把地址（base_url）换掉。
所以一份代码能接所有厂商，换模型只改配置不改代码。

【API Key 放哪】
优先级：环境变量 DEEPSEEK_API_KEY > code/llm_config.json 文件里的 api_key。
不建议把 Key 写死在代码里（代码会被复制/分享，Key 就泄漏了）。
没有 Key 时不会报错崩溃，而是退化成"只返回检索到的原文"，保证链路可测。
"""

import json
import os
import threading

# ---------------------------------------------------------------- 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # .../code
PROJECT_ROOT = os.path.dirname(BASE_DIR)                         # 项目根
CONFIG_PATH = os.path.join(BASE_DIR, "llm_config.json")          # 配置文件位置

# ---------------------------------------------------------------- 默认配置
# deepseek-chat = DeepSeek 的通用对话模型，中文强、便宜（比同级别便宜一个数量级）
# base_url 用官方地址；如果你的网络环境需要代理，改这里的 v1 结尾即可
DEFAULT_CONFIG = {
    "provider": "deepseek",
    "api_key": "",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "temperature": 0.2,      # 0~2，越小越"保守稳定"。问答场景要的是复述资料，不是创作，所以调低
    "max_tokens": 800,       # 回答最长长度。太长浪费钱，800 字对口语问答足够
    "timeout": 60,           # 单次请求超时（秒）。网络不好时避免一直挂着
    "history_turns": 3,      # 记住最近几轮对话（追问"它怎么用"要靠它）
    "top_k": 3,              # 检索几块知识
}

# 系统提示词：定义大模型的"人设"和"边界"，是回答质量的关键
SYSTEM_PROMPT = """你是一个中文语音助手，通过语音识别接收用户提问。
回答要求：
1. 只依据下面提供的《知识库资料》作答，不要使用资料之外的信息；
2. 如果资料里没有能回答该问题的内容，直接回答"知识库里没有找到相关内容"，不要编造；
3. 资料来自语音识别，可能有错别字，请结合上下文合理理解用户意图；
4. 回答用口语化的简体中文，控制在 200 字以内，不要用 Markdown 符号；
5. 不要重复用户的问题，直接给答案。"""

# 没有配置 API Key 时的退化答复（保证链路可测，而不是整条链路报错）
_NO_KEY_HINT = (
    "（尚未配置大模型 API Key，当前只做了知识检索，下面是资料原文）"
)

_agent = None
_agent_lock = threading.Lock()


class VoiceAgent:
    """语音问答 Agent：把问题 + 检索资料交给大模型，产出答案。"""

    def __init__(self, config: dict = None):
        self.config = dict(DEFAULT_CONFIG)
        # 环境变量优先级最高（部署时不用改文件，设个变量就行）
        env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if env_key:
            self.config["api_key"] = env_key
        # 再读配置文件（存在就合并覆盖）
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    file_cfg = json.load(f)
                for k, v in (file_cfg or {}).items():
                    if v not in (None, ""):
                        self.config[k] = v
            except Exception as e:
                # 配置文件写坏了不能让服务起不来，只打日志
                print(f"[agent] 读取 {CONFIG_PATH} 失败，用默认配置：{e}", flush=True)
        if config:
            self.config.update(config)

        self._client = None

    # ------------------------------------------------------------ 客户端
    @property
    def api_key(self) -> str:
        return (self.config.get("api_key") or "").strip()

    def has_key(self) -> bool:
        """是否配置了可用的 Key。没 Key 时接口不会报错，只会走退化答复。"""
        return bool(self.api_key)

    def _get_client(self):
        """懒加载 OpenAI 兼容客户端。openai 包体积不小，用到才导入。"""
        if self._client is None:
            from openai import OpenAI   # 注意：包名是 openai，不是 OpenAI
            if not self.api_key:
                return None
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.config["base_url"],
                timeout=self.config.get("timeout", 60),
            )
        return self._client

    # ------------------------------------------------------------ 核心问答
    def answer(self, question: str, contexts: list, history: list = None) -> dict:
        """基于检索资料回答。

        question：用户问题（语音识别后的文本）
        contexts：RAG 检索结果 [{text, source, score}]
        history：[{role:'user'|'assistant', content:'...'}]，最近几轮对话

        返回 dict：
          answer        给用户看的文字回答
          used_llm      是否真的调了大模型（False = 退化模式/出错）
          error         出错原因（正常时为 ""）
          sources       引用的资料名列表
        """
        contexts = contexts or []
        sources = []
        for c in contexts:
            name = c.get("source", "未知来源")
            if name not in sources:
                sources.append(name)

        # ---- 情况一：知识库没检索到东西 ----
        # 直接返回固定话术，不花 API 钱去问一个"无资料"的问题
        if not contexts:
            return {
                "answer": "知识库里没有找到相关内容，请先上传相关资料，或换个说法再问一次。",
                "used_llm": False,
                "error": "",
                "sources": [],
            }

        # ---- 情况二：没有 Key，退化模式 ----
        if not self.has_key():
            snippet = "\n\n".join(
                f"【{c.get('source','未知')}】{self._clip(c.get('text',''), 200)}"
                for c in contexts
            )
            return {
                "answer": f"{_NO_KEY_HINT}\n\n{snippet}",
                "used_llm": False,
                "error": "",
                "sources": sources,
            }

        # ---- 情况三：正常调用大模型 ----
        materials = "\n\n".join(
            f"【资料{i + 1}｜来源：{c.get('source','未知')}｜相似度{c.get('score',0)}】\n{c.get('text','')}"
            for i, c in enumerate(contexts)
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # 只带最近 N 轮历史，太长的历史既费钱又干扰当前问题
        for h in (history or [])[-self.config.get("history_turns", 3) * 2:]:
            role = h.get("role")
            content = (h.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({
            "role": "user",
            "content": f"《知识库资料》\n{materials}\n\n用户问题：{question}",
        })

        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=self.config.get("temperature", 0.2),
                max_tokens=self.config.get("max_tokens", 800),
                stream=False,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                text = "模型返回了空内容，请重试一次。"
            return {"answer": text, "used_llm": True, "error": "", "sources": sources}
        except Exception as e:
            # 常见错误：Key 错误(401)、余额不足(402)、网络超时、模型名写错。
            # 全部兜住，返回"错误说明 + 检索原文"，用户至少能看到资料。
            err = self._friendly_error(e)
            snippet = "\n\n".join(
                f"【{c.get('source','未知')}】{self._clip(c.get('text',''), 200)}"
                for c in contexts
            )
            return {
                "answer": f"{err}\n\n以下是检索到的资料原文：\n\n{snippet}",
                "used_llm": False,
                "error": err,
                "sources": sources,
            }

    # ------------------------------------------------------------ 工具
    @staticmethod
    def _clip(text: str, limit: int) -> str:
        """截断过长文本用于展示。"""
        text = (text or "").strip().replace("\n", " ")
        return text if len(text) <= limit else text[:limit] + "…"

    @staticmethod
    def _friendly_error(e: Exception) -> str:
        """把英文技术报错翻译成看得懂的中文提示，方便定位问题。"""
        msg = str(e)
        low = msg.lower()
        if "401" in msg or "invalid_api_key" in low or "authentication" in low:
            return "大模型鉴权失败（401）：API Key 不正确或已失效，请检查 code/llm_config.json。"
        if "402" in msg or "insufficient" in low or "balance" in low:
            return "大模型账户余额不足（402）：请到 DeepSeek 控制台充值后重试。"
        if "404" in msg or "model_not_found" in low:
            return "模型名不存在（404）：请检查 llm_config.json 里的 model 字段。"
        if "timeout" in low or "timed out" in low:
            return "调用大模型超时：网络不通或响应太慢，请稍后重试。"
        if "connect" in low or "ssl" in low or "proxy" in low:
            return "无法连接大模型服务：请检查网络/代理是否正常。"
        return f"调用大模型出错：{msg[:200]}"


def get_agent() -> VoiceAgent:
    """获取全局 Agent 单例。"""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = VoiceAgent()
    return _agent


def reload_agent():
    """重新读取配置（前端改完 Key 后可调用，无需重启服务）。"""
    global _agent
    with _agent_lock:
        _agent = None
    return get_agent()


def config_status() -> dict:
    """返回当前 LLM 配置状态（**不含 Key 明文**，只回显尾 4 位，避免泄漏）。"""
    a = get_agent()
    key = a.api_key
    return {
        "provider": a.config.get("provider", "deepseek"),
        "model": a.config.get("model", ""),
        "base_url": a.config.get("base_url", ""),
        "has_key": bool(key),
        "key_tail": ("****" + key[-4:]) if len(key) >= 4 else "",
        "config_path": CONFIG_PATH,
    }
