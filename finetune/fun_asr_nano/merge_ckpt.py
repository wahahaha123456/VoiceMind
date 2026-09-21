# -*- coding: utf-8 -*-
"""
把 Fun-ASR-Nano 微调 checkpoint 合并进基座模型，产出可直接加载的完整模型目录。

背景（踩坑记录）：
  finetune 时用 freeze 冻结了 llm，训练器保存 checkpoint 时会跳过 llm.* 的键
  （训练日志里 3110 条 "key: llm.xxx matching: llm., not save it"），
  所以 outputs/model.pt.avg3 里只有 audio_encoder / audio_adaptor / ctc 的权重。

  而 AutoModel(..., init_param=<ckpt>) 是"覆盖式"加载：一旦显式传入 init_param，
  基座 model.pt 的加载路径就被顶掉，LLM 权重缺失 = 随机初始化 → 解码输出乱码。
  （实测：直接 init_param 解码得到 "regnregnregn..." 这类重复垃圾 token。）

  正确做法就是本脚本：先取基座全量权重，再用微调 checkpoint 覆盖对应键，
  存成一个完整模型目录，之后按普通本地模型加载即可。

用法：
  python merge_ckpt.py                    # 默认 avg3
  python merge_ckpt.py ++ckpt="outputs/model.pt.best" ++out_dir="outputs_merged_best"
"""
import os
import shutil

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

BASE_MODEL_DIR = r"C:\Users\iiiis\.cache\modelscope\models\FunAudioLLM--Fun-ASR-Nano-2512\snapshots\master"


def _to_plain(cfg_item):
    if isinstance(cfg_item, ListConfig):
        return OmegaConf.to_container(cfg_item, resolve=True)
    if isinstance(cfg_item, DictConfig):
        return {k: _to_plain(v) for k, v in cfg_item.items()}
    return cfg_item


def _state_dict(obj):
    """从 checkpoint 里取出真正的 state_dict（兼容 {'state_dict': {...}} / {'model': {...}} / 扁平）"""
    if not isinstance(obj, dict):
        raise TypeError(f"unexpected checkpoint type: {type(obj)}")
    for key in ("state_dict", "model"):
        inner = obj.get(key)
        if isinstance(inner, dict) and inner and all(
            isinstance(v, torch.Tensor) for v in list(inner.values())[:5]
        ):
            return inner
    return obj


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    kwargs = _to_plain(cfg)
    base_dir = kwargs.get("base_dir", BASE_MODEL_DIR)
    ckpt = kwargs["ckpt"]
    out_dir = kwargs["out_dir"]

    print(f"[merge] 基座: {base_dir}")
    print(f"[merge] 微调 checkpoint: {ckpt}")
    print(f"[merge] 输出目录: {out_dir}")

    base = _state_dict(torch.load(os.path.join(base_dir, "model.pt"), map_location="cpu", mmap=True))
    ft = _state_dict(torch.load(ckpt, map_location="cpu", mmap=True))
    print(f"[merge] 基座键数 {len(base)}，微调 checkpoint 键数 {len(ft)}")

    merged = dict(base)
    replaced, added, shape_mismatch = 0, 0, []
    for k, v in ft.items():
        if k in merged:
            if merged[k].shape == v.shape:
                merged[k] = v
                replaced += 1
            else:
                shape_mismatch.append((k, tuple(merged[k].shape), tuple(v.shape)))
        else:
            merged[k] = v
            added += 1
    print(f"[merge] 覆盖 {replaced} 个键，新增 {added} 个键，形状不符 {len(shape_mismatch)} 个")
    for item in shape_mismatch[:5]:
        print("   形状不符:", item)

    os.makedirs(out_dir, exist_ok=True)
    # 复制基座目录里除权重/缓存外的所有内容（配置、remote code、tokenizer）。
    # 注意必须包含 Qwen3-0.6B 子目录：基座 config.yaml 里 tokenizer 用的是相对路径
    # "Qwen3-0.6B"，缺了它 AutoModel 会去连 huggingface.co 并报 OSError。
    SKIP_NAMES = {"__pycache__", ".git", ".msc", ".mdl", "model.pt"}
    copied_dirs = []
    for name in os.listdir(base_dir):
        if name in SKIP_NAMES or name.startswith("model.pt."):
            continue
        src = os.path.join(base_dir, name)
        dst = os.path.join(out_dir, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*SKIP_NAMES))
            copied_dirs.append(name)
        else:
            shutil.copy2(src, dst)
    print(f"[merge] 已复制配置与子目录: {copied_dirs}")

    out_pt = os.path.join(out_dir, "model.pt")
    print(f"[merge] 保存合并权重 → {out_pt}（约 2GB，请稍候）")
    torch.save(merged, out_pt)
    print("[merge] 完成。用 decode_ft.py 时把 base_model 指向该目录、不要传 init_param。")


if __name__ == "__main__":
    main_hydra()
