# -*- coding: utf-8 -*-
"""
Fun-ASR-Nano 微调结果解码（正确加载方式）。

为什么需要这个脚本：
  finetune 时冻结的模块（audio_encoder / llm）权重不会写进 checkpoint
  （训练日志里大量 "key: llm.xxx matching: llm., not save it"），
  所以官方 decode.py 那种「只传一个模型目录」的方式加载微调权重会得到乱码输出。
  正确做法：model = 基座模型（或本地基座目录），init_param = 微调 checkpoint。

用法：
  # 微调后
  python decode_ft.py ++base_model="<基座目录>" ++init_param="outputs/model.pt.avg3" ^
      ++scp_file="data/all_wav.scp" ++output_file="data/output_finetuned.txt"
  # 基线（不加 init_param）
  python decode_ft.py ++base_model="<基座目录>" ^
      ++scp_file="data/all_wav.scp" ++output_file="data/output_baseline.txt"
"""
import os

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def _to_plain(cfg_item):
    if isinstance(cfg_item, ListConfig):
        return OmegaConf.to_container(cfg_item, resolve=True)
    if isinstance(cfg_item, DictConfig):
        return {k: _to_plain(v) for k, v in cfg_item.items()}
    return cfg_item


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    kwargs = _to_plain(cfg)

    base_model = kwargs.get("base_model", "FunAudioLLM/Fun-ASR-Nano-2512")
    init_param = kwargs.get("init_param", "") or ""
    scp_file = kwargs["scp_file"]
    output_file = kwargs["output_file"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    from funasr import AutoModel

    model_kwargs = dict(
        model=base_model,
        trust_remote_code=True,
        remote_code="./model.py",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
    )
    if init_param:
        model_kwargs["init_param"] = init_param
        print(f"[decode_ft] 基座: {base_model}\n[decode_ft] 微调权重: {init_param}")
    else:
        print(f"[decode_ft] 基线解码（未加载微调权重）: {base_model}")

    model = AutoModel(**model_kwargs)

    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(scp_file, "r", encoding="utf-8") as f1, \
         open(output_file, "w", encoding="utf-8") as f2:
        for line in f1:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                text = model.generate(input=[parts[1]], cache={}, batch_size=1)[0]["text"]
                f2.write(f"{parts[0]}\t{text}\n")
                print(f"{parts[0]}\t{text[:80]}")


if __name__ == "__main__":
    main_hydra()
