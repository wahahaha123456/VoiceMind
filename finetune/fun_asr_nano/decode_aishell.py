# -*- coding: utf-8 -*-
"""
Fun-ASR-Nano 批量解码（基线 / 微调通用）。

相对 decode_ft.py 的改进：
  - 支持一次喂多条音频（batch），3000 条 dev 从十几分钟降到几分钟
  - 分批失败自动降级为逐条，不影响整体进度
  - 输出 uid<TAB>text，便于直接算 CER

用法：
  基线:  python decode_aishell.py ++base_model="<基座目录>" \
             ++scp_file="dataset/list/dev_wav.scp" ++output_file=".../output_baseline.txt"
  微调后: 同上再加 ++init_param="outputs_aishell/model.pt.avg3"（或直接指向合并后的模型目录）
"""
import os
import sys
import time

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
    chunk = int(kwargs.get("chunk", 8))
    use_vad = kwargs.get("use_vad", True)
    if isinstance(use_vad, str):
        use_vad = use_vad.lower() in ("1", "true", "yes")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from funasr import AutoModel

    model_kwargs = dict(
        model=base_model,
        trust_remote_code=True,
        remote_code="./model.py",
        device=device,
    )
    # AISHELL-1 句段本身只有 2~15 秒，VAD 分段只是额外开销；关掉更快，
    # 且基线/微调用同一配置，CER 对比依然公平。
    if use_vad:
        model_kwargs["vad_model"] = "fsmn-vad"
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    print(f"[decode] use_vad={use_vad}, chunk={chunk}", flush=True)
    if init_param:
        model_kwargs["init_param"] = init_param
        print(f"[decode] 基座={base_model}\n[decode] 权重={init_param}", flush=True)
    else:
        print(f"[decode] 基线解码（不加载微调权重）: {base_model}", flush=True)

    t0 = time.time()
    model = AutoModel(**model_kwargs)
    print(f"[decode] 模型加载耗时 {time.time()-t0:.1f}s", flush=True)

    items = []
    with open(scp_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                items.append((parts[0], parts[1]))
    print(f"[decode] 待解码 {len(items)} 条，chunk={chunk}", flush=True)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    done = 0
    t1 = time.time()
    with open(output_file, "w", encoding="utf-8") as fout:
        for i in range(0, len(items), chunk):
            group = items[i:i + chunk]
            try:
                res = model.generate(input=[p for _, p in group], cache={},
                                     batch_size=len(group))
                texts = [r.get("text", "") for r in res]
                if len(texts) != len(group):
                    raise ValueError(f"返回条数不符 {len(texts)} != {len(group)}")
            except Exception as e:
                print(f"[decode] 批处理失败({type(e).__name__}: {e})，降级逐条", flush=True)
                texts = []
                for _, p in group:
                    try:
                        texts.append(model.generate(input=[p], cache={},
                                                    batch_size=1)[0].get("text", ""))
                    except Exception as e2:
                        print(f"[decode] 单条失败 {p}: {e2}", flush=True)
                        texts.append("")
            for (uid, _), t in zip(group, texts):
                fout.write(f"{uid}\t{t}\n")
                done += 1
            fout.flush()
            if done % 200 == 0 or done == len(items):
                el = time.time() - t1
                print(f"[decode] {done}/{len(items)}  ({el:.0f}s, {done/max(el,1e-9):.1f} 条/秒)",
                      flush=True)
    print(f"[decode] 完成，总耗时 {(time.time()-t0)/60:.1f} 分钟 -> {output_file}", flush=True)


if __name__ == "__main__":
    main_hydra()
