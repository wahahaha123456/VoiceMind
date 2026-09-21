@echo off
REM ============================================================
REM  AISHELL-1 微调结果解码 + CER 评测（Windows）
REM
REM  三步：
REM    1) merge_ckpt.py  把微调 checkpoint 合并进基座 -> outputs_aishell_merged
REM       （必须合并：冻结的 llm 权重不在 checkpoint 里，直接加载会输出乱码）
REM    2) decode_aishell.py 解码评测集
REM    3) compute_cer.py 与基线和人工标注对比
REM
REM  用法: decode_aishell.bat [checkpoint文件名，默认 model.pt.avg3]
REM ============================================================
setlocal
cd /d %~dp0

set PYTHON=..\..\funasr_env\Scripts\python.exe
set BASE=C:/Users/iiiis/.cache/modelscope/models/FunAudioLLM--Fun-ASR-Nano-2512/snapshots/master
set LIST=C:/Users/iiiis/Desktop/FunASR/dataset/list
set CUDA_VISIBLE_DEVICES=0
set PYTHONUNBUFFERED=1

set CKPT=%1
if "%CKPT%"=="" set CKPT=model.pt.avg3

if not exist "outputs_aishell\%CKPT%" (
  echo [ERROR] 找不到 outputs_aishell\%CKPT%
  dir /b outputs_aishell
  pause & exit /b 1
)

echo [1/3] 合并权重 -> outputs_aishell_merged
%PYTHON% merge_ckpt.py ++ckpt="outputs_aishell/%CKPT%" ++out_dir="outputs_aishell_merged" || (pause & exit /b 1)

echo [2/3] 解码评测集
%PYTHON% decode_aishell.py ^
  ++base_model="C:/Users/iiiis/Desktop/FunASR/finetune/fun_asr_nano/outputs_aishell_merged" ^
  ++scp_file="%LIST%/eval_wav.scp" ^
  ++output_file="%LIST%/output_finetuned.txt" ^
  ++chunk=32 ++use_vad=false || (pause & exit /b 1)

echo [3/3] CER 对比
%PYTHON% compute_cer.py "%LIST%/eval_text.txt" "%LIST%/output_baseline.txt" "%LIST%/output_finetuned.txt"

echo.
echo 完成。合并后的模型目录: %~dp0outputs_aishell_merged
pause
