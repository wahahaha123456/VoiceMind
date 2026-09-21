@echo off
REM ============================================================
REM  Decode with a finetuned checkpoint (merge + decode + baseline compare).
REM  Usage: decode_local.bat [outputs\model.pt.avg3]
REM
REM  Important: the raw finetune checkpoint contains ONLY the trainable
REM  modules (audio_encoder / audio_adaptor / ctc); frozen LLM weights are
REM  skipped at save time (see train log: "key: llm.xxx ... not save it").
REM  Feeding that checkpoint straight to AutoModel(init_param=...) overrides
REM  the base weight path, leaving the LLM random-initialized -> garbage tokens.
REM  So we merge it into the base model first (merge_ckpt.py), then decode.
REM ============================================================
cd /d %~dp0

set PYTHON=..\..\funasr_env\Scripts\python.exe
set CKPT=%1
if "%CKPT%"=="" set CKPT=outputs/model.pt.avg3

echo [1/3] Merging base model + finetuned weights into outputs_merged ...
%PYTHON% merge_ckpt.py ++ckpt="%CKPT%" ++out_dir="outputs_merged"
if errorlevel 1 (pause & exit /b 1)

echo [2/3] Decoding with finetuned model ...
%PYTHON% decode_ft.py ++base_model="outputs_merged" ^
  ++scp_file="..\data\all_wav.scp" ^
  ++output_file="..\data\output_finetuned.txt"

echo [3/3] Decoding baseline (no finetune) for comparison ...
%PYTHON% decode_ft.py ^
  ++base_model="C:/Users/iiiis/.cache/modelscope/models/FunAudioLLM--Fun-ASR-Nano-2512/snapshots/master" ^
  ++scp_file="..\data\all_wav.scp" ^
  ++output_file="..\data\output_baseline.txt"

echo.
echo Results: ..\data\output_finetuned.txt  vs  ..\data\output_baseline.txt
echo Labels : ..\data\all_text.txt
pause
