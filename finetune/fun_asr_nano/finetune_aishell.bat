@echo off
REM ============================================================
REM  Fun-ASR-Nano finetune on AISHELL-1 (Windows, single GPU)
REM  Strategy: freeze audio_encoder + llm, train audio_adaptor only
REM            (official recommendation for < 1000 h of data)
REM
REM  Data : C:\Users\iiiis\Desktop\FunASR\dataset\list\train.jsonl  (11986 utts / 15.2 h)
REM         C:\Users\iiiis\Desktop\FunASR\dataset\list\dev.jsonl    (3000 utts / 3.8 h, speaker-disjoint)
REM         labels are the official human-verified AISHELL-1 transcripts
REM  Out  : .\outputs_aishell
REM
REM  NOTE 1: never use torchrun here -- the Windows CUDA torch build has no
REM          libuv and torchrun aborts with "use_libuv was requested but
REM          PyTorch was built without libuv support" (USE_LIBUV=0 does not help).
REM          train_ds reads RANK/LOCAL_RANK/WORLD_SIZE from the environment.
REM  NOTE 2: funasr steps the optimizer only when
REM          (batch_idx + 1) %% accum_grad == 0 -- there is NO end-of-epoch flush.
REM  NOTE 3: the saved checkpoint contains ONLY trainable (non-frozen) weights.
REM          To decode it you must merge it onto the base model first
REM          (see merge_ckpt.py / decode_aishell.bat).
REM ============================================================
setlocal
cd /d %~dp0

set PYTHON=..\..\funasr_env\Scripts\python.exe
set CUDA_VISIBLE_DEVICES=0

REM ---- tunables (8 GB VRAM, measured 0.26 s/step at batch 8 => ~6.5 min/epoch on 15.2 h) ----
set BATCH_SIZE=8
set BATCH_TYPE=example
set ACCUM_GRAD=1
set MAX_EPOCH=3
set LR=0.0001
set SAVE_INTERVAL=1500
set VAL_INTERVAL=1500

set TRAIN_JSONL=C:/Users/iiiis/Desktop/FunASR/dataset/list/train.jsonl
set VAL_JSONL=C:/Users/iiiis/Desktop/FunASR/dataset/list/dev_val150.jsonl
set OUT=./outputs_aishell

set RANK=0
set LOCAL_RANK=0
set WORLD_SIZE=1
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=26669
set PYTHONUNBUFFERED=1

if not exist outputs_aishell mkdir outputs_aishell

echo [1/2] Checking torch CUDA...
%PYTHON% -c "import torch; assert torch.cuda.is_available(), 'torch.cuda not available'; print('CUDA OK:', torch.cuda.get_device_name(0))" || (pause & exit /b 1)

echo [2/2] Starting AISHELL-1 finetune (audio_adaptor, batch=%BATCH_SIZE% %BATCH_TYPE%, epoch=%MAX_EPOCH%)...
%PYTHON% -m funasr.bin.train_ds ^
  ++model="FunAudioLLM/Fun-ASR-Nano-2512" ^
  ++trust_remote_code=true ^
  ++train_data_set_list="%TRAIN_JSONL%" ^
  ++valid_data_set_list="%VAL_JSONL%" ^
  ++dataset_conf.data_split_num=1 ^
  ++dataset_conf.batch_sampler="BatchSampler" ^
  ++dataset_conf.batch_size=%BATCH_SIZE% ^
  ++dataset_conf.sort_size=64 ^
  ++dataset_conf.batch_type="%BATCH_TYPE%" ^
  ++dataset_conf.num_workers=0 ^
  ++train_conf.max_epoch=%MAX_EPOCH% ^
  ++train_conf.accum_grad=%ACCUM_GRAD% ^
  ++train_conf.log_interval=5 ^
  ++train_conf.resume=true ^
  ++train_conf.validate_interval=%VAL_INTERVAL% ^
  ++train_conf.save_checkpoint_interval=%SAVE_INTERVAL% ^
  ++train_conf.keep_nbest_models=3 ^
  ++train_conf.avg_nbest_model=3 ^
  ++train_conf.use_deepspeed=false ^
  ++optim_conf.lr=%LR% ^
  ++audio_encoder_conf.freeze=true ^
  ++audio_adaptor_conf.freeze=false ^
  ++llm_conf.freeze=true ^
  ++output_dir="%OUT%"

echo.
echo Done. Checkpoints in %~dp0outputs_aishell
pause
