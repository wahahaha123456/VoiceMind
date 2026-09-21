@echo off
REM ============================================================
REM  Fun-ASR-Nano LoRA finetune (Windows, single GPU) - FALLBACK
REM  Use this if finetune_local.bat runs out of memory (8 GB GPU).
REM  Pure-LoRA: only lora_A/lora_B inside the Qwen3-0.6B LLM train.
REM  Out: finetune\fun_asr_nano\outputs_lora
REM ============================================================
cd /d %~dp0

set PYTHON=..\..\funasr_env\Scripts\python.exe
set CUDA_VISIBLE_DEVICES=0

if not exist outputs_lora mkdir outputs_lora

echo [1/2] Checking torch CUDA...
%PYTHON% -c "import torch; assert torch.cuda.is_available(), 'torch.cuda not available - is the CUDA build installed?'; print('CUDA OK:', torch.cuda.get_device_name(0))" || (pause & exit /b 1)

echo [2/2] Starting LoRA finetune...
REM NOTE: torchrun does not work on Windows (CUDA torch built without libuv).
REM Run the trainer module directly instead; it reads RANK/LOCAL_RANK/WORLD_SIZE.
set RANK=0
set LOCAL_RANK=0
set WORLD_SIZE=1
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=26669

%PYTHON% -m funasr.bin.train_ds ^
  ++model="FunAudioLLM/Fun-ASR-Nano-2512" ^
  ++trust_remote_code=true ^
  ++train_data_set_list="C:/Users/iiiis/Desktop/FunASR/finetune/data/train.jsonl" ^
  ++valid_data_set_list="C:/Users/iiiis/Desktop/FunASR/finetune/data/val.jsonl" ^
  ++dataset_conf.data_split_num=1 ^
  ++dataset_conf.batch_sampler="BatchSampler" ^
  ++dataset_conf.batch_size=1 ^
  ++dataset_conf.sort_size=32 ^
  ++dataset_conf.batch_type="example" ^
  ++dataset_conf.num_workers=0 ^
  ++train_conf.max_epoch=10 ^
  ++train_conf.accum_grad=2 ^
  ++train_conf.log_interval=1 ^
  ++train_conf.resume=true ^
  ++train_conf.validate_interval=10 ^
  ++train_conf.save_checkpoint_interval=10 ^
  ++train_conf.keep_nbest_models=3 ^
  ++train_conf.avg_nbest_model=3 ^
  ++train_conf.use_deepspeed=false ^
  ++optim_conf.lr=0.0001 ^
  ++llm_conf.use_lora=true ^
  ++lora_only=true ^
  ++llm_conf.freeze=true ^
  ++audio_encoder_conf.freeze=true ^
  ++audio_adaptor_conf.freeze=true ^
  ++output_dir="./outputs_lora"

echo.
echo Done. Checkpoints in %~dp0outputs_lora
pause
