@echo off
REM ============================================================
REM  Fun-ASR-Nano audio_adaptor finetune (Windows, single GPU)
REM  Strategy: freeze audio_encoder + llm, train audio_adaptor
REM  (official recommendation for < 1000 hours of data)
REM  Data : finetune\data\train.jsonl / val.jsonl
REM  Out  : finetune\fun_asr_nano\outputs
REM  Base model already pre-downloaded to ModelScope cache.
REM
REM  Memory settings for RTX 4060 Laptop 8GB:
REM    batch_size = 1 sample + accum_grad = 2  (effective batch = 2)
REM    NOTE: funasr only steps the optimizer when (batch_idx+1) %% accum_grad == 0,
REM          there is NO end-of-epoch flush. With only 7 mini training samples,
REM          accum_grad must stay <= samples_per_epoch or nothing is ever updated.
REM          Once real data is in place (hundreds of samples), raise accum_grad to 8.
REM  If it still OOMs -> use lora_local.bat
REM ============================================================
cd /d %~dp0

set PYTHON=..\..\funasr_env\Scripts\python.exe
set CUDA_VISIBLE_DEVICES=0

if not exist outputs mkdir outputs

echo [1/2] Checking torch CUDA...
%PYTHON% -c "import torch; assert torch.cuda.is_available(), 'torch.cuda not available - is the CUDA build installed?'; print('CUDA OK:', torch.cuda.get_device_name(0))" || (pause & exit /b 1)

echo [2/2] Starting finetune (audio_adaptor)...
REM NOTE: do NOT use torchrun here. On Windows the CUDA torch build has no libuv,
REM and torchrun fails with "use_libuv was requested but PyTorch was built without
REM libuv support" (even with USE_LIBUV=0). Running the trainer module directly works:
REM train_ds reads RANK/LOCAL_RANK/WORLD_SIZE from the environment (defaults = single GPU).
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
  ++audio_encoder_conf.freeze=true ^
  ++audio_adaptor_conf.freeze=false ^
  ++llm_conf.freeze=true ^
  ++output_dir="./outputs"

echo.
echo Done. Checkpoints in %~dp0outputs
pause
