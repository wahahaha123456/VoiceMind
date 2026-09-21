@echo off
REM ============================================================
REM  Fun-ASR-Nano finetune v2 —— 针对「语速快 + 环境嘈杂」
REM
REM  数据（约 5.4 万条 / 60 小时）：
REM    AISHELL-1 train  11986   干净朗读（保底）
REM    aug_noise         7000   加噪 SNR 0~15dB
REM    aug_reverb        7000   真实房间脉冲响应卷积（远场混响）
REM    aug_fast          7000   语速 x1.15~1.40
REM    aug_hard          7000   混响+噪声+快语速（最贴近目标场景）
REM    WenetSpeech dev  13675   真实讲座/演讲（口语化、自然语速、环境声）
REM
REM  验证：AISHELL dev 150（干净）+ WenetSpeech dev 150（口语）
REM  测试：AISHELL eval 1200（干净）+ WenetSpeech meeting 8370（真实会议）
REM
REM  策略：冻结 encoder + LLM，只训 audio_adaptor（官方建议 <1000h 用此法）
REM  从基座重新开始 —— 本次数据分布与上一版差异过大，增量训练无意义
REM
REM  NOTE: 不要用 torchrun（本机 CUDA torch 无 libuv），train_ds 自读 RANK 环境变量
REM ============================================================
setlocal
cd /d %~dp0

set PYTHON=..\..\funasr_env\Scripts\python.exe
set CUDA_VISIBLE_DEVICES=0

REM ---- tunables ----
REM  上轮：11986 条 x 3 epoch = 35958 样本通过，实测 0.256 s/step (batch 8)
REM  本轮：53661 条 x 2 epoch = 107322 样本通过（上轮的 3 倍），故 2 epoch 足够
REM  实测该配置下步时与 batch 近似线性，说明瓶颈在 GPU 而非数据加载，
REM  因此加大 batch 不会缩短总时长，保持 batch=8 最稳（峰值显存 1.06GB）
set BATCH_SIZE=8
set BATCH_TYPE=example
set ACCUM_GRAD=1
set MAX_EPOCH=2
set LR=0.0001
set SAVE_INTERVAL=1500
set VAL_INTERVAL=1500

set TRAIN_JSONL=C:/Users/iiiis/Desktop/FunASR/dataset/list/merged_train.jsonl
set VAL_JSONL=C:/Users/iiiis/Desktop/FunASR/dataset/list/merged_valid.jsonl
set OUT=./outputs_v2

set RANK=0
set LOCAL_RANK=0
set WORLD_SIZE=1
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=26669
set PYTHONUNBUFFERED=1

if not exist outputs_v2 mkdir outputs_v2

echo [1/2] Checking torch CUDA...
%PYTHON% -c "import torch; assert torch.cuda.is_available(), 'torch.cuda not available'; print('CUDA OK:', torch.cuda.get_device_name(0))" || (pause & exit /b 1)

echo [2/2] Starting finetune v2 (audio_adaptor, batch=%BATCH_SIZE% %BATCH_TYPE%, epoch=%MAX_EPOCH%)...
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
  ++train_conf.log_interval=20 ^
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
echo Done. Checkpoints in %~dp0outputs_v2
pause
