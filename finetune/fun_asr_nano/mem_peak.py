# -*- coding: utf-8 -*-
"""训练显存峰值测量：轮询 torch.cuda.mem_get_info 记录最低可用显存。"""
import subprocess, sys, time, torch, os

log = open("mem_poll.log", "w")
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
log.write(f"GPU total: {total:.2f} GB\n")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
cmd = sys.argv[1:]
p = subprocess.Popen(cmd, env=env)
min_free = total
while p.poll() is None:
    free, _ = torch.cuda.mem_get_info()
    min_free = min(min_free, free / 1024**3)
    time.sleep(1.5)
log.write(f"训练进程退出码: {p.returncode}\n")
log.write(f"峰值占用: {total - min_free:.2f} GB / 共 {total:.2f} GB\n")
log.close()
print(f"PEAK_USED_GB={total - min_free:.2f}")
