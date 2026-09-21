# -*- coding: utf-8 -*-
"""
VoiceMind 声智助手 —— 停止脚本（供「停止VoiceMind.bat」调用）

结束 8000（Web 服务）/ 8001（WebSocket 服务）两个端口上的服务进程。
两步走：
  1) 按 logs/pids.json 里记录的启动器 PID 收尾 —— 启动器退出时其子进程随之退出；
  2) 按端口兜底清理 —— 覆盖「直接用命令行启动服务、没走启动器」的情况。
"""
import json
import os
import subprocess
import sys
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
PID_FILE = os.path.join(BASE_DIR, "logs", "pids.json")
PORTS = (8000, 8001)

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def log(msg=""):
    print(msg, flush=True)


def run_quiet(cmd, timeout=15):
    """执行外部命令并安静地取回输出（编码异常时忽略，避免中文系统 GBK 报错）"""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="ignore",
                              timeout=timeout, creationflags=_NO_WINDOW)
    except Exception:
        return None


def taskkill(pid):
    """强行结束进程及其子进程，返回是否执行成功"""
    result = run_quiet(["taskkill", "/F", "/T", "/PID", str(pid)])
    return bool(result and result.returncode == 0)


def listening_pids(port):
    """解析 netstat，返回正在监听指定端口的进程号集合"""
    pids = set()
    result = run_quiet(["netstat", "-ano", "-p", "TCP"])
    if not result or not result.stdout:
        return pids
    for line in result.stdout.splitlines():
        parts = line.split()
        # 形如：TCP  0.0.0.0:8000  0.0.0.0:0  LISTENING  10656
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING":
            if parts[1].rsplit(":", 1)[-1] == str(port):
                pids.add(parts[4])
    return pids


def read_launcher_pid():
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("launcher")
    except Exception:
        return None


def main():
    log("正在停止 VoiceMind 服务...")

    # 1) 先结束启动器：作业对象会让它的子进程一起退出
    launcher_pid = read_launcher_pid()
    if launcher_pid and taskkill(launcher_pid):
        log("  已结束启动器进程 PID %s（含其子服务）" % launcher_pid)
    time.sleep(1)

    # 2) 端口兜底
    for port in PORTS:
        for pid in sorted(listening_pids(port)):
            if taskkill(pid):
                log("  已结束占用端口 %d 的进程 PID %s" % (port, pid))
        time.sleep(0.3)

    # 3) 确认结果
    time.sleep(1)
    remaining = {p: sorted(listening_pids(p)) for p in PORTS}
    remaining = {p: v for p, v in remaining.items() if v}

    if remaining:
        log()
        log("⚠ 以下端口仍被占用：")
        for port, pids in remaining.items():
            log("  端口 %d → PID %s" % (port, ", ".join(pids)))
        log("  可在任务管理器中手动结束这些进程。")
        return 1

    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except OSError:
            pass

    log()
    log("✓ 服务已全部停止，端口 8000 / 8001 均已释放。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
