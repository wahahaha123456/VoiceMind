# -*- coding: utf-8 -*-
"""
VoiceMind 声智助手 —— 一键启动器（无窗口常驻版，由 pythonw.exe 运行）

双击 启动VoiceMind.bat 后：
  1) 本启动器以 pythonw 无窗口方式常驻后台（不弹任何终端）
  2) 后台静默拉起 app.py（:8000）与 ws_server.py（:8001）
  3) 端口就绪后自动打开浏览器
  4) 停止方式：双击 停止VoiceMind.bat（按 logs/pids.json 精确结束，端口兜底）
  5) 服务运行中再次双击 启动VoiceMind.bat：只重新打开页面，不重复启动服务

日志（排查问题看这里）：
  logs/launcher.log  启动器自身
  logs/app.log       Web 服务
  logs/ws.log        WebSocket 流式服务

实现要点：本进程常驻并持有 Windows 作业对象（KILL_ON_JOB_CLOSE），
停止脚本按 PID 结束本进程时，两个服务作为子进程随之退出。
"""
import ctypes
import json
import logging
import os
import socket
import subprocess
import sys
import time
import webbrowser

# ------------------------------------------------------------ 路径与端口
CODE_DIR = os.path.dirname(os.path.abspath(__file__))            # .../code
BASE_DIR = os.path.dirname(CODE_DIR)                             # 项目根目录
PYTHON_EXE = os.path.join(BASE_DIR, "funasr_env", "Scripts", "python.exe")
LOG_DIR = os.path.join(BASE_DIR, "logs")
PID_FILE = os.path.join(LOG_DIR, "pids.json")

APP_PORT = 8000
WS_PORT = 8001
HOME_URL = "http://127.0.0.1:%d/index.html" % APP_PORT

# 模型加载较慢：Web 服务要载 SenseVoice+VAD+标点，流式服务要载 paraformer-zh-streaming
APP_TIMEOUT = 180
WS_TIMEOUT = 240

# 保留子进程的日志文件句柄，防止对象被回收后句柄提前关闭
_LOG_HANDLES = []

# ------------------------------------------------------------ 日志（pythonw 下没有 stdout，必须写文件）
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "launcher.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    # 必须显式指定 UTF-8：不写的话中文 Windows 上按 cp936 写入，
    # 用 UTF-8 工具查看就是乱码（子进程日志也统一按 UTF-8 落盘）
    encoding="utf-8",
)


def log(msg=""):
    logging.info(msg)


def log_err(msg):
    logging.error(msg)


# ------------------------------------------------------------ 基础探测
def port_in_use(port):
    """探测本机端口是否已被监听"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def pid_alive(pid):
    """不依赖 tasklist 输出格式的进程存活探测（供二次启动去重用）"""
    if not pid or os.name != "nt":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32")
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        return False


# ------------------------------------------------------------ 进程收尾保障
def tail_log(log_name, max_lines=12):
    """读取服务日志的最后几行，用于失败时弹窗里给出原因"""
    path = os.path.join(LOG_DIR, log_name)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f.read().splitlines() if ln.strip()]
        return "\n".join(lines[-max_lines:])
    except Exception:
        return "（日志读取失败）"


def show_error_box(title, message):
    """
    启动失败时弹窗提示。

    背景：本启动器以 pythonw 无窗口方式运行，服务崩溃时用户只会看到
    「双击快捷方式毫无反应」，根本不知道要去 logs/ 翻日志。
    失败即弹窗，把日志末尾几行直接摆到桌面上，是最小代价的可见性补丁。
    用 try/except 包住：弹窗本身绝不能反过来影响启停流程。
    """
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10 | 0x1000)
    except Exception:
        pass


def create_kill_on_close_job():
    """
    创建 Windows 作业对象，并设置「作业关闭时杀死全部子进程」。

    意义：本启动器常驻后台，停止脚本按 PID 强杀本进程时，
    Python 的 finally 不一定来得及执行；而作业对象由内核托管，
    句柄随进程销毁而关闭，子服务必定被连带杀死。
    非 Windows 平台或创建失败时返回 None，退回 stop.py 兜底清理。
    """
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32")

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMIT),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            return None
        return (kernel32, job)
    except Exception:
        return None


def assign_to_job(job, process):
    """把子进程挂进作业对象（失败仅意味着退出时靠 stop.py 兜底）"""
    if not job:
        return
    try:
        kernel32, handle = job
        kernel32.AssignProcessToJobObject(handle, int(process._handle))
    except Exception:
        pass


# ------------------------------------------------------------ 服务管理
def start_service(script_name, port, service_name, log_name):
    """
    以无窗口方式后台启动一个服务脚本，输出重定向到 logs/<log_name>。
    端口已被占用时跳过启动（避免重复加载几十秒的模型）。
    """
    if port_in_use(port):
        log("%s 已在运行（端口 %d），跳过启动" % (service_name, port))
        return None
    script_path = os.path.join(CODE_DIR, script_name)
    if not os.path.exists(script_path):
        log_err("找不到脚本：%s" % script_path)
        return None
    log_file = open(os.path.join(LOG_DIR, log_name), "wb")
    _LOG_HANDLES.append(log_file)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        [PYTHON_EXE, "-u", script_path],
        cwd=CODE_DIR,                    # 工作目录固定为 code/，脚本内部再用 __file__ 找资源
        creationflags=flags,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    log("%s 启动中（端口 %d，PID %d）" % (service_name, port, process.pid))
    return process


def wait_for_service(port, timeout, process=None):
    """轮询等待端口就绪；若子进程中途退出则立即返回 False（不白等）"""
    start = time.time()
    while time.time() - start < timeout:
        if port_in_use(port):
            return True
        if process is not None and process.poll() is not None:
            return False          # 进程已崩溃，日志里有原因
        time.sleep(0.5)
    return False


def stop_children(processes):
    """启动器正常退出时收尾（被 taskkill 强杀时由作业对象兜底）"""
    for p in processes:
        if p is None or p.poll() is not None:
            continue
        try:
            p.terminate()
        except Exception:
            pass
    deadline = time.time() + 5
    for p in processes:
        if p is None or p.poll() is not None:
            continue
        try:
            p.wait(timeout=max(0.5, deadline - time.time()))
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


# ------------------------------------------------------------ PID 记录
def read_pid_file():
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_pid_file(app_proc, ws_proc):
    try:
        data = {
            "launcher": os.getpid(),
            "app": app_proc.pid if app_proc else None,
            "ws": ws_proc.pid if ws_proc else None,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(PID_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def remove_pid_file():
    """只允许 pid 文件的主人删除，避免后来者误删正在运行实例的记录"""
    data = read_pid_file()
    if data and data.get("launcher") != os.getpid():
        return
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except OSError:
        pass


def open_home_page():
    """按环境变量决定是否真的打开浏览器（FUNASR_NO_BROWSER=1 供自动化测试用）"""
    if os.environ.get("FUNASR_NO_BROWSER") == "1":
        log("FUNASR_NO_BROWSER 已设置，跳过打开浏览器")
        return
    time.sleep(1)
    webbrowser.open(HOME_URL)
    log("已打开浏览器 %s" % HOME_URL)


# ------------------------------------------------------------ 主流程
def main():
    log("=" * 50)
    log("VoiceMind 启动器启动（无窗口模式）")

    if not os.path.exists(PYTHON_EXE):
        log_err("找不到虚拟环境解释器：%s" % PYTHON_EXE)
        return 1

    # 已有启动器常驻（pids.json 且进程存活）→ 不重复拉服务，就绪后直接开页面
    previous = read_pid_file()
    if previous and pid_alive(previous.get("launcher")):
        log("检测到已有启动器在运行（PID %s），等服务就绪后直接打开页面" % previous.get("launcher"))
        if wait_for_service(APP_PORT, APP_TIMEOUT):
            open_home_page()
        else:
            log("等待超时，服务仍未就绪（详见 logs/app.log）")
        return 0

    # 先声明主权（只记 launcher PID），把“双击两次”的竞态窗口压到最小
    write_pid_file(None, None)

    job = create_kill_on_close_job()

    log("正在启动 Web 服务（端口几秒内就绪，模型在首次使用时才加载）...")
    app_proc = start_service("app.py", APP_PORT, "Web服务", "app.log")
    assign_to_job(job, app_proc)
    log("正在启动 WebSocket 流式服务...")
    ws_proc = start_service("ws_server.py", WS_PORT, "WebSocket服务", "ws.log")
    assign_to_job(job, ws_proc)

    # 两个端口都已被占用：服务归别的实例所有，本实例只开页面就退出，不留常驻进程
    if app_proc is None and ws_proc is None:
        log("两个服务均已在运行，直接打开页面后退出")
        open_home_page()
        return 0

    write_pid_file(app_proc, ws_proc)

    if wait_for_service(APP_PORT, APP_TIMEOUT, app_proc):
        log("Web 服务已就绪  %s" % HOME_URL)
    elif app_proc is not None and app_proc.poll() is not None:
        log_err("Web 服务启动失败，详见 logs/app.log")
        show_error_box(
            "VoiceMind 启动失败",
            "Web 服务（端口 %d）启动即崩溃，页面无法打开。\n\n"
            "错误日志末尾：\n%s\n\n"
            "完整日志：%s" % (APP_PORT, tail_log("app.log"), os.path.join(LOG_DIR, "app.log")),
        )
        return 1
    else:
        log("Web 服务等待超时，可稍后刷新页面（详见 logs/app.log）")

    open_home_page()

    if not port_in_use(WS_PORT):
        log("WebSocket 服务仍在加载模型，实时录音功能需再等一会儿...")
    if wait_for_service(WS_PORT, WS_TIMEOUT, ws_proc):
        log("WebSocket 服务已就绪  ws://127.0.0.1:%d" % WS_PORT)
    elif ws_proc is not None and ws_proc.poll() is not None:
        log_err("WebSocket 服务启动失败，实时录音不可用（详见 logs/ws.log）")
        show_error_box(
            "VoiceMind 部分功能不可用",
            "WebSocket 服务（端口 %d）启动即崩溃，网页能打开但「实时录音」不可用。\n\n"
            "错误日志末尾：\n%s\n\n"
            "完整日志：%s" % (WS_PORT, tail_log("ws.log"), os.path.join(LOG_DIR, "ws.log")),
        )
    else:
        log("WebSocket 服务等待超时（详见 logs/ws.log）")

    log("系统已启动。启动器常驻后台；停止请运行 停止VoiceMind.bat")

    # 常驻：本进程活着，作业对象就把两个服务拴住；被强杀时服务随之退出
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_children([app_proc, ws_proc])
        remove_pid_file()
    return 0


if __name__ == "__main__":
    sys.exit(main())
