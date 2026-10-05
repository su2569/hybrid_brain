#!/usr/bin/env python3
"""HybridBrain 服务守护工具（跨平台 7 种方案）。

用法：
    python3 scripts/serve_guard.py detect           # 检测环境 + 推荐方案
    python3 scripts/serve_guard.py install [方案]   # 安装守护
    python3 scripts/serve_guard.py start [方案]     # 启动
    python3 scripts/serve_guard.py stop             # 停止
    python3 scripts/serve_guard.py status           # 查看状态
    python3 scripts/serve_guard.py logs             # 查看日志
    python3 scripts/serve_guard.py uninstall [方案] # 卸载

方案：
    watchdog    Shell/PowerShell watchdog（容器/DSW 首选）
    systemd     Linux systemd
    supervisor  Linux supervisor
    nssm        Windows NSSM
    task        Windows Task Scheduler
    docker      Docker Compose
    pm2         pm2（跨平台，需 Node）
    auto        自动选择（根据 detect 结果）
"""
import os
import sys
import platform
import subprocess
import shutil
import argparse
import textwrap
from pathlib import Path


# ============ 路径配置 ============
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
LOGS = Path("/mnt/workspace/logs") if os.name != "nt" else ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

SERVICE_NAME = "hybridbrain"
PORT = 8000
PYTHON = sys.executable
UVICORN_CMD = f"{PYTHON} -m uvicorn serve.app:app --host 0.0.0.0 --port {PORT}"

IS_WIN = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")


# ============ 通用工具 ============
def run(cmd, check=False, capture=False, shell=False):
    if isinstance(cmd, str) and not shell:
        cmd = cmd.split()
    try:
        if capture:
            r = subprocess.run(cmd, shell=shell, capture_output=True,
                               text=True, timeout=60)
            return r.returncode, r.stdout.strip(), r.stderr.strip()
        else:
            r = subprocess.run(cmd, shell=shell, check=check, timeout=300)
            return r.returncode, "", ""
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def has_cmd(name):
    return shutil.which(name) is not None


def say(tag, msg):
    icons = {"ok": "✅", "warn": "⚠️ ", "fail": "❌", "info": "ℹ️ "}
    print(f"{icons.get(tag, '  ')} {msg}")


# ============ 环境检测 ============
def detect_env():
    env = {
        "os": platform.system(),
        "python": PYTHON,
        "root": str(ROOT),
        "has_systemd": False,
        "has_supervisor": False,
        "has_pm2": False,
        "has_nssm": False,
        "has_docker": False,
        "in_container": False,
        "recommend": None,
        "reasons": [],
    }

    # Linux
    if IS_LINUX:
        # 容器检测
        if os.path.exists("/.dockerenv"):
            env["in_container"] = True
        try:
            with open("/proc/1/cgroup") as f:
                if "docker" in f.read() or "kubepods" in open("/proc/1/cgroup").read():
                    env["in_container"] = True
        except Exception:
            pass

        # systemd（PID 1 且不是容器）
        if has_cmd("systemctl") and not env["in_container"]:
            rc, _, _ = run(["systemctl", "is-system-running"], capture=True)
            env["has_systemd"] = rc in (0, 1)

        env["has_supervisor"] = has_cmd("supervisord") or os.path.exists("/etc/supervisor")
        env["has_pm2"] = has_cmd("pm2")
        env["has_docker"] = has_cmd("docker")

    # Windows
    elif IS_WIN:
        env["has_nssm"] = has_cmd("nssm") or os.path.exists(r"C:\nssm\win64\nssm.exe")
        env["has_pm2"] = has_cmd("pm2")
        env["has_docker"] = has_cmd("docker")

    # 推荐
    if env["in_container"]:
        env["recommend"] = "watchdog"
        env["reasons"].append("容器环境 → watchdog 最稳")
    elif env["has_systemd"]:
        env["recommend"] = "systemd"
        env["reasons"].append("有 systemd → 原生支持")
    elif env["has_supervisor"]:
        env["recommend"] = "supervisor"
        env["reasons"].append("有 supervisor → 无需 systemd")
    elif env["has_pm2"]:
        env["recommend"] = "pm2"
        env["reasons"].append("有 pm2 → 跨平台")
    elif IS_WIN and env["has_nssm"]:
        env["recommend"] = "nssm"
        env["reasons"].append("Windows 有 NSSM")
    elif IS_WIN:
        env["recommend"] = "task"
        env["reasons"].append("Windows → Task Scheduler")
    else:
        env["recommend"] = "watchdog"
        env["reasons"].append("无守护工具 → watchdog 兜底")

    return env


# ============ 方案实现 ============

# ---- 1. watchdog ----
def watchdog_script():
    if IS_WIN:
        return textwrap.dedent(f'''
            # 自动生成：Windows watchdog
            $port = {PORT}
            $dir  = "{ROOT}"
            $logDir = "{LOGS}"
            $wdLog = "$logDir\\watchdog.log"
            New-Item -ItemType Directory -Force -Path $logDir | Out-Null
            function Log($msg) {{
                "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg" |
                    Out-File -Append -FilePath $wdLog
            }}
            Log "watchdog started"
            while ($true) {{
                $l = Get-NetTCPConnection -LocalPort $port -State Listen `
                     -ErrorAction SilentlyContinue
                if (-not $l) {{
                    Log "服务未运行，启动中..."
                    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
                    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
                    $srvLog = "$logDir\\serve_$ts.log"
                    $p = Start-Process -FilePath "{PYTHON}" `
                        -ArgumentList "-m","uvicorn","serve.app:app","--host","0.0.0.0","--port","$port" `
                        -WorkingDirectory $dir `
                        -RedirectStandardOutput $srvLog `
                        -RedirectStandardError "$srvLog.err" `
                        -PassThru -WindowStyle Hidden
                    Log "启动 PID=$($p.Id)"
                    Start-Sleep 30
                    try {{
                        $r = Invoke-WebRequest -Uri "http://localhost:$port/docs" `
                             -TimeoutSec 10 -UseBasicParsing
                        Log "健康检查 HTTP $($r.StatusCode)"
                    }} catch {{ Log "健康检查失败: $_" }}
                    Get-ChildItem "$logDir\\serve_*.log" |
                        Sort-Object LastWriteTime -Descending |
                        Select-Object -Skip 10 | Remove-Item -Force
                }}
                Start-Sleep 15
            }}
        ''').strip()
    else:
        return textwrap.dedent(f'''#!/bin/bash
            # 自动生成：Linux watchdog
            cd "{ROOT}"
            PORT={PORT}
            LOG_DIR="{LOGS}"
            mkdir -p "$LOG_DIR"
            WD_LOG="$LOG_DIR/serve_watchdog.log"
            log() {{ echo "[$(date '+%F %T')] $*" >> "$WD_LOG"; }}
            log "watchdog started"

            while true; do
                if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
                    sleep 15; continue
                fi
                log "服务未运行，启动中..."
                pkill -9 -f "uvicorn serve.app" 2>/dev/null
                sleep 2
                TS=$(date +%Y%m%d_%H%M%S)
                SRV_LOG="$LOG_DIR/serve_${{TS}}.log"
                nohup {PYTHON} -m uvicorn serve.app:app \\
                    --host 0.0.0.0 --port $PORT \\
                    >> "$SRV_LOG" 2>&1 &
                echo $! > "$PWD/.serve.pid"
                log "启动 PID=$!"
                sleep 30
                CODE=$(curl -s -o /dev/null -w "%{{http_code}}" \\
                       --max-time 10 "http://localhost:$PORT/docs" 2>/dev/null || echo 000)
                log "健康检查 HTTP $CODE"
                ls -t "$LOG_DIR"/serve_*.log 2>/dev/null | tail -n +11 | xargs -r rm -f
                sleep 15
            done
        ''').strip()


def install_watchdog():
    if IS_WIN:
        path = SCRIPTS / "watchdog.ps1"
        path.write_text(watchdog_script(), encoding="utf-8")
        say("ok", f"生成 {path}")
        say("info", f"启动：powershell -File {path}")
    else:
        path = SCRIPTS / "watchdog.sh"
        path.write_text(watchdog_script(), encoding="utf-8")
        path.chmod(0o755)
        say("ok", f"生成 {path}")
        return path


def start_watchdog():
    install_watchdog()
    if IS_WIN:
        subprocess.Popen(["powershell", "-NoProfile", "-File",
                          str(SCRIPTS / "watchdog.ps1")],
                         creationflags=0x00000008)  # DETACHED_PROCESS
    else:
        run(f"pkill -f watchdog.sh 2>/dev/null || true", shell=True)
        run(f"pkill -9 -f 'uvicorn serve.app' 2>/dev/null || true", shell=True)
        subprocess.Popen(["nohup", "bash", str(SCRIPTS / "watchdog.sh")],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         start_new_session=True)
    say("ok", "watchdog 已启动")


# ---- 2. systemd ----
def install_systemd():
    if not IS_LINUX or not has_cmd("systemctl"):
        say("fail", "当前环境无 systemd")
        return
    unit = f"""[Unit]
Description=HybridBrain AI Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={ROOT}
ExecStart={PYTHON} -m uvicorn serve.app:app --host 0.0.0.0 --port {PORT}
Restart=always
RestartSec=10
StartLimitBurst=5
StartLimitIntervalSec=120
StandardOutput=append:{LOGS}/serve_systemd.log
StandardError=append:{LOGS}/serve_systemd.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"""
    path = Path(f"/etc/systemd/system/{SERVICE_NAME}.service")
    path.write_text(unit, encoding="utf-8")
    run(f"systemctl daemon-reload", shell=True)
    run(f"systemctl enable {SERVICE_NAME}", shell=True)
    say("ok", f"已写入 {path}")


def start_systemd():
    run(f"systemctl restart {SERVICE_NAME}", shell=True)
    say("ok", "systemd 服务已启动")


# ---- 3. supervisor ----
def install_supervisor():
    if not has_cmd("supervisord") and not os.path.exists("/etc/supervisor"):
        say("fail", "无 supervisor")
        return
    conf = f"""[program:{SERVICE_NAME}]
command={PYTHON} -m uvicorn serve.app:app --host 0.0.0.0 --port {PORT}
directory={ROOT}
autostart=true
autorestart=true
startsecs=60
startretries=5
user=root
stdout_logfile={LOGS}/serve_supervisor.log
stderr_logfile={LOGS}/serve_supervisor.err
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=5
environment=PYTHONUNBUFFERED="1"
"""
    path = Path(f"/etc/supervisor/conf.d/{SERVICE_NAME}.conf")
    path.write_text(conf, encoding="utf-8")
    run("supervisorctl reread", shell=True)
    run("supervisorctl update", shell=True)
    say("ok", f"已写入 {path}")


def start_supervisor():
    run(f"supervisorctl start {SERVICE_NAME}", shell=True)
    say("ok", "supervisor 已启动")


# ---- 4. nssm ----
def install_nssm():
    if not IS_WIN:
        say("fail", "NSSM 仅 Windows")
        return
    nssm = "nssm"
    if not has_cmd("nssm"):
        if os.path.exists(r"C:\nssm\win64\nssm.exe"):
            nssm = r"C:\nssm\win64\nssm.exe"
        else:
            say("fail", "未找到 nssm.exe，请先下载 https://nssm.cc/download")
            return
    cmds = [
        f'{nssm} install {SERVICE_NAME} "{PYTHON}" "-m uvicorn serve.app:app --host 0.0.0.0 --port {PORT}"',
        f'{nssm} set {SERVICE_NAME} AppDirectory "{ROOT}"',
        f'{nssm} set {SERVICE_NAME} AppEnvironmentExtra PYTHONUNBUFFERED=1',
        f'{nssm} set {SERVICE_NAME} AppExit Default Restart',
        f'{nssm} set {SERVICE_NAME} AppRestartDelay 10000',
        f'{nssm} set {SERVICE_NAME} AppThrottle 120000',
        f'{nssm} set {SERVICE_NAME} AppStdout "{LOGS}\\serve.log"',
        f'{nssm} set {SERVICE_NAME} AppStderr "{LOGS}\\serve.err"',
    ]
    for c in cmds:
        run(c, shell=True)
    say("ok", f"NSSM 服务 {SERVICE_NAME} 已注册")


def start_nssm():
    run(f"net start {SERVICE_NAME}", shell=True)
    say("ok", "NSSM 服务已启动")


# ---- 5. task ----
def install_task():
    if not IS_WIN:
        say("fail", "Task Scheduler 仅 Windows")
        return
    ps = f'''
$Action = New-ScheduledTaskAction `
    -Execute "{PYTHON}" `
    -Argument "-m uvicorn serve.app:app --host 0.0.0.0 --port {PORT}" `
    -WorkingDirectory "{ROOT}"
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Settings = New-ScheduledTaskSettingsSet `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -ExecutionTimeLimit (New-TimeSpan -Days 365) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable
$Principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName "{SERVICE_NAME}" `
    -Action $Action -Trigger $Trigger `
    -Settings $Settings -Principal $Principal `
    -Description "HybridBrain AI Service" -Force
'''
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True)
    if r.returncode == 0:
        say("ok", f"Task Scheduler 已注册 {SERVICE_NAME}")
    else:
        say("fail", f"注册失败: {r.stderr[:200]}")


def start_task():
    run(f'schtasks /run /tn "{SERVICE_NAME}"', shell=True)
    say("ok", "Task 已启动")


# ---- 6. docker ----
def install_docker():
    if not has_cmd("docker"):
        say("fail", "无 docker")
        return
    compose = f"""version: '3.8'

services:
  {SERVICE_NAME}:
    build: .
    image: {SERVICE_NAME}:latest
    container_name: {SERVICE_NAME}
    restart: unless-stopped
    ports:
      - "{PORT}:{PORT}"
    volumes:
      - ./data:/app/data
      - /mnt/workspace/models:/app/models:ro
      - /mnt/workspace/checkpoints:/app/checkpoints:ro
      - /mnt/workspace/logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{PORT}/docs"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
"""
    path = ROOT / "docker-compose.yml"
    path.write_text(compose, encoding="utf-8")

    # 顺便写 Dockerfile
    df = ROOT / "Dockerfile"
    if not df.exists():
        df.write_text(f"""FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \\
    torch transformers sentence-transformers peft \\
    fastapi uvicorn pydantic \\
    openai numpy scikit-learn

EXPOSE {PORT}
CMD ["python3", "-m", "uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "{PORT}"]
""", encoding="utf-8")
    say("ok", f"已生成 {path} + Dockerfile")


def start_docker():
    run("docker compose up -d", shell=True)
    say("ok", "docker 已启动")


# ---- 7. pm2 ----
def install_pm2():
    if not has_cmd("pm2"):
        say("fail", "无 pm2，请先 npm install -g pm2")
        return
    say("ok", "pm2 可用（无需安装文件）")


def start_pm2():
    if not has_cmd("pm2"):
        say("fail", "无 pm2")
        return
    run(f'pm2 delete {SERVICE_NAME} 2>/dev/null || true', shell=True)
    cmd = (f'pm2 start "{UVICORN_CMD}" --name {SERVICE_NAME} '
           f'--cwd "{ROOT}" --interpreter none '
           f'--max-restarts 10 --restart-delay 10000 --time '
           f'--log "{LOGS}/pm2.log"')
    run(cmd, shell=True)
    say("ok", f"pm2 已启动 {SERVICE_NAME}")


# ============ 方案调度 ============
SCHEMES = {
    "watchdog":   (install_watchdog,   start_watchdog),
    "systemd":    (install_systemd,    start_systemd),
    "supervisor": (install_supervisor, start_supervisor),
    "nssm":       (install_nssm,       start_nssm),
    "task":       (install_task,       start_task),
    "docker":     (install_docker,     start_docker),
    "pm2":        (install_pm2,        start_pm2),
}


# ============ 命令 ============
def cmd_detect(args):
    env = detect_env()
    print("=" * 60)
    print("环境检测")
    print("=" * 60)
    for k, v in env.items():
        if k == "reasons":
            print(f"  {k}:")
            for r in v:
                print(f"    - {r}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)
    print(f"\n推荐方案: {env['recommend']}")
    return env


def cmd_install(args):
    scheme = args.scheme or detect_env()["recommend"]
    if scheme == "auto":
        scheme = detect_env()["recommend"]
    if scheme not in SCHEMES:
        say("fail", f"未知方案: {scheme}")
        return
    say("info", f"安装方案: {scheme}")
    SCHEMES[scheme][0]()


def cmd_start(args):
    scheme = args.scheme or detect_env()["recommend"]
    if scheme == "auto":
        scheme = detect_env()["recommend"]
    if scheme not in SCHEMES:
        say("fail", f"未知方案: {scheme}")
        return
    say("info", f"启动方案: {scheme}")
    SCHEMES[scheme][1]()


def cmd_stop(args):
    if IS_WIN:
        run(f'net stop {SERVICE_NAME} 2>nul', shell=True)
    else:
        run(f"pkill -9 -f 'uvicorn serve.app' 2>/dev/null || true", shell=True)
        run(f"pkill -f watchdog.sh 2>/dev/null || true", shell=True)
        run(f"systemctl stop {SERVICE_NAME} 2>/dev/null || true", shell=True)
    say("ok", "已停止")


def cmd_status(args):
    print("=" * 60)
    print("服务状态")
    print("=" * 60)

    # 检查端口
    if IS_WIN:
        rc, out, _ = run(f'netstat -ano | findstr :{PORT}', shell=True, capture=True)
        listening = f":{PORT}" in out
    else:
        rc, out, _ = run(f"ss -tlnp | grep ':{PORT}'", shell=True, capture=True)
        listening = bool(out.strip())

    if listening:
        say("ok", f"端口 {PORT} 正在监听")
    else:
        say("fail", f"端口 {PORT} 无服务")

    # 健康检查
    if IS_WIN:
        cmd = (f"(Invoke-WebRequest -Uri 'http://localhost:{PORT}/docs' "
               f"-UseBasicParsing -TimeoutSec 5).StatusCode")
        rc, out, _ = run(f'powershell -Command "{cmd}"', shell=True, capture=True)
    else:
        rc, out, _ = run(f"curl -s -o /dev/null -w '%{{http_code}}' "
                         f"--max-time 5 http://localhost:{PORT}/docs",
                         shell=True, capture=True)
    say("info", f"健康检查 HTTP: {out.strip() or 'FAIL'}")

    # watchdog 状态
    if not IS_WIN:
        rc, out, _ = run("pgrep -af watchdog.sh", shell=True, capture=True)
        if out.strip():
            say("ok", f"watchdog 运行中: {out.strip()[:80]}")
        else:
            say("warn", "watchdog 未运行")


def cmd_logs(args):
    # 找最新日志
    if not LOGS.exists():
        say("fail", f"日志目录不存在: {LOGS}")
        return
    logs = sorted(LOGS.glob("serve_*.log"), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    if not logs:
        say("warn", "无日志文件")
        return
    latest = logs[0]
    say("info", f"最新日志: {latest}")
    n = args.lines or 50
    if IS_WIN:
        run(f'powershell -Command "Get-Content {latest} -Tail {n}"', shell=True)
    else:
        run(f"tail -n {n} {latest}", shell=True)


def cmd_uninstall(args):
    scheme = args.scheme or detect_env()["recommend"]
    print(f"卸载 {scheme}...")
    if scheme == "systemd":
        run(f"systemctl stop {SERVICE_NAME} 2>/dev/null || true", shell=True)
        run(f"systemctl disable {SERVICE_NAME} 2>/dev/null || true", shell=True)
        p = Path(f"/etc/systemd/system/{SERVICE_NAME}.service")
        if p.exists():
            p.unlink()
            run("systemctl daemon-reload", shell=True)
    elif scheme == "supervisor":
        p = Path(f"/etc/supervisor/conf.d/{SERVICE_NAME}.conf")
        if p.exists():
            p.unlink()
        run("supervisorctl update", shell=True)
    elif scheme == "nssm" and IS_WIN:
        run(f"net stop {SERVICE_NAME}", shell=True)
        run(f"nssm remove {SERVICE_NAME} confirm", shell=True)
    elif scheme == "task" and IS_WIN:
        run(f'schtasks /delete /tn "{SERVICE_NAME}" /f', shell=True)
    elif scheme == "docker":
        run("docker compose down", shell=True)
    elif scheme == "pm2":
        run(f"pm2 delete {SERVICE_NAME}", shell=True)
    else:
        # watchdog
        if IS_WIN:
            run("Get-Process powershell -ErrorAction SilentlyContinue | "
                "Where-Object {$_.CommandLine -like '*watchdog.ps1*'} | "
                "Stop-Process", shell=True)
        else:
            run("pkill -f watchdog.sh 2>/dev/null || true", shell=True)
    say("ok", "卸载完成")


# ============ main ============
def main():
    p = argparse.ArgumentParser(description="HybridBrain 服务守护工具")
    sub = p.add_subparsers(dest="cmd")

    p_detect = sub.add_parser("detect", help="检测环境 + 推荐方案")
    p_detect.set_defaults(func=cmd_detect)

    for name, fn in [
        ("install", cmd_install),
        ("start", cmd_start),
        ("uninstall", cmd_uninstall),
    ]:
        sp = sub.add_parser(name, help=f"{name} 服务")
        sp.add_argument("scheme", nargs="?",
                        choices=list(SCHEMES.keys()) + ["auto"],
                        help="方案名（默认自动）")
        sp.set_defaults(func=fn)

    p_stop = sub.add_parser("stop", help="停止服务")
    p_stop.set_defaults(func=cmd_stop)

    p_status = sub.add_parser("status", help="查看状态")
    p_status.set_defaults(func=cmd_status)

    p_logs = sub.add_parser("logs", help="查看日志")
    p_logs.add_argument("-n", "--lines", type=int, default=50)
    p_logs.set_defaults(func=cmd_logs)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
