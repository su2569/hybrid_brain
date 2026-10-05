#!/bin/bash
            # 自动生成：Linux watchdog
            cd "/mnt/workspace/hybrid_brain"
            PORT=8000
            LOG_DIR="/mnt/workspace/logs"
            mkdir -p "$LOG_DIR"
            WD_LOG="$LOG_DIR/serve_watchdog.log"
            log() { echo "[$(date '+%F %T')] $*" >> "$WD_LOG"; }
            log "watchdog started"

            while true; do
                if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
                    sleep 15; continue
                fi
                log "服务未运行，启动中..."
                pkill -9 -f "uvicorn serve.app" 2>/dev/null
                sleep 2
                TS=$(date +%Y%m%d_%H%M%S)
                SRV_LOG="$LOG_DIR/serve_${TS}.log"
                nohup /usr/local/bin/python3 -m uvicorn serve.app:app \
                    --host 0.0.0.0 --port $PORT \
                    >> "$SRV_LOG" 2>&1 &
                echo $! > "$PWD/.serve.pid"
                log "启动 PID=$!"
                sleep 30
                CODE=$(curl -s -o /dev/null -w "%{http_code}" \
                       --max-time 10 "http://localhost:$PORT/docs" 2>/dev/null || echo 000)
                log "健康检查 HTTP $CODE"
                ls -t "$LOG_DIR"/serve_*.log 2>/dev/null | tail -n +11 | xargs -r rm -f
                sleep 15
            done