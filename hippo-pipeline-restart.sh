#!/bin/bash
# Hippo Pipeline 一键重启 R0+R1
# 用法: ./hippo-pipeline-restart.sh [--r1-only | --r0-only] [--prompt "xxx"] [--max-tokens 200] [--rp 1.2] [--temp 0.0] [--clean-ports]
#
# 默认：重启 R1 并启动 R0 推理
# --r1-only: 只重启 R1
# --r0-only: 只启动 R0（假设 R1 已在线）
# --clean-ports: 清理本地残留端口后再启动

set -e

# ── 配置 ──
R1_HOST="${R1_HOST:-192.168.1.11}"
R1_USER="finance"
R1_PORT=9998
R1_DIR="/Users/finance/hippo/pipeline"
R0_DIR="$(cd "$(dirname "$0")" && pwd)"
THUNDERBOLT_HOST="${R1_THUNDERBOLT:-169.254.2.225}"

# ── 默认参数 ──
PROMPT="Hello, how are you?"
MAX_TOKENS=128
TEMP=0.0
RP=1.0
R1_ONLY=false
R0_ONLY=false
CLEAN_PORTS=false

while [[ $# -gt 0 ]]; do
	case $1 in
	--r1-only)
		R1_ONLY=true
		shift
		;;
	--r0-only)
		R0_ONLY=true
		shift
		;;
	--clean-ports)
		CLEAN_PORTS=true
		shift
		;;
	--prompt)
		PROMPT="$2"
		shift 2
		;;
	--max-tokens)
		MAX_TOKENS="$2"
		shift 2
		;;
	--rp)
		RP="$2"
		shift 2
		;;
	--temp)
		TEMP="$2"
		shift 2
		;;
	--port)
		R1_PORT="$2"
		shift 2
		;;
	*)
		echo "Unknown: $1"
		exit 1
		;;
	esac
done

# ── 清理本地残留端口 ──
clean_local_ports() {
	echo "🧹 清理本地残留端口..."
	# Kill any Python processes holding port R1_PORT+1 (R0 receiver)
	R0_RECV_PORT=$((R1_PORT + 1))
	for port in "$R0_RECV_PORT" "$R1_PORT"; do
		PIDS=$(/usr/sbin/lsof -ti :"$port" 2>/dev/null || true)
		if [ -n "$PIDS" ]; then
			echo "   端口 $port 被占用 (PIDs: $PIDS)，清理中..."
			echo "$PIDS" | xargs kill -9 2>/dev/null || true
			sleep 1
		fi
	done
	echo "✅ 本地端口已清理"
}

# ── 重启 R1 ──
restart_r1() {
	echo "🔄 重启 R1 (${R1_USER}@${R1_HOST}:${R1_PORT})..."
	ssh -o ConnectTimeout=5 ${R1_USER}@${R1_HOST} \
		"pkill -f sharded_inference 2>/dev/null; pkill -f sd_verify 2>/dev/null; sleep 2" 2>/dev/null || true

	ssh ${R1_USER}@${R1_HOST} \
		"source ~/.zshrc && cd ${R1_DIR} && nohup python3 sharded_inference.py --rank 1 --port ${R1_PORT} --rank0-host ${RANK0_HOST:-192.168.1.10} > /tmp/rank1.log 2>&1 &"

	echo "⏳ 等待 R1 加载（10s）..."
	sleep 10

	# 验证 R1 就绪
	READY=$(ssh ${R1_USER}@${R1_HOST} "lsof -i :${R1_PORT} 2>/dev/null | grep LISTEN" 2>/dev/null)
	if [ -n "$READY" ]; then
		echo "✅ R1 就绪 (${R1_HOST}:${R1_PORT})"
	else
		echo "❌ R1 未就绪，检查 /tmp/rank1.log"
		exit 1
	fi
}

# ── 启动 R0 ──
start_r0() {
	echo "🚀 启动 R0 (prompt=${PROMPT:0:30}..., max_tokens=${MAX_TOKENS}, temp=${TEMP}, rp=${RP})..."
	cd "${R0_DIR}"
	python3 sharded_inference.py \
		--rank 0 \
		--host ${R1_HOST} \
		--port ${R1_PORT} \
		--prompt "${PROMPT}" \
		--max-tokens ${MAX_TOKENS} \
		--temperature ${TEMP} \
		--repetition-penalty ${RP}
}

# ── 主流程 ──
if [ "$CLEAN_PORTS" = true ]; then
	clean_local_ports
fi

if [ "$R0_ONLY" = true ]; then
	echo "⚡ R0-only 模式（跳过 R1 重启）"
	start_r0
elif [ "$R1_ONLY" = true ]; then
	restart_r1
else
	restart_r1
	echo ""
	start_r0
fi
