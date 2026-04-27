#!/bin/bash
# Hippo Pipeline Benchmark Shell Wrapper
#
# 用法:
#   ./benchmark.sh 3 50              # 3 runs, 50 tokens
#   ./benchmark.sh 5 100 thunderbolt # 5 runs, 100 tokens, thunderbolt
#
# 需要 R1 已在远端运行

set -e

RUNS=${1:-3}
MAX_TOKENS=${2:-50}
HOST=${3:-${R1_HOST:-192.168.1.11}} # override via env R1_HOST, or pass as 3rd arg
PORT=9998
PROMPT="The capital of France is"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="/tmp/hippo_bench_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "🔬 Hippo Pipeline Benchmark (shell wrapper)"
echo "============================================================"
echo "Host: $HOST:$PORT"
echo "Prompt: '$PROMPT'"
echo "Max tokens: $MAX_TOKENS | Runs: $RUNS"
echo "Results: $RESULTS_DIR"
echo "============================================================"

SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "# Benchmark Results" >"$SUMMARY_FILE"
echo "# Date: $(date)" >>"$SUMMARY_FILE"
echo "# Host: $HOST | Max tokens: $MAX_TOKENS" >>"$SUMMARY_FILE"
echo "" >>"$SUMMARY_FILE"

ALL_TOK_S=()

for i in $(seq 1 $RUNS); do
	echo ""
	echo "────────────────────────────────────────"
	echo "🏃 Run $i/$RUNS"
	echo "────────────────────────────────────────"

	RUN_LOG="$RESULTS_DIR/run_${i}.log"

	python3 "$SCRIPT_DIR/sharded_inference.py" \
		--rank 0 \
		--host "$HOST" \
		--port "$PORT" \
		--prompt "$PROMPT" \
		--max-tokens "$MAX_TOKENS" \
		2>&1 | tee "$RUN_LOG"

	# Parse tok/s from last step
	LAST_TOK=$(grep "tok/s" "$RUN_LOG" | tail -1 | grep -oE '[0-9]+\.[0-9]+ tok/s' | head -1 | awk '{print $1}')
	AVG_TOK=$(grep "tok/s" "$RUN_LOG" | awk -F'|' '{print $NF}' | grep -oE '[0-9]+\.[0-9]+' | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "0"}')

	echo "  📊 Run $i: last=${LAST_TOK:-N/A} tok/s, avg=${AVG_TOK:-N/A} tok/s"
	echo "Run $i: last=${LAST_TOK:-N/A} avg=${AVG_TOK:-N/A}" >>"$SUMMARY_FILE"

	if [ -n "$AVG_TOK" ] && [ "$AVG_TOK" != "0" ]; then
		ALL_TOK_S+=("$AVG_TOK")
	fi

	# Restart R1 between runs (R1 crashes on disconnect)
	# NOTE: With graceful disconnect fix, R1 should survive
	if [ $i -lt $RUNS ]; then
		echo "  ⏳ Waiting 3s before next run..."
		sleep 3
	fi
done

echo ""
echo "============================================================"
echo "📊 SUMMARY"
echo "============================================================"
echo "  Runs: $RUNS"
echo "  Successful: ${#ALL_TOK_S[@]}"
if [ ${#ALL_TOK_S[@]} -gt 0 ]; then
	echo "  Results: ${ALL_TOK_S[*]}"
	# Calculate avg
	python3 -c "
values = [${ALL_TOK_S[*]}]
import statistics
print(f'  tok/s avg: {statistics.mean(values):.2f}')
print(f'  tok/s min: {min(values):.2f}')
print(f'  tok/s max: {max(values):.2f}')
if len(values) > 1:
    print(f'  tok/s σ:   {statistics.stdev(values):.2f}')
"
fi
echo "============================================================"
echo ""
echo "📄 Full results: $RESULTS_DIR"
