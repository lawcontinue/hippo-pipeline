#!/usr/bin/env python3
"""Hippo Pipeline Benchmark v3

使用 subprocess + PYTHONUNBUFFERED=1 解析 sharded_inference.py 输出。

用法:
    python3 benchmark.py --runs 3 --max-tokens 50
    python3 benchmark.py --runs 5 --json results.json --csv results.csv
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class StepTiming:
    step: int
    token_text: str
    step_ms: float
    r0_ms: float
    tok_s: float


@dataclass
class RunResult:
    run_id: int
    avg_tok_s: float
    min_tok_s: float
    max_tok_s: float
    stddev_tok_s: float
    avg_step_ms: float
    prefill_s: float
    output_tokens: int
    steps: list = field(default_factory=list)


def get_mlx_version():
    try:
        r = subprocess.run([sys.executable, '-m', 'pip', 'show', 'mlx'],
                          capture_output=True, text=True, timeout=5)
        for line in r.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':', 1)[1].strip()
    except:
        pass
    return 'unknown'


def run_once(host, port, prompt, max_tokens, temperature, timeout, run_id):
    """Run sharded_inference.py as subprocess, parse output."""
    cmd = [
        sys.executable, str(Path(__file__).parent / 'sharded_inference.py'),
        '--rank', '0', '--host', host, '--port', str(port),
        '--prompt', prompt, '--max-tokens', str(max_tokens),
        '--temperature', str(temperature),
    ]
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    
    steps = []
    prefill_s = 0
    
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            line = line.strip()
            if not line:
                continue
            
            # Parse: Step 5: ' token' | r0=0.0s step=0.125s | 7.49 tok/s
            if line.startswith('Step') and 'tok/s' in line:
                parts = line.split('|')
                step_part = parts[0].strip()
                r0_part = parts[1].strip() if len(parts) > 1 else ''
                tok_part = parts[2].strip() if len(parts) > 2 else ''
                
                step_num = int(re.match(r'Step (\d+)', step_part).group(1))
                token_text = step_part.split(":", 1)[1].strip().strip("'\"") if ':' in step_part else ''
                
                r0_ms = step_ms = 0
                for item in r0_part.split():
                    if item.startswith('r0='):
                        r0_ms = float(item.replace('r0=', '').rstrip('s')) * 1000
                    elif item.startswith('step='):
                        step_ms = float(item.replace('step=', '').rstrip('s')) * 1000
                
                tok_s = float(tok_part.replace('tok/s', '').strip()) if tok_part else 0
                steps.append(StepTiming(step_num, token_text, step_ms, r0_ms, tok_s))
            
            # Parse prefill
            if 'Prefill' in line and '完成' in line:
                m = re.search(r'\((\d+\.?\d*)s\)', line)
                if m:
                    prefill_s = float(m.group(1))
            
            # Stop at generation complete
            if '生成完成' in line:
                break
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=5)
            except: proc.kill()
    
    if not steps:
        raise RuntimeError(f"No steps parsed from subprocess output")
    
    tok_s_list = [s.tok_s for s in steps]
    step_ms_list = [s.step_ms for s in steps]
    
    return RunResult(
        run_id=run_id,
        avg_tok_s=statistics.mean(tok_s_list),
        min_tok_s=min(tok_s_list),
        max_tok_s=max(tok_s_list),
        stddev_tok_s=statistics.stdev(tok_s_list) if len(tok_s_list) > 1 else 0,
        avg_step_ms=statistics.mean(step_ms_list),
        prefill_s=prefill_s,
        output_tokens=len(steps) + 1,
        steps=[asdict(s) for s in steps],
    )


def main():
    parser = argparse.ArgumentParser(description="Hippo Pipeline Benchmark v3")
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=9998)
    parser.add_argument('--prompt', default='The capital of France is')
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--json', help='Export JSON')
    parser.add_argument('--csv', help='Export CSV')
    args = parser.parse_args()
    
    mlx_ver = get_mlx_version()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    transport = 'thunderbolt' if args.host.startswith('169.254.') else 'wifi'
    
    print("=" * 60)
    print("🔬 Hippo Pipeline Benchmark v3")
    print("=" * 60)
    print(f"MLX: {mlx_ver} | Python: {py_ver} | Transport: {transport}")
    print(f"Target: {args.host}:{args.port}")
    print(f"Prompt: {args.prompt!r} | Max tokens: {args.max_tokens}")
    print(f"Runs: {args.runs} | Timeout: {args.timeout}s")
    print("=" * 60)
    
    results = []
    for run_id in range(1, args.runs + 1):
        print(f"\n🏃 Run {run_id}/{args.runs}")
        print("─" * 30)
        try:
            r = run_once(args.host, args.port, args.prompt,
                        args.max_tokens, args.temperature, args.timeout, run_id)
            results.append(r)
            print(f"  ✅ {r.output_tokens} tokens | {r.avg_tok_s:.2f} tok/s "
                  f"(min={r.min_tok_s:.2f} max={r.max_tok_s:.2f} σ={r.stddev_tok_s:.2f})")
            print(f"  Step avg: {r.avg_step_ms:.1f}ms | Prefill: {r.prefill_s:.1f}s")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    if not results:
        print("\n❌ No successful runs")
        return
    
    # Summary
    all_tok = [r.avg_tok_s for r in results]
    print(f"\n{'=' * 60}")
    print(f"📊 SUMMARY ({len(results)}/{args.runs} runs)")
    print(f"{'=' * 60}")
    print(f"  tok/s:  avg={statistics.mean(all_tok):.2f} min={min(all_tok):.2f} "
          f"max={max(all_tok):.2f} σ={statistics.stdev(all_tok) if len(all_tok)>1 else 0:.2f}")
    print(f"  Step:   avg={statistics.mean([r.avg_step_ms for r in results]):.1f}ms")
    print(f"  Prefill: avg={statistics.mean([r.prefill_s for r in results]):.1f}s")
    
    if len(results) > 1:
        print(f"\n  {'Run':>3} | {'tok/s':>8} | {'Step':>8} | {'Prefill':>8} | {'Tokens':>6}")
        print(f"  {'─'*3}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*6}")
        for r in results:
            print(f"  {r.run_id:>3} | {r.avg_tok_s:>8.2f} | {r.avg_step_ms:>7.1f}ms | "
                  f"{r.prefill_s:>7.1f}s | {r.output_tokens:>6}")
    
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({
                'mlx_version': mlx_ver, 'python_version': py_ver,
                'transport': transport, 'host': args.host, 'port': args.port,
                'prompt': args.prompt, 'max_tokens': args.max_tokens,
                'agg_tok_s_avg': statistics.mean(all_tok),
                'agg_tok_s_min': min(all_tok), 'agg_tok_s_max': max(all_tok),
                'results': [asdict(r) for r in results],
            }, f, indent=2, ensure_ascii=False)
        print(f"\n📄 JSON: {args.json}")
    
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['run','tok_s_avg','tok_s_min','tok_s_max','tok_s_std',
                        'step_ms_avg','prefill_s','tokens'])
            for r in results:
                w.writerow([r.run_id, f"{r.avg_tok_s:.2f}", f"{r.min_tok_s:.2f}",
                           f"{r.max_tok_s:.2f}", f"{r.stddev_tok_s:.2f}",
                           f"{r.avg_step_ms:.1f}", f"{r.prefill_s:.1f}", r.output_tokens])
        print(f"📄 CSV: {args.csv}")


if __name__ == '__main__':
    main()
