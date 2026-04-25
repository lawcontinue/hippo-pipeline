#!/usr/bin/env python3
"""
sharded_inference.py — CLI entry point for distributed Gemma3 inference.

Usage:
  Rank 1: python3 sharded_inference.py --rank 1 --host 0.0.0.0
  Rank 0: python3 sharded_inference.py --rank 0 --host <R1_IP> --prompt "Hello"
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import build_parser
from rank0 import rank0_generate
from rank1 import rank1_serve


def main():
    args = build_parser().parse_args()
    if args.rank == 1:
        asyncio.run(rank1_serve(args.host, args.port, args.rank0_host, batch_size=args.batch_size))
    else:
        asyncio.run(rank0_generate(
            args.host, args.port, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            repetition_penalty=args.repetition_penalty,
        ))


if __name__ == "__main__":
    main()
