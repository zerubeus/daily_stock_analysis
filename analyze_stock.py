#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI wrapper for single-stock analysis with JSON output.

Usage:
    python analyze_stock.py AAPL --json
    python analyze_stock.py 600519 --json
"""

import argparse
import json
import logging
import os
import sys

# Ensure project root is on sys.path so `src.*` imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import setup_env


def main():
    parser = argparse.ArgumentParser(description="Analyze a single stock")
    parser.add_argument("ticker", help="Stock ticker / code (e.g. AAPL, 600519)")
    parser.add_argument("--json", dest="json_output", action="store_true",
                        help="Print result as JSON to stdout")
    args = parser.parse_args()

    # Load .env before any other imports that depend on config
    setup_env()

    # When --json is used, suppress all logging so only JSON hits stdout
    if args.json_output:
        logging.disable(logging.CRITICAL)

    from analyzer_service import analyze_stock

    result = analyze_stock(args.ticker)

    if result is None:
        if args.json_output:
            json.dump({"success": False, "error": "Analysis returned no result"}, sys.stdout)
        else:
            print("Analysis returned no result.", file=sys.stderr)
        sys.exit(1)

    if args.json_output:
        json.dump(result.to_dict(), sys.stdout, ensure_ascii=False, indent=2)
    else:
        print(f"[{result.code}] {result.name}")
        print(f"  Score: {result.sentiment_score}  Trend: {result.trend_prediction}")
        print(f"  Advice: {result.operation_advice}")
        if result.analysis_summary:
            print(f"  Summary: {result.analysis_summary}")


if __name__ == "__main__":
    main()
