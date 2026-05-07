#!/usr/bin/env python3
"""
LoReS-Bench  —  Master Runner
Usage (single merged file):
  python run_lores_bench.py --input merged.jsonl --output_dir results/

Usage (one JSONL per model in a directory):
  python run_lores_bench.py \
      --input_dir /path/to/all-merged/ \
      --output_dir results/
"""
import argparse
from pathlib import Path
from lores_bench_analysis import run_all
from lores_bench_plots     import plot_all

def main():
    p = argparse.ArgumentParser(description="LoReS-Bench full pipeline")
    p.add_argument("--input",      default=None, help="Single merged JSONL file")
    p.add_argument("--input_dir",  default=None, help="Directory tree of JSONL files")
    p.add_argument("--output_dir", default="results")
    a = p.parse_args()

    if not a.input and not a.input_dir:
        p.error("Provide --input or --input_dir")

    out = Path(a.output_dir)
    all_results, df_flat = run_all(
        input_path=a.input,
        input_dir=a.input_dir,
        output_dir=str(out),
    )
    plot_all(all_results, df_flat, output_dir=str(out / "figures"))

    print(f"\n{'='*60}")
    print(f"Pipeline complete.")
    print(f"  CSVs  → {out.resolve()}/")
    print(f"  Plots → {(out/'figures').resolve()}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
