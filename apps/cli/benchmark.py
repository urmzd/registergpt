"""
Benchmark CLI — run all (or selected) model versions under identical conditions
and produce a comparison table.

Usage:
    benchmark [--versions v1,v3,sparse] [--minutes 10] [--batch 491520] [--warmup 5]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ALL_VERSIONS = [
    "v1_attention", "v2_conv", "v3_assoc", "v4_golf",
    "v5_gauss", "v6_wave", "v7_lgp", "v8_graph",
    "v9_meta", "v10_policy", "v11_brainwave", "v11_tpg",
    "v12_sparse", "v13_embed", "v14_adaptive",
    "v15_predictive", "v16_columnar",
]


def detect_gpus() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return len([l for l in out.strip().splitlines() if l.strip()])
    except Exception:
        return 1


def run_one(version: str, minutes: float, batch: int, warmup: int, nproc: int) -> dict | None:
    env = {
        **os.environ,
        "MAX_WALLCLOCK_SECONDS": str(int(minutes * 60)),
        "ITERATIONS": "999999",
        "WARMUP_STEPS": str(warmup),
        "VAL_LOSS_EVERY": "999999",
        "TRAIN_LOG_EVERY": "1",
        "TRAIN_BATCH_TOKENS": str(batch),
        "MODEL_VERSION": version,
        "CHECKPOINT_EVERY": "0",
    }
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={nproc}", "train.py",
    ]
    print(f"\n{'='*60}")
    print(f"  MODEL_VERSION={version}  ({minutes}min, batch={batch})")
    print(f"{'='*60}")
    sys.stdout.flush()
    proc = subprocess.Popen(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)
    proc.wait()

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode})")
        return None

    # Find the manifest from output
    for line in output_lines:
        if "manifest:" in line:
            manifest_path = line.split("manifest:", 1)[1].strip()
            if Path(manifest_path).exists():
                with open(manifest_path) as f:
                    return json.load(f)
    return None


def print_table(results: list[dict]):
    if not results:
        print("\nNo results to display.")
        return

    headers = ["version", "params", "steps", "val_loss", "val_bpb", "train_loss", "tok/s", "time_s"]
    rows = []
    for r in results:
        train_ms = r.get("train_time_ms", 0)
        steps = r.get("steps_trained", 0)
        avg_ms = train_ms / max(steps, 1)
        batch = r.get("batch_tokens", 491520)
        toks = batch / (avg_ms / 1000) if avg_ms > 0 else 0
        rows.append([
            r.get("model_version", "?"),
            f"{r.get('params', 0)/1e3:.0f}K",
            str(steps),
            f"{r.get('val_loss', 0):.4f}",
            f"{r.get('val_bpb', 0):.4f}",
            f"{r.get('final_train_loss', 0) or 0:.4f}",
            f"{toks:.0f}",
            f"{train_ms/1000:.1f}",
        ])

    # Sort by val_bpb ascending
    rows.sort(key=lambda r: float(r[3]))

    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "  "
    header_line = sep.join(h.rjust(w) for h, w in zip(headers, widths))
    print(f"\n{'='*len(header_line)}")
    print("  BENCHMARK RESULTS (sorted by val_loss)")
    print(f"{'='*len(header_line)}")
    print(header_line)
    print(sep.join("-" * w for w in widths))
    for row in rows:
        print(sep.join(v.rjust(w) for v, w in zip(row, widths)))
    print()


def save_results(results: list[dict], out: Path):
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark model versions under identical conditions")
    parser.add_argument("--versions", type=str, default=None,
                        help=f"Comma-separated model versions (default: all)")
    parser.add_argument("--minutes", type=float, default=10,
                        help="Wallclock minutes per model (default: 10)")
    parser.add_argument("--batch", type=int, default=491520,
                        help="TRAIN_BATCH_TOKENS (default: 491520)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup steps (default: 5)")
    parser.add_argument("--output", type=str, default="logs/benchmark_results.json",
                        help="Output JSON path (default: logs/benchmark_results.json)")
    args = parser.parse_args()

    versions = args.versions.split(",") if args.versions else ALL_VERSIONS
    nproc = detect_gpus()

    print(f"Benchmarking {len(versions)} models: {', '.join(versions)}")
    print(f"  {args.minutes}min each, batch={args.batch}, warmup={args.warmup}, GPUs={nproc}")

    results = []
    for v in versions:
        manifest = run_one(v, args.minutes, args.batch, args.warmup, nproc)
        if manifest:
            results.append(manifest)

    print_table(results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, out)


if __name__ == "__main__":
    main()
