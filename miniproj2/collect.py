"""
Collect ncu metrics for the flash-attention prefill kernel and cache them.

Relevant metrics and what they validate in the model:
  dram__bytes_read.sum / dram__bytes_write.sum
      → model bytes_dram (v1/v2/v3 DRAM bound)
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld/st.sum
      → model bytes_l2 (bytes requested from L2 to SM, v2/v3 L2 bound)
  lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld/st.sum
      → actual L2 miss traffic (bytes that had to go to DRAM from L2)
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
      → model flops / 2  (each FFMA = 2 FLOPs)
  sm__sass_data_bytes_mem_shared.sum
      → shared-memory traffic for derived shared-memory arithmetic intensity
  gpu__time_duration.sum
      → model pred_ms  (reported in nanoseconds)
  sm__warps_active.avg.pct_of_peak_sustained_active
      → occupancy % (validates wave/block model in v3)

Usage:
    python3 collect.py              # collect for DEFAULT_SIZES
    python3 collect.py 4096 65536   # collect for specific S values
    python3 collect.py --Br 8 --Bc 16 4096 65536
    python3 collect.py --Br 8 --Bc 16 --show
                                    # print matching cache contents without collecting
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys

CACHE_FILE_TEMPLATE = "ncu_cache_Br{Br}_Bc{Bc}.json"
BINARY     = "./flash_attention"

METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum",
    "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_data_bytes_mem_shared.sum",
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

DEFAULT_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 65536]
DEFAULT_BR = 16
DEFAULT_BC = 16


def cache_key(S, Br=DEFAULT_BR, Bc=DEFAULT_BC):
    return f"S{S}_Br{Br}_Bc{Bc}"


def cache_file(Br=DEFAULT_BR, Bc=DEFAULT_BC):
    return CACHE_FILE_TEMPLATE.format(Br=Br, Bc=Bc)


def run_ncu(S):
    """Run ncu for one S value; returns (stdout, stderr)."""
    cmd = [
        "ncu",
        "--launch-skip",  "1",   # skip warmup kernel
        "--launch-count", "1",   # profile only the timed run
        "--cache-control", "all",
        "--csv",
        "--metrics", ",".join(METRICS),
        BINARY, str(S),
    ]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr


def parse_csv(stdout):
    """
    ncu --csv output contains status lines (==PROF==), binary stdout, and CSV rows.
    Only CSV rows start with a double-quote character; filter to those.
    Returns {metric_name: float_value}.
    """
    csv_lines = [l for l in stdout.splitlines() if l.startswith('"')]
    if not csv_lines:
        return {}

    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    metrics = {}
    for row in reader:
        name  = row.get("Metric Name", "").strip().strip('"')
        value = row.get("Metric Value", "").strip().strip('"').replace(",", "")
        if not name:
            continue
        try:
            metrics[name] = float(value)
        except ValueError:
            pass   # skip non-numeric metrics (units, labels)
    return metrics


def load_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache, path):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def show_cache(path):
    cache = load_cache(path)
    if not cache:
        print("Cache is empty.")
        return
    for key, entry in cache.items():
        print(f"\n[{key}]")
        ns = entry.get("gpu__time_duration.sum")
        print(f"  runtime:          {ns/1e6:.4f} ms" if ns else "  runtime:          n/a")

        dr = entry.get("dram__bytes_read.sum", 0)
        dw = entry.get("dram__bytes_write.sum", 0)
        print(f"  DRAM total:       {(dr+dw)/1e6:.2f} MB  (reads {dr/1e6:.2f}  writes {dw/1e6:.2f})")

        l2r = entry.get("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum", 0)
        l2w = entry.get("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum", 0)
        print(f"  L2 total:         {(l2r+l2w)/1e6:.2f} MB  (reads {l2r/1e6:.2f}  writes {l2w/1e6:.2f})")

        l1r = entry.get("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", 0)
        l1w = entry.get("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum", 0)
        print(f"  L1 total:         {(l1r+l1w)/1e6:.2f} MB  (reads {l1r/1e6:.2f}  writes {l1w/1e6:.2f})")

        ff = entry.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
        print(f"  FP32 FMAs:        {ff:.3e}  ({ff*2:.3e} FLOPs)" if ff else "  FP32 FMAs:        n/a")
        occ = entry.get("sm__warps_active.avg.pct_of_peak_sustained_active")
        print(f"  occupancy:        {occ:.1f}%" if occ else "  occupancy:        n/a")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="*", type=int)
    parser.add_argument("--Br", type=int, default=DEFAULT_BR)
    parser.add_argument("--Bc", type=int, default=DEFAULT_BC)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    path = cache_file(Br=args.Br, Bc=args.Bc)

    if args.show:
        show_cache(path)
        return

    sizes = args.sizes if args.sizes else DEFAULT_SIZES

    cache = load_cache(path)

    for S in sizes:
        key = cache_key(S, Br=args.Br, Bc=args.Bc)
        print(f"\nCollecting S={S}  (key={key}) ...")
        stdout, stderr = run_ncu(S)
        metrics = parse_csv(stdout)

        if not metrics:
            print(f"  ERROR: no metrics parsed.")
            if stderr:
                print(f"  stderr: {stderr[:400]}")
            continue

        cache[key] = {"S": S, "Br": args.Br, "Bc": args.Bc, **metrics}
        ns = metrics.get("gpu__time_duration.sum")
        print(f"  OK — {len(metrics)} metrics  runtime={ns/1e6:.4f} ms" if ns
              else f"  OK — {len(metrics)} metrics")

    save_cache(cache, path)
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
