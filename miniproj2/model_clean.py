"""
Performance models for the flash-attention prefill kernel on TitanV (GV100).

Usage:
    python3 model_clean.py                   # sweep DEFAULT_SIZES
    python3 model_clean.py 4096 65536        # specific sequence lengths
    python3 model_clean.py --Br 16 --Bc 16 4096 65536

Tile parameters (Br, Bc) are keyword arguments on each model function;
they default to the values compiled into flash_attention.cu.
"""

import argparse
import json
import os
from math import ceil
from math import floor

NCU_CACHE_FILE_TEMPLATE = "ncu_cache_Br{Br}_Bc{Bc}.json"

# ---------------------------------------------------------------------------
# TitanV hardware parameters
# ---------------------------------------------------------------------------
PEAK_FLOPS  = 13.8e12    # FP32 FLOP/s
DRAM_BW     = 652.8e9    # bytes/s (HBM2)
L2_BW       = 3000e9     # bytes/s (Volta estimate)
L2_CAPACITY = 4.5e6      # bytes
SMEM_PER_SM = 96e3       # bytes (max configurable)
NUM_SMS     = 80
SM_CLOCK    = 1.200e9    # Hz (TitanV boost)

BYTES_PER_FLOAT = 4

# Volta instruction latencies (cycles) — tunable
FMA_LAT  = 4    # FP32 FMA
EXP_LAT = 20
SHFL_LAT = 32   # __shfl_down_sync
SYNC_LAT = 30   # __syncthreads
N_SHFL   = 5    # log2(32) reduction steps
N_SYNCS  = 7    # __syncthreads calls per KV tile
L2_LD_LAT = 192
SMEM_LAT = 19
DRAM_LAT = 500    # very rough estimate (varies widely based on access pattern and concurrency)
# ---------------------------------------------------------------------------
# Kernel tile parameters (match flash_attention.cu)
# ---------------------------------------------------------------------------
D  = 64
Br = 16
Bc = 16

# ---------------------------------------------------------------------------
# Measured runtimes — fill in after running ./flash_attention
# ---------------------------------------------------------------------------
measured_ms = {
    4096:  5.3279,
    65536: 1102.8531,
}

DEFAULT_SIZES = [512, 1024, 2048, 4096, 8192, 65536]

# ---------------------------------------------------------------------------
# Model v1: simple roofline
#   Assumes minimum possible DRAM traffic (Q+K+V+O each loaded/stored once).
# ---------------------------------------------------------------------------
def model_roofline(S, D=D, Br=Br, Bc=Bc):
    flops      = 2 * S * S * D
    theoretical_bytes_dram = 4 * S * D * BYTES_PER_FLOAT   
    kv_bytes = 2 * S * D * BYTES_PER_FLOAT
    shared_mem_bytes_per_block = (2 * Br * D +
                                  2 * Bc * D + 
                                  Br * Bc +
                                  2 * Br * Bc +
                                  5 * Br) * BYTES_PER_FLOAT

    # wave calculation
    num_blocks = (S + Br - 1) // Br
    threads_per_block = Br * D
    blocks_per_sm_by_shared_mem = floor(SMEM_PER_SM / shared_mem_bytes_per_block)
    max_blocks_per_sm = min(2048 / threads_per_block, blocks_per_sm_by_shared_mem)
    num_waves = ceil(num_blocks / (max_blocks_per_sm * NUM_SMS))

    if theoretical_bytes_dram > L2_CAPACITY:
        bytes_dram = theoretical_bytes_dram
        bytes_dram += (num_waves - 1) * kv_bytes
        bytes_dram_write = S * D * BYTES_PER_FLOAT
        bytes_dram_read = bytes_dram - bytes_dram_write
    else:
        bytes_dram = 3 * S * D * BYTES_PER_FLOAT 
        bytes_dram_read = 3 * S * D * BYTES_PER_FLOAT
        bytes_dram_write = 0  


    intensity = flops / bytes_dram
    
    t_compute = flops / PEAK_FLOPS
    t_memory  = bytes_dram / DRAM_BW
    t_pred    = max(t_compute, t_memory)

    performance = flops / (t_pred * 1e12)   # in TFLOP/s

    return {
        "pred_ms":  t_pred * 1e3,
        "bound":    "compute" if t_compute >= t_memory else "DRAM",
        "flops":    flops,
        "tflops/s":  performance,
        "bytes_dram": bytes_dram,
        "bytes_dram_read": bytes_dram_read,
        "bytes_dram_write": bytes_dram_write,
        "bytes_l2": 0,
        "bytes_l2_read": 0,
        "bytes_l2_write": 0,
        "bytes_l1": 0,
        "bytes_l1_read": 0,
        "bytes_l1_write": 0,
        "intensity_shared_mem": 0,
        "intensity_l1": 0,
        "intensity_l2": 0,
        "intensity_dram": intensity
    }



def model_roofline_l2_serial(S, D=D, Br=Br, Bc=Bc):
    # 3S^2 for softmax computations (row sum, score - max, scaling by root d)
    # 2S^2D / Bc for the line Oi[row * D + d] = Oi[row * D + d] * corr[row] + acc;
    # 1 FMA for each KV tile (# tiles = S / Bc) for each entry in output matrix (S * D)
    # mm_flops = 2 * (S + 1) * S * D + (4 * S ** 2) + (2 * (S ** 2) * D / Bc)
    
    flops = (8 * S * S * D) + (2 * (S ** 2) * D / Bc) + (5 * S * S) + (3 * (S * S) / Bc)

    theoretical_bytes_dram = 4 * S * D * BYTES_PER_FLOAT   
    kv_bytes = 2 * S * D * BYTES_PER_FLOAT
    shared_mem_bytes_per_block = (2 * Br * D +
                                  2 * Bc * D + 
                                  Br * Bc +
                                  2 * Br * Bc +
                                  5 * Br) * BYTES_PER_FLOAT

    # wave calculation
    num_blocks = (S + Br - 1) // Br
    threads_per_block = Br * D
    blocks_per_sm_by_shared_mem = floor(SMEM_PER_SM / shared_mem_bytes_per_block)
    max_blocks_per_sm = min(2048 / threads_per_block, blocks_per_sm_by_shared_mem)
    num_waves = ceil(num_blocks / (max_blocks_per_sm * NUM_SMS))

    num_blocks_last_wave = num_blocks - (num_waves - 1) * max_blocks_per_sm * NUM_SMS
    
    if num_blocks_last_wave <= NUM_SMS:
        num_blocks_reuse = num_blocks - num_blocks_last_wave - (num_waves - 1) * NUM_SMS
    else:
        num_blocks_reuse = num_blocks - num_waves * NUM_SMS

    # shared mem
    intensity_shared_mem = 0.0

    num_iters  = S // Bc

    # initialize row max, row sum, q tile and o tile
    I = 8 * Br * (D + 1)

    # repeated S/Bc (num_iters) times
    P = 4 * (
        2 * Bc * D + # write Kj, Vj
        Bc * Br * D * 2 + # each thread computes Qi[] * Kj[]
        Br * Bc * 2 + # write to dot_warp
        Bc * Br * 2 + # read from dot_warp
        Bc * Br + # write to Sij
        Br + # read from rowmax
        Bc * Br + # read from Sij
        3 * Br + # corr, rowmax, rowmaxnew
        2 * Br * Bc + # read and write Sij
        Br + # read rowMaxNew, read once for each warp
        Br * Bc + # read Sij
        3 * Br + # rsumNew, rsum, corr
        Br * Bc * 2 + # read Sij, all threads in a warp read same address = 1 access
        Br * Bc * D + # Vij reads
        2 * Br * D + # read and write Oi
        Br * 2 + # corr, read once for each warp
        Br * 4
    )
    F = 4 * Br * (D)
    bytes_shared_mem_total = num_blocks * (I + num_iters * P + F)
    intensity_shared_mem = flops / bytes_shared_mem_total

    # l1
    bytes_l1_write = S * D * BYTES_PER_FLOAT
    bytes_l1_read = ((S / Bc) * 2 + 1) * S * D * BYTES_PER_FLOAT
    bytes_l1 = bytes_l1_read + bytes_l1_write

    # L2: misses from L1
    l1_hit_bytes = num_blocks_reuse * kv_bytes
    
    if Br == 16:
        l1_hit_rate = 0.45
    else:
        l1_hit_rate = 0.65

    bytes_l2_read = bytes_l1_read - (l1_hit_bytes * l1_hit_rate) 
    bytes_l2_write = S * D * BYTES_PER_FLOAT
    bytes_l2 = bytes_l2_read + bytes_l2_write

    if theoretical_bytes_dram > L2_CAPACITY:
        bytes_dram = theoretical_bytes_dram
        bytes_dram += (num_waves - 1) * kv_bytes
        bytes_dram_write = S * D * BYTES_PER_FLOAT
        bytes_dram_read = bytes_dram - bytes_dram_write
    else:
        bytes_dram = 3 * S * D * BYTES_PER_FLOAT # Q, K, V (O stays in L2)
        bytes_dram_read = 3 * S * D * BYTES_PER_FLOAT
        bytes_dram_write = 0

    intensity_dram = flops / bytes_dram
    intensity_l1 = flops / bytes_l1
    intensity_l2 = flops / bytes_l2
    
    t_compute = flops / PEAK_FLOPS
    t_dram  = bytes_dram / DRAM_BW
    t_l2 = bytes_l2 / L2_BW

    # KV tile load: one latency + bandwidth transfer term
    kv_tile_bytes      = 2 * Bc * D * BYTES_PER_FLOAT
    l2_bw_per_sm_bpc   = (L2_BW  / NUM_SMS) / SM_CLOCK   # bytes/cycle/SM
    dram_bw_per_sm_bpc = (DRAM_BW / NUM_SMS) / SM_CLOCK

    # does total KV data fit in L2? if not must go to DRAM, increasing latency
    total_kv = 2 * S * D * BYTES_PER_FLOAT
    ld_lat = (L2_LD_LAT + kv_tile_bytes / l2_bw_per_sm_bpc if total_kv < L2_CAPACITY
            else DRAM_LAT + kv_tile_bytes / dram_bw_per_sm_bpc)

    kv_lat = (N_SYNCS * SYNC_LAT +
            ld_lat +
            Bc * (N_SHFL * SHFL_LAT + FMA_LAT) +   # QK dot products
            (D // 32) * (SMEM_LAT + FMA_LAT) +       # warp partial reduce
            Bc * (SMEM_LAT + FMA_LAT) +              # rowmax scan
            (Bc + 1) * EXP_LAT +                     # exp(Sij) + exp(corr)
            Bc * (SMEM_LAT + FMA_LAT) +              # rowsum
            Bc * (2 * SMEM_LAT + 2 * FMA_LAT) +      # PV accumulate
            2 * SMEM_LAT + FMA_LAT +                 # Oi correction
            2 * SMEM_LAT)                            # rowmax/rsum commit

    q_load_lat = DRAM_LAT

    t_lat = num_waves * (kv_lat * num_iters + q_load_lat) / SM_CLOCK

    t_pred = max(t_compute, t_dram, t_l2, t_lat)

    if t_pred == t_compute: bound = "compute"
    elif t_pred == t_dram:  bound = "DRAM"
    elif t_pred == t_l2: bound = "L2"
    else: bound = "serial"

    # in TFLOPS
    performance = flops / (t_pred * 1e12)

    return {
        "pred_ms":  t_pred * 1e3,
        "bound":    bound,
        "flops":    flops,
        "tflops/s":  performance,
        "bytes_dram": bytes_dram,
        "bytes_dram_read": bytes_dram_read,
        "bytes_dram_write": bytes_dram_write,
        "bytes_l2": bytes_l2,
        "bytes_l2_read": bytes_l2_read,
        "bytes_l2_write": bytes_l2_write,
        "bytes_l1": bytes_l1,
        "bytes_l1_read": bytes_l1_read,
        "bytes_l1_write": bytes_l1_write,
        "intensity_shared_mem": intensity_shared_mem,
        "intensity_l1": intensity_l1,
        "intensity_l2": intensity_l2,
        "intensity_dram": intensity_dram
    }




# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    # "roofline":     model_roofline,
    "roofline_l2_serial": model_roofline_l2_serial
}


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def _mape(pred, meas):
    return abs(pred - meas) / meas * 100.0 if meas is not None else None


def compare(sizes, Br=Br, Bc=Bc):
    names = list(MODELS.keys())
    col_w = 26   # width per model column (pred ms + MAPE% + TFLOPS)

    # Header
    sep = "  |  "
    hdr1 = f"{'S':>8}{sep}"
    hdr2 = f"{'':>8}{sep}"
    div  = "-" * 8 + "-+-"
    for name in names:
        hdr1 += f"{name:^{col_w}s}  "
        hdr2 += f"{'pred ms':>8}{'MAPE%':>8}{'TFLOPS':>8}  "
        div  += "-" * col_w + "--"
    hdr1 += sep + "meas ms"
    hdr2 += sep + ""
    div  += "-+---------"

    print(f"\n=== Model comparison (flash-attention prefill, TitanV) ===")
    print(hdr1)
    print(hdr2)
    print(div)

    for s in sizes:
        m = measured_ms.get(s)
        row = f"{s:>8,}{sep}"
        for name, fn in MODELS.items():
            r = fn(s, Br=Br, Bc=Bc)
            mape_str  = f"{_mape(r['pred_ms'], m):>7.1f}%" if m is not None else f"{'':>8}"
            tflops_str = f"{r['tflops/s'] / 1e12:>7.2f}T"
            row += f"{r['pred_ms']:>8.3f}{mape_str}{tflops_str}  "
        row += sep + (f"{m:>8.4f}" if m is not None else "")
        print(row)

    print()
    print("  Hardware:  "
          f"Peak={PEAK_FLOPS/1e12:.1f} TFLOPS  "
          f"DRAM={DRAM_BW/1e9:.0f} GB/s  "
          f"L2={L2_BW/1e9:.0f} GB/s (cap {L2_CAPACITY/1e6:.1f} MB)  "
          f"Clock={SM_CLOCK/1e9:.3f} GHz  "
          f"{NUM_SMS} SMs")
    print(f"  Latencies: FMA={FMA_LAT}cy  SHFL={SHFL_LAT}cy  SYNC={SYNC_LAT}cy  "
          f"(v3 uses cycles_kv={Bc*(2*FMA_LAT+N_SHFL*SHFL_LAT)+N_SYNCS*SYNC_LAT} per KV tile)")


# ---------------------------------------------------------------------------
# NCU cache helpers
# ---------------------------------------------------------------------------

def ncu_cache_file(Br=Br, Bc=Bc):
    return NCU_CACHE_FILE_TEMPLATE.format(Br=Br, Bc=Bc)


def load_ncu_cache(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _ncu_key(S, Br=Br, Bc=Bc):
    return f"S{S}_Br{Br}_Bc{Bc}"


def get_visualization_data(S, tile_pairs=((4, 4), (8, 8), (16, 16)), model_fn=model_roofline_l2_serial):
    metrics = [
        ("pred_ms", "runtime", "ms"),
        ("flops", "FLOPs", "GFLOPs"),
        ("tflops/s", "performance", "TFLOP/s"),
        ("bytes_dram", "DRAM total", "MB"),
        ("bytes_dram_read", "DRAM reads", "MB"),
        ("bytes_dram_write", "DRAM writes", "MB"),
        ("bytes_l2", "L2 total", "MB"),
        ("bytes_l2_read", "L2 reads", "MB"),
        ("bytes_l2_write", "L2 writes", "MB"),
        ("bytes_l1", "L1 total", "MB"),
        ("bytes_l1_read", "L1 reads", "MB"),
        ("bytes_l1_write", "L1 writes", "MB"),
        ("intensity_dram", "DRAM intensity", "FLOP/B"),
        ("intensity_l2", "L2 intensity", "FLOP/B"),
        ("intensity_l1", "L1 intensity", "FLOP/B"),
        ("intensity_shared_mem", "shared mem intensity", "FLOP/B"),
    ]
    data = {
        "S": S,
        "tile_pairs": list(tile_pairs),
        "metrics": {
            key: {"label": label, "unit": unit, "values": []}
            for key, label, unit in metrics
        },
    }

    def safe_div(num, den):
        return num / den if num is not None and den else None

    for tile_Br, tile_Bc in tile_pairs:
        cache = load_ncu_cache(ncu_cache_file(Br=tile_Br, Bc=tile_Bc))
        entry = cache.get(_ncu_key(S, tile_Br, tile_Bc))
        r = model_fn(S, Br=tile_Br, Bc=tile_Bc)

        def nget(*keys):
            if not entry:
                return None
            return sum(entry.get(k, 0) for k in keys)

        ncu_flops = (nget("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum")
                     + 2 * nget("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
                     + nget("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")
                     if entry else None)
        ncu_runtime_s = (nget("gpu__time_duration.sum") / 1e9 if entry else None)
        actuals = {
            "pred_ms": nget("gpu__time_duration.sum") / 1e6 if entry else None,
            "flops": ncu_flops / 1e9 if ncu_flops is not None else None,
            "tflops/s": safe_div(ncu_flops, ncu_runtime_s * 1e12 if ncu_runtime_s else None),
            "bytes_dram": nget("dram__bytes_read.sum", "dram__bytes_write.sum"),
            "bytes_dram_read": nget("dram__bytes_read.sum"),
            "bytes_dram_write": nget("dram__bytes_write.sum"),
            "bytes_l2": nget("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum",
                              "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum"),
            "bytes_l2_read": nget("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum"),
            "bytes_l2_write": nget("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum"),
            "bytes_l1": nget("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
                              "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"),
            "bytes_l1_read": nget("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"),
            "bytes_l1_write": nget("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"),
            "intensity_dram": safe_div(ncu_flops, nget("dram__bytes_read.sum",
                                                       "dram__bytes_write.sum")),
            "intensity_l2": safe_div(ncu_flops,
                                      nget("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum",
                                           "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum")),
            "intensity_l1": safe_div(ncu_flops,
                                      nget("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
                                           "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum")),
            "intensity_shared_mem": safe_div(ncu_flops, nget("sm__sass_data_bytes_mem_shared.sum")),
        }

        for key, _, _ in metrics:
            pred = r.get(key)
            if key == "flops":
                pred = pred / 1e9 if pred is not None else None
            elif key.startswith("bytes_"):
                pred = pred / 1e6 if pred is not None else None
            actual = actuals[key]
            if key.startswith("bytes_"):
                actual = actual / 1e6 if actual is not None else None
            data["metrics"][key]["values"].append({
                "Br": tile_Br,
                "Bc": tile_Bc,
                "predicted": pred,
                "actual": actual,
                "error_pct": _mape(pred, actual) if pred is not None and actual else None,
            })

    return data


def compare_ncu(sizes, model_fn=None, Br=Br, Bc=Bc):
    """
    For each S in sizes, compare model predictions against ncu_cache.json.

    Memory hierarchy mapping (ncu metric → what it measures):
      L1 boundary  — l1tex__t_bytes_pipe_lsu_mem_global_op_{ld,st}.sum
                     bytes requested by load/store instructions (shader → L1)
      L2 boundary  — lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_{ld,st}.sum
                     bytes that missed L1 (L1 → L2)
      DRAM         — dram__bytes_{read,write}.sum
                     bytes that missed L2 (L2 → HBM)

    Model fields used:
      bytes_dram → DRAM total
      bytes_l2   → L1 total  (model labels this "L2→SM" traffic)

    If model_fn is None, iterates over all models in MODELS.
    """
    if model_fn is None:
        for name, fn in MODELS.items():
            compare_ncu(sizes, model_fn=fn, Br=Br, Bc=Bc)
        return
    cache = load_ncu_cache(ncu_cache_file(Br=Br, Bc=Bc))

    col = 12
    hdr_line = f"{'S':>8}  {'quantity':<24}  {'model':>{col}}  {'ncu':>{col}}  {'error%':>8}"
    print(f"\n=== NCU metric comparison  (model: {model_fn.__name__}) ===")
    print(hdr_line)
    print("-" * len(hdr_line))

    for S in sizes:
        key   = _ncu_key(S, Br, Bc)
        entry = cache.get(key)
        r     = model_fn(S, Br=Br, Bc=Bc)

        def nget(*keys):
            if not entry:
                return None
            return sum(entry.get(k, 0) for k in keys)

        def safe_div(num, den):
            return num / den if num is not None and den else None

        ncu_flops = (nget("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum")
                     + 2 * nget("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
                     + nget("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")
                     if entry else None)
        ncu_dram_bytes = nget("dram__bytes_read.sum", "dram__bytes_write.sum")
        ncu_l2_bytes = nget("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum",
                            "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum")
        ncu_l1_bytes = nget("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
                            "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum")
        ncu_shared_mem_bytes = nget("sm__sass_data_bytes_mem_shared.sum")
        ncu_runtime_s = (nget("gpu__time_duration.sum") / 1e9 if entry else None)

        # Rows: (label, model_value, ncu_value, unit)
        # unit: "MB" | "GFLOPs" | "FLOP/B" | "TFLOP/s" | "ms"
        #
        # Model key convention (matches model_roofline_l2 and friends):
        #   bytes_dram / bytes_dram_read / bytes_dram_write  → dram__bytes_*
        #   bytes_l2   / bytes_l2_read   / bytes_l2_write    → lts__* (L1-miss traffic)
        #   bytes_l1   / bytes_l1_read   / bytes_l1_write    → l1tex__* (all instruction traffic)
        rows = [
            # ---- DRAM ----
            ("DRAM total (MB)",
             r.get("bytes_dram"),
             nget("dram__bytes_read.sum", "dram__bytes_write.sum"),
             "MB"),
            ("  reads (MB)",
             r.get("bytes_dram_read"),
             nget("dram__bytes_read.sum"),
             "MB"),
            ("  writes (MB)",
             r.get("bytes_dram_write"),
             nget("dram__bytes_write.sum"),
             "MB"),
            # ---- L2 (L1-miss traffic) ----
            ("L2 total (MB)",
             r.get("bytes_l2"),
             nget("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum",
                  "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum"),
             "MB"),
            ("  reads (MB)",
             r.get("bytes_l2_read"),
             nget("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum"),
             "MB"),
            ("  writes (MB)",
             r.get("bytes_l2_write"),
             nget("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum"),
             "MB"),
            # ---- L1 (all global-mem instruction traffic) ----
            ("L1 total (MB)",
             r.get("bytes_l1"),
             nget("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
                  "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"),
             "MB"),
            ("  reads (MB)",
             r.get("bytes_l1_read"),
             nget("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"),
             "MB"),
            ("  writes (MB)",
             r.get("bytes_l1_write"),
             nget("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"),
             "MB"),
            # ---- arithmetic intensity ----
            ("DRAM intensity",
             r.get("intensity_dram"),
             safe_div(ncu_flops, ncu_dram_bytes),
             "FLOP/B"),
            ("L2 intensity",
             r.get("intensity_l2"),
             safe_div(ncu_flops, ncu_l2_bytes),
             "FLOP/B"),
            ("L1 intensity",
             r.get("intensity_l1"),
             safe_div(ncu_flops, ncu_l1_bytes),
             "FLOP/B"),
            ("shared mem intensity",
             r.get("intensity_shared_mem"),
             safe_div(ncu_flops, ncu_shared_mem_bytes),
             "FLOP/B"),
            # ---- compute / timing ----
            ("FLOPs (GFLOPs)",
             r["flops"] / 1e9,
             (ncu_flops / 1e9 if ncu_flops is not None else None),
             "GFLOPs"),
            ("performance (TFLOP/s)",
             r.get("tflops/s"),
             safe_div(ncu_flops, ncu_runtime_s * 1e12 if ncu_runtime_s else None),
             "TFLOP/s"),
            ("runtime (ms)",
             r["pred_ms"],
             (nget("gpu__time_duration.sum") / 1e6 if entry else None),
             "ms"),
        ]

        for i, (label, pred, meas, unit) in enumerate(rows):
            s_str = f"{S:,}" if i == 0 else ""

            if unit == "MB":
                def fmt(v): return f"{v/1e6:{col}.2f}"
            elif unit == "GFLOPs":
                def fmt(v): return f"{v:{col}.1f}"
            elif unit == "FLOP/B":
                def fmt(v): return f"{v:{col}.2f}"
            elif unit == "TFLOP/s":
                def fmt(v): return f"{v:{col}.2f}"
            elif unit == "ms":
                def fmt(v): return f"{v:{col}.3f}"
            else:
                def fmt(v): return f"{v:{col}.1f}"

            pred_s = fmt(pred) if pred is not None else f"{'—':>{col}}"
            meas_s = fmt(meas) if meas is not None else f"{'no data':>{col}}"
            err_s  = (f"{_mape(pred, meas):>8.1f}%"
                      if pred is not None and meas is not None and meas != 0
                      else f"{'—':>9}")

            print(f"{s_str:>8}  {label:<24}  {pred_s}  {meas_s}  {err_s}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="*", type=int)
    parser.add_argument("--Br", type=int, default=Br)
    parser.add_argument("--Bc", type=int, default=Bc)
    args = parser.parse_args()

    sizes = args.sizes if args.sizes else DEFAULT_SIZES
    compare(sizes, Br=args.Br, Bc=args.Bc)
    compare_ncu(sizes, Br=args.Br, Bc=args.Bc)   # iterates over all MODELS when no model_fn given
