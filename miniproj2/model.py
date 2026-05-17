"""
Performance models for the flash-attention prefill kernel on TitanV (GV100).

Usage:
    python3 model.py                   # sweep DEFAULT_SIZES
    python3 model.py 4096 65536        # specific sequence lengths

Tile parameters (Br, Bc) are keyword arguments on each model function;
they default to the values compiled into flash_attention.cu.
"""

import json
import os
import sys
from math import ceil

NCU_CACHE_FILE = "ncu_cache.json"

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
L2_LD_LAT = 192 # TODO: account for DRAM
SMEM_LAT = 19
# ---------------------------------------------------------------------------
# Kernel tile parameters (match flash_attention.cu)
# ---------------------------------------------------------------------------
D  = 64
Br = 8
Bc = 16

# ---------------------------------------------------------------------------
# Measured runtimes — fill in after running ./flash_attention
# ---------------------------------------------------------------------------
measured_ms = {
    4096:  5.3279,
    65536: 1102.8531,
}

DEFAULT_SIZES = [512, 1024, 2048, 4096, 8192, 65536]

# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------
MISS_RATE_PER_WAVE = 0.

# ---------------------------------------------------------------------------
# Model v1: simple roofline
#   Assumes minimum possible DRAM traffic (Q+K+V+O each loaded/stored once).
#   Ignores repeated K/V reloads and L2 hierarchy.
# ---------------------------------------------------------------------------
def model_roofline(S, D=D, Br=Br, Bc=Bc):
    flops      = 2 * S * S * D
    theoretical_bytes_dram = 4 * S * D * BYTES_PER_FLOAT   
    kv_bytes = 2 * S * D * BYTES_PER_FLOAT

    # wave calculation
    num_blocks = (S + Br - 1) // Br
    threads_per_block = Br * D
    # TODO: check this
    blocks_per_sm = 2048 / threads_per_block
    num_waves = ceil(num_blocks / (blocks_per_sm * NUM_SMS)) 

    if theoretical_bytes_dram > L2_CAPACITY:
        bytes_dram = theoretical_bytes_dram
        bytes_dram += (num_waves - 1) * kv_bytes
    else:
        bytes_dram = 3 * S * D * BYTES_PER_FLOAT   



    intensity = flops / bytes_dram
    
    t_compute = flops / PEAK_FLOPS
    t_memory  = bytes_dram / DRAM_BW
    t_pred    = max(t_compute, t_memory)

    return {
        "pred_ms":  t_pred * 1e3,
        "bound":    "compute" if t_compute >= t_memory else "DRAM",
        "flops":    flops,
        "bytes_dram": bytes_dram,
        "bytes_dram_read": 0,
        "bytes_dram_write": 0,
        "bytes_l2": 0,
        "bytes_l2_read": 0,
        "bytes_l2_write": 0,
        "bytes_l1": 0,
        "bytes_l1_read": 0,
        "bytes_l1_write": 0,
        "flops/s":  flops / t_pred,
    }


def model_roofline_l2(S, D=D, Br=Br, Bc=Bc):
    mm_flops      = 2 * S * S * D
    sm_flops = 4 * S * S
    flops = mm_flops + sm_flops
    theoretical_bytes_dram = 4 * S * D * BYTES_PER_FLOAT   
    kv_bytes = 2 * S * D * BYTES_PER_FLOAT

    # wave calculation
    num_blocks = (S + Br - 1) // Br
    threads_per_block = Br * D
    # TODO: check this
    max_blocks_per_sm = 2048 / threads_per_block
    num_waves = ceil(num_blocks / (max_blocks_per_sm * NUM_SMS)) 
    sms_with_two_blocks = (num_waves - 1) * NUM_SMS + (num_blocks % (NUM_SMS * max_blocks_per_sm)) % NUM_SMS
    

    # l1
    bytes_l1_write = S * D * BYTES_PER_FLOAT
    bytes_l1_read = ((S / Br) * 2 + 1) * S * D * BYTES_PER_FLOAT
    bytes_l1 = bytes_l1_read + bytes_l1_write

    # L2: misses from L1
    l1_hit_bytes = sms_with_two_blocks * kv_bytes
    l1_hit_rate = 0.5
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

    intensity = flops / bytes_dram
    
    t_compute = flops / PEAK_FLOPS
    t_dram  = bytes_dram / DRAM_BW
    t_l2 = bytes_l2 / L2_BW
    t_pred    = max(t_compute, t_dram, t_l2)

    return {
        "pred_ms":  t_pred * 1e3,
        "bound":    "compute" if t_compute >= t_dram else "DRAM",
        "flops":    flops,
        "bytes_dram": bytes_dram,
        "bytes_dram_read": bytes_dram_read,
        "bytes_dram_write": bytes_dram_write,
        "bytes_l2": bytes_l2,
        "bytes_l2_read": bytes_l2_read,
        "bytes_l2_write": bytes_l2_write,
        "bytes_l1": bytes_l1,
        "bytes_l1_read": bytes_l1_read,
        "bytes_l1_write": bytes_l1_write,
        "flops/s":  flops / t_pred,
    }

def model_roofline_l2_serial(S, D=D, Br=Br, Bc=Bc):
    # 3S^2 for softmax computations (row sum, score - max, scaling by root d)
    # 2S^2D / Bc for the line Oi[row * D + d] = Oi[row * D + d] * corr[row] + acc;
    # 1 FMA for each KV tile (# tiles = S / Bc) for each entry in output matrix (S * D)
    mm_flops      = 2 * (S + 1) * S * D + (3 * S ** 2) + (2 * (S ** 2) * D / Bc)
    sm_flops = 4 * S * S
    flops = mm_flops + sm_flops
    theoretical_bytes_dram = 4 * S * D * BYTES_PER_FLOAT   
    kv_bytes = 2 * S * D * BYTES_PER_FLOAT

    # wave calculation
    num_blocks = (S + Br - 1) // Br
    threads_per_block = Br * D
    # TODO: check this
    max_blocks_per_sm = 2048 / threads_per_block
    num_waves = ceil(num_blocks / (max_blocks_per_sm * NUM_SMS))
    num_sms_at_least_one_block_last_wave = num_blocks - (num_waves - 1) * NUM_SMS
    sms_with_two_blocks = (num_waves - 1) * NUM_SMS + (num_blocks % (NUM_SMS * max_blocks_per_sm)) % min(NUM_SMS, num_sms_at_least_one_block_last_wave)
    

    # l1
    bytes_l1_write = S * D * BYTES_PER_FLOAT
    bytes_l1_read = ((S / Bc) * 2 + 1) * S * D * BYTES_PER_FLOAT
    bytes_l1 = bytes_l1_read + bytes_l1_write

    # L2: misses from L1
    l1_hit_bytes = sms_with_two_blocks * kv_bytes
    l1_hit_rate = 0.45
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
    intensity_l2 = flops / bytes_l2
    
    t_compute = flops / PEAK_FLOPS
    t_dram  = bytes_dram / DRAM_BW
    t_l2 = bytes_l2 / L2_BW

    # sync_lat = N_SYNCS * SYNC_LAT
    # qk_lat = Bc * (N_SHFL * SHFL_LAT + FMA_LAT)
    # pv_lat = Bc * FMA_LAT + 1
    # max_lat = Bc * FMA_LAT
    # row_sum_lat = Bc * FMA_LAT + 1
    # kv_lat = sync_lat + qk_lat + pv_lat + max_lat + row_sum_lat

    sync_lat    = N_SYNCS * SYNC_LAT
    ld_lat      = L2_LD_LAT
    qk_lat      = Bc * (N_SHFL * SHFL_LAT + FMA_LAT)
    reduce_lat  = (D // 32) * (SMEM_LAT + FMA_LAT)
    max_lat     = Bc * (SMEM_LAT + FMA_LAT)
    exp_lat     = 2 * EXP_LAT
    rowsum_lat  = Bc * (SMEM_LAT + FMA_LAT)
    pv_lat      = Bc * (2 * SMEM_LAT + 2 * FMA_LAT)

    kv_lat = sync_lat + ld_lat + qk_lat + reduce_lat + max_lat + exp_lat + rowsum_lat + pv_lat

    stall_fraction = Bc * N_SHFL * SHFL_LAT / kv_lat
    effective_lat  = kv_lat / (1 - stall_fraction)   # stalls inflate wall-clock time

    # TODO: check nubmer of blocks per sm
    t_lat = num_waves * effective_lat * ceil(S / Bc)
    t_lat = t_lat / SM_CLOCK

    t_pred    = max(t_compute, t_dram, t_l2, t_lat)

    if t_pred == t_compute: bound = "compute"
    elif t_pred == t_dram:  bound = "DRAM"
    elif t_pred == t_l2: bound = "L2"
    else:                   bound = "serial"

    return {
        "pred_ms":  t_pred * 1e3,
        "bound":    bound,
        "flops":    flops,
        "bytes_dram": bytes_dram,
        "bytes_dram_read": bytes_dram_read,
        "bytes_dram_write": bytes_dram_write,
        "bytes_l2": bytes_l2,
        "bytes_l2_read": bytes_l2_read,
        "bytes_l2_write": bytes_l2_write,
        "bytes_l1": bytes_l1,
        "bytes_l1_read": bytes_l1_read,
        "bytes_l1_write": bytes_l1_write,
        "flops/s":  flops / t_pred,
    }


# ---------------------------------------------------------------------------
# Model v2: L2-aware roofline
#
#  DRAM traffic depends on three regimes:
#
#  Regime 1 — K+V fits in L2 after subtracting active Q+O tiles
#    (l2_for_kv = L2 − concurrent_blocks × Br×D×4 × 2):
#    K+V loaded once from DRAM.  O writes stay in L2 when num_waves=1;
#    when num_waves>1, wave 2 evicts wave 1's O output to DRAM.
#
#  Regime 2 — K+V > l2_for_kv but kv_bytes ≤ 750 × L1_SIZE (~24 MB):
#    K+V overflows L2 but the two SM-resident blocks share KV tiles via
#    SM L1 (32 KB).  The 160-block wave effectively loads K+V once per wave.
#    kv_reloads = num_waves.
#
#  Regime 3 — kv_bytes > 750 × L1_SIZE:
#    SM-pair L1 sharing breaks down (too many tiles for blocks to stay in
#    sync).  80-way L2 broadcast still holds: one DRAM load per NUM_SMS SMs.
#    kv_reloads = num_q_blocks // NUM_SMS.
# ---------------------------------------------------------------------------
L1_SIZE = 32 * 1024   # Volta per-SM L1 cache, bytes

def model_v2(S, D=D, Br=Br, Bc=Bc):
    num_q_blocks = (S + Br - 1) // Br
    flops        = 2 * S * S * D * (Bc + 2) / Bc
    kv_bytes     = 2 * S * D * BYTES_PER_FLOAT

    threads_per_block = D * Br
    smem_per_block    = (4 * Br * D + Br * Bc + Br * (D // 32) * Bc + 5 * Br) * BYTES_PER_FLOAT
    blocks_per_sm     = min(2048 // threads_per_block, int(SMEM_PER_SM) // smem_per_block)
    concurrent_blocks = NUM_SMS * blocks_per_sm
    num_waves         = ceil(num_q_blocks / concurrent_blocks)
# L2 capacity available for K+V once active Q+O tiles are accounted for l2_for_kv = L2_CAPACITY - concurrent_blocks * Br * D * BYTES_PER_FLOAT * 2

    if kv_bytes <= l2_for_kv:
        kv_reloads = 1
        # O stays in L2 when a single wave covers all query blocks;
        # a second wave evicts the first wave's output lines.
        o_to_dram  = S * D * BYTES_PER_FLOAT if num_waves > 1 else 0
        bytes_dram = 3 * S * D * BYTES_PER_FLOAT + o_to_dram   # Q + K + V (+ O if wave≥2)
        bytes_l2   = (2 + 2 * num_q_blocks) * S * D * BYTES_PER_FLOAT
    elif kv_bytes <= 750 * L1_SIZE:
        # SM-pair L1 sharing intact: 160-way broadcast, one K+V load per wave
        kv_reloads = num_waves
        bytes_dram = kv_reloads * kv_bytes + 2 * S * D * BYTES_PER_FLOAT  # Q + O
        bytes_l2   = (2 + 2 * num_q_blocks) * S * D * BYTES_PER_FLOAT
    else:
        # SM-pair L1 sharing breaks down: 80-way L2 broadcast only
        kv_reloads = num_q_blocks // NUM_SMS
        bytes_dram = kv_reloads * kv_bytes + 2 * S * D * BYTES_PER_FLOAT  # Q + O
        bytes_l2   = (2 + 2 * num_q_blocks) * S * D * BYTES_PER_FLOAT

    t_compute = flops / PEAK_FLOPS
    t_dram    = bytes_dram / DRAM_BW
    t_l2      = bytes_l2   / L2_BW


    # latency time
    # total latency per KV: sync latencies + QK + PV
    # QK latency =  Bc * (FMA_LAT + 5 * SHLF_LAT)
    # PV latency = Bc * FMA_LAT
    # N_SYNCS = 7 
    # SYNC_LAT
    # SHLF_LAT
    # FMA_LAT

    sync_lat = N_SYNCS * SYNC_LAT
    mm_lat = Bc * (N_SHFL * SHLF_LAT + 2 * FMA_LAT)
    kv_lat = sync_lat + qk_lat + pv_lat

    thread_lat = kv_lat * ceil(S / Bc)
    # TODO: check nubmer of blocks per sm
    t_lat = thread_lat * num_waves * blocks_per_sm

    t_pred    = max(t_compute, t_dram, t_l2, t_lat)

    

    if t_pred == t_compute: bound = "compute"
    elif t_pred == t_dram:  bound = "DRAM"
    else:                   bound = "L2"

    return {
        "pred_ms":      t_pred * 1e3,
        "bound":        bound,
        "flops":        flops,
        "bytes_dram":   bytes_dram,
        "bytes_l2":     bytes_l2,
        "kv_fits_l2":   kv_bytes <= l2_for_kv,
        "kv_reloads":   kv_reloads,
        "num_waves":    num_waves,
        "blocks_per_sm": blocks_per_sm,
        "flops/s":      flops / t_pred,
    }


# ---------------------------------------------------------------------------
# Model v3: sequential-dependency model
#   Adds a critical-path latency term for the two serial dependency chains
#   in the inner loop of each KV tile:
#
#   1. QK warp reduction (lines 76-84): Bc iterations, each with a
#      5-step __shfl_down_sync chain → Bc × (FMA_LAT + N_SHFL × SHFL_LAT)
#
#   2. PV accumulation (lines 123-125): Bc FMAs through a dependent
#      accumulator → Bc × FMA_LAT
#
#   3. __syncthreads overhead: N_SYNCS × SYNC_LAT per KV tile
#
#   Blocks per SM is limited by thread count (max 2048 threads/SM on Volta)
#   and shared memory. Waves = ceil(num_q_blocks / (NUM_SMS × blocks_per_sm)).
# ---------------------------------------------------------------------------
def model_v3(S, D=D, Br=Br, Bc=Bc):
    base = model_v2(S, D=D, Br=Br, Bc=Bc)

    num_q_blocks = (S + Br - 1) // Br
    num_kv_tiles = (S + Bc - 1) // Bc
    num_waves    = base["num_waves"]

    # Two-regime SHFL latency: low occupancy (few active SMs) vs. high occupancy.
    # Back-solved from ncu gpu__time_duration: low-occ (~9156 cy/tile) when
    # num_q_blocks < NUM_SMS (not all SMs busy); high-occ (~15054 cy/tile) otherwise.
    shfl_lat  = 111 if num_q_blocks < NUM_SMS else 185
    cycles_kv = Bc * (2 * FMA_LAT + N_SHFL * shfl_lat) + N_SYNCS * SYNC_LAT
    t_serial  = num_waves * num_kv_tiles * cycles_kv / SM_CLOCK

    t_pred = max(base["pred_ms"] * 1e-3, t_serial)

    if t_pred == t_serial:          bound = "serial"
    else:                           bound = base["bound"]

    return {
        **base,
        "pred_ms":    t_pred * 1e3,
        "bound":      bound,
        "flops/s":    base["flops"] / t_pred,
        "t_serial_ms": t_serial * 1e3,
        "cycles_kv":  cycles_kv,
        "shfl_lat":   shfl_lat,
        "num_waves":  num_waves,
    }


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    # "v1_simple":     model_v1,
    # "roofline":     model_roofline,
    # "roofline_l2": model_roofline_l2,
    "roofline_l2_serial": model_roofline_l2_serial,
    # "v1_loop":     model_v1_5,
    # "v2_l2_aware":   model_v2,
    # "v3_sequential": model_v3,
}


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def _mape(pred, meas):
    return abs(pred - meas) / meas * 100.0 if meas is not None else None


def compare(sizes):
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
            r = fn(s)
            mape_str  = f"{_mape(r['pred_ms'], m):>7.1f}%" if m is not None else f"{'':>8}"
            tflops_str = f"{r['flops/s'] / 1e12:>7.2f}T"
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

def load_ncu_cache():
    if not os.path.exists(NCU_CACHE_FILE):
        return {}
    with open(NCU_CACHE_FILE) as f:
        return json.load(f)


def _ncu_key(S, Br=Br, Bc=Bc):
    return f"S{S}_Br{Br}_Bc{Bc}"


def compare_ncu(sizes, model_fn=None):
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
            compare_ncu(sizes, model_fn=fn)
        return
    cache = load_ncu_cache()

    col = 12
    hdr_line = f"{'S':>8}  {'quantity':<24}  {'model':>{col}}  {'ncu':>{col}}  {'error%':>8}"
    print(f"\n=== NCU metric comparison  (model: {model_fn.__name__}) ===")
    print(hdr_line)
    print("-" * len(hdr_line))

    for S in sizes:
        key   = _ncu_key(S, Br, Bc)
        entry = cache.get(key)
        r     = model_fn(S)

        def nget(*keys):
            if not entry:
                return None
            return sum(entry.get(k, 0) for k in keys)

        # Rows: (label, model_value, ncu_value, unit)
        # unit: "MB" | "GFLOPs" | "ms" | "pct"
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
            # ---- compute / timing ----
            ("FLOPs (GFLOPs)",
             r["flops"] / 1e9,
             (nget("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum") * 2 / 1e9
              if entry else None),
             "GFLOPs"),
            ("runtime (ms)",
             r["pred_ms"],
             (nget("gpu__time_duration.sum") / 1e6 if entry else None),
             "ms"),
            ("occupancy (%)",
             None,
             (entry.get("sm__warps_active.avg.pct_of_peak_sustained_active")
              if entry else None),
             "pct"),
        ]

        for i, (label, pred, meas, unit) in enumerate(rows):
            s_str = f"{S:,}" if i == 0 else ""

            if unit == "MB":
                def fmt(v): return f"{v/1e6:{col}.2f}"
            elif unit == "GFLOPs":
                def fmt(v): return f"{v:{col}.1f}"
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
    sizes = list(map(int, sys.argv[1:])) if len(sys.argv) > 1 else DEFAULT_SIZES
    compare(sizes)
    compare_ncu(sizes)   # iterates over all MODELS when no model_fn given
