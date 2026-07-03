# Phase 1 — Speed Optimization Log

**ViT-B/16 @224, 6× RTX 5090 (PCIe, no NVLink), bf16, accelerate DDP.**
**Result: 4,955 → 5,803 img/s real data (+17%); 7,454 synthetic ceiling. Data pipeline is now the wall.**

## The journey
| milestone | img/s | note |
|---|---|---|
| original eager bs256 | 4,955 | comm-bound (DDP all-reduce over PCIe) |
| + compile (naive, compile-after-prepare) | 3,715 | WORSE — 396 graph breaks compiling DDP's forward |
| + compile-BEFORE-prepare `DDP(compile)` | 4,341 | 0 graph breaks, but comm still exposed |
| + **bf16 gradient comm hook** | 5,892 | the unlock — halves the 344MB all-reduce |
| + **bs384** (amortize comm; bs512 OOMs) | **7,454 synthetic** | hits the ViT-B/16 ceiling |
| same recipe, REAL data | **5,803** | dataloader now the bottleneck |

## What actually mattered (the fixes, in order of impact)
1. **`accelerate launch --gpu_ids 0,1,2,3,5,6`** (NOT CUDA_VISIBLE_DEVICES) — fixes the 3090-rank crash. *Mandatory.*
2. **bf16 gradient comm hook** — `model.register_comm_hook(None, default_hooks.bf16_compress_hook)` after prepare. Halves the PCIe all-reduce. **The single biggest lever** (these are PCIe cards, no NVLink → comm is the cap).
3. **compile BEFORE prepare** — `DDP(torch.compile(model))`, NOT `compile(DDP(...))`. The latter shatters into ~400 graph breaks on DDP's Python forward.
4. **bs384** — biggest batch that fits 32GB; amortizes the all-reduce.
5. drop channels_last (hurts), kill per-step `loss.item()` syncs.

## ROOT CAUSE (corrected): NUMA cross-socket all-reduce
`nvidia-smi topo -m`: the six 5090s span **two CPU sockets** — GPU0-3 on NUMA 0, GPU5,6 on NUMA 1,
linked by **SYS** (host bridge). The all-reduce ring crosses that boundary → ~3 GB/s (host-routed),
NOT PCIe-P2P's ~25 GB/s. *That's* the wall — and it's structural (no 6-5090 set avoids both sockets).
The **bf16 comm hook fixes it by halving the bytes over that slow link** — the dominant lever.

## Diagnostics that ruled things out
- **Not input-bound (originally):** synthetic 4,957 ≡ real 4,955 → dataloader innocent *until* GPU path fixed.
- **Not math attention:** fused/flash SDPA on (18× faster than math).
- **NOT the torch version:** 2.7 vs 2.11 identical (untested: compiled-autograd, the real 2.10 lever — but moot, see below).
- **NOT compile-order / DDPOptimizer overlap:** overlap *alone* (compile-after, no hook) = 4,812 ≈ eager. The cross-socket comm is too slow to hide behind the ~120ms backward. compile-before(7,454) ≈ compile-after-best(7,366) once the hook is on.
- **EMA breaks under compile** (`_orig_mod`) — use `dynamo_backend="inductor"` or rebuild EMA from `_orig_mod`.
- **GPU path (7,454) now exceeds data (5,803)** → further GPU-side work (compiled-autograd etc.) is moot; the dataloader is the only remaining lever.

## To reach the 7,454 ceiling (data pipeline is now the bottleneck)
- **uint8 batches + GPU-side Normalize** — 4× less H2D, AND a uint8 bs384 batch (58MB) pins under the box's 147MB limit (re-enabling pin_memory). Best cheap win.
- pre-resize ImageNet to ~256px; DALI (nvJPEG GPU decode); FFCV.
