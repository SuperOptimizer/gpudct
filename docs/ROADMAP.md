# gpudct — Implementation roadmap

Ordering principle: **a correct, slow, complete codec before a fast, incomplete one.**
The scalar CPU reference is the oracle everything else is validated against, so it comes
first and it never gets deleted.

---

## Status as of the first implementation pass

M0-M5 are implemented and tested; M6 is partly done. What exists and works:

- Scalar reference codec end to end, brick + chunk container, all seven dtypes,
  arbitrary dimensions (`src/core/`).
- Float Lee-factorized 16-point DCT, validated against a naive `double` reference.
- Static-model rANS with division-free encoding, hierarchical 64-bit significance
  masks, bit-length-coded magnitudes.
- Entropy tables trained on real Vesuvius scroll data (`tools/train_tables`,
  `src/core/tables_v2.inc`). Worth +7% to +36% ratio over the analytic priors at
  identical quality, the larger gains at the higher-quality end.
- Raw-brick fallback, so incompressible input never expands.
- Bounded-error correction layer: absolute, relative, and percentile modes. The
  percentile mode costs ~2.3x less than the equivalent hard bound.
- Deblocking filter (+2.5 dB on smooth content at low rate).
- Full metric suite and RD bench harness; 63 tests across 6 binaries.
- Lane-parallel SIMD transform, bit-identical to scalar.

CUDA encode and decode are implemented and validated (README has the numbers).
Encode is byte-identical to the CPU encoder; decode agrees within 1 LSB.

Since done: RDO (`--effort high`), per-brick clustered entropy tables, and the
3D causal neighbour-sum level context. DC-plane prediction is retired rather
than pending -- DC carries 0.1-0.5% of coded bits, so there is nothing there to
predict. The quantizer matrix has been swept and is a PSNR/SSIM frontier rather
than a missed win (docs/QUALITY.md).

Not yet done: Zarr integration, baselines against ZFP/SZ3, CI of any kind,
and CUDA stream overlap -- every device transfer is still a synchronous
`cudaMemcpy` on the default stream, so no batch's host preparation overlaps any
other batch's kernels. 3ddct is measured at -16.97% BD-rate at `--effort high`
(see docs/QUALITY.md).

Measured on a real full-resolution 128^3 brick from `PHerc0332` (one brick = one
Zarr chunk of the open dataset), balanced profile:

| quality | ratio | PSNR | 3D-SSIM | p99 err | max err |
|---|---|---|---|---|---|
| 0.125 | 51.4x | 31.1 dB | 0.925 |  20 | 57 |
| 0.25  | 31.1x | 34.7 dB | 0.963 |  13 | 36 |
| 0.5   | 19.5x | 38.4 dB | 0.983 |   8 | 25 |
| 1.0   | 12.5x | 42.1 dB | 0.992 |   5 | 16 |
| 2.0   |  8.1x | 45.6 dB | 0.996 |   3 | 12 |

Sampling resolution matters more than anything else here. The same codec at
quality 0.5 gets 19.5x on a full-resolution brick and 5.7x on a level-3
(8x downsampled) brick of the same scroll: downsampling has already removed the
redundancy the transform feeds on. Benchmark on the resolution you intend to
store.

### M0 — Scaffold
CMake (LLVM 23 / libc++ / lld only, as chosen), CI matrix (Linux + Windows, clang-23,
Debug/Release/ASan/UBSan), clang-format/clang-tidy, test framework, `tools/fetch_corpus`,
the synthetic volume generators from QUALITY.md §3.

*Done when:* `cmake --build . && ctest` is green with a trivial test on both platforms.

### M1 — Scalar reference codec, end to end
`tools/gen_transform` (integer 16-point DCT-II + validation), forward/inverse transform,
level shift and per-chunk exponent, dead-zone quantizer, the significance hierarchy, and
a **placeholder byte-oriented entropy coder** (not rANS yet — just enough to close the
loop). Brick/chunk container, header parsing, padding/edge handling for non-multiple
dimensions. `gpudct compress` / `decompress` work on real data.

*Done when:* round-trip on the full corpus, transform round-trip within tolerance of the
`double` reference, dynamic-range interval proof passes, golden bitstreams established.

### M2 — Measurement before optimization
`tools/gpudct metrics` with the full QUALITY.md §1 metric set, the `bench/` RD sweep
harness, and the baseline integrations (3ddct, ZFP, SZ3, blosc2+zstd). JSON output,
plots.

*Done when:* we can produce an RD curve of gpudct-M1 against all baselines. Expect to
lose to 3ddct here — that's the point, it establishes the gap to close and tells us
whether the quant matrix and dead-zone parameters are in the right neighbourhood before
any of it is hard to change.

### M3 — rANS + static model training
Replace the placeholder coder: 32-bit/16-bit-renorm rANS, 12-bit probabilities, 32-way
interleave, `P` streams per brick, the context definitions from DESIGN.md §3.4.
`tools/train_tables` produces the shipped static tables from the corpus; header-embedded
table override. Exhaustive rANS unit tests.

*Done when:* ratio is within ~5 % of 3ddct at equal 3D-SSIM. If the gap is much larger,
the context modelling needs work before proceeding — the parallel-decode constraint costs
something, but not that much.

### M4 — Quality and ratio work
Now that measurement and entropy coding are real, tune the things that move the curve:
quant matrix profiles (`archival`/`balanced`/`viewing`), RDO level decision, the DC plane
and its recursive nesting, the deblocking filter, `--denoise` pre-filter. Each change
justified by an RD curve, not by intuition.

*Done when:* gpudct beats 3ddct's ratio at equal 3D-SSIM on the scroll corpus, blockiness
index < 1.1, and the power-spectrum ratio stays flat to ~0.7 Nyquist at `archival`.

### M5 — Bounded error and percentile modes
The correction layer, `--max-error` / `--rel-error` / percentile-bounded mode. Exhaustive
bound-invariant testing. Comparison against ZFP/SZ3 in *their* natural regime
(fixed-accuracy), which is the fair way to benchmark this feature.

*Done when:* declared bounds hold on every corpus volume and every synthetic adversarial
case, and the RD curve in fixed-accuracy mode is competitive with ZFP.

### M6 — CPU SIMD
Per-ISA kernels (AVX2, AVX-512, NEON; SVE2 if hardware is available), the `vec<T,N>`
wrapper over `std::simd`/intrinsics, runtime dispatch, the 16×16 in-register transpose,
SIMD interleaved rANS, brick-parallel threading.

*Done when:* ≥ 1 GB/s/core decode, near-linear thread scaling, and **byte-identical
output to the scalar reference** on the whole corpus.

### M7 — CUDA
K1 (warp-per-stream entropy decode → sparse levels) and K2 (block-per-chunk dequant +
IDCT), then the encode mirror. Batched multi-brick launches, CUDA streams, pinned/async
H2D, optional GPUDirect Storage. `clang++ -x cuda` build with an nvcc-compatible
fallback. `compute-sanitizer` clean.

*Done when:* byte-identical to the CPU reference on the whole corpus, ≥ 50× single-core
CPU throughput, and profiling confirms the memory-bound regime the design predicts.

### M8 — Integration and tuning
Zarr v3 codec plugin (one Zarr chunk = one brick), a C ABI + Python binding, a Fenix-side
reader, and -- if wanted -- a band-ordered profile with an LOD decode path (see
DESIGN.md 3.7, removed rather than left half-built). Then the
optimization backlog: K1/K2 fusion if profiling justifies it, the tensor-core transform
experiment, occupancy and shared-memory tuning, `P` auto-selection.

*Done when:* a real scroll volume can be compressed, served, and randomly read at LOD by
a viewer, and the M8 success criterion in QUALITY.md §2 is met.

---

## Sequencing rationale

Three orderings here are deliberate and worth defending:

**Metrics (M2) before entropy coding (M3).** Ratio work without an RD harness is
guesswork. Building the measurement first means every subsequent change is justified by
a curve, and it forces the baselines to exist early — which is also when it's cheapest to
discover that, say, a wavelet beats us on some dataset and the design needs revisiting.

**Ratio (M4) before speed (M6/M7).** The bitstream format is what's expensive to change
once golden files and real archives exist. Speed work touches implementations, not the
format. So the format-affecting decisions get made and validated while they're still
cheap, and the SIMD/CUDA work targets a frozen spec.

**CPU SIMD (M6) before CUDA (M7).** The GPU kernels need an oracle to be validated
against, and a fast CPU path makes the whole test loop faster. Doing CUDA first means
debugging a warp-synchronous entropy decoder against a reference that takes minutes per
corpus run.
