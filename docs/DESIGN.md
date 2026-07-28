# gpudct — Design

A lossy (and bounded-error) codec for massive 3D scalar volumes, designed so that the
*same bitstream* decodes at full speed on a GPU and on a SIMD CPU, bit-exactly.

Target workload: Vesuvius Challenge scroll CT (multi-TB `u8`/`u16` volumes), plus
microCT / MRI / simulation fields (`f32`).

Prior art we borrow ideas from, but are not compatible with:
- [SuperOptimizer/3ddct](https://github.com/SuperOptimizer/3ddct) — 16³ float DCT-II,
  dead-zone quantizer, adaptive binary range coder, per-chunk range normalization,
  optional exact-correction pass, deblocking filters.
- [SuperOptimizer/fenix](https://github.com/SuperOptimizer/fenix) — the consumer side:
  `.fxvol`-style archives, 64³ chunk storage, whole-scroll streaming.

---

## 1. Non-negotiable requirements

These drive every decision below.

| # | Requirement | Consequence |
|---|---|---|
| R1 | Fully parallel decode — no serial dependency chains across the volume | Static-model rANS, not an adaptive binary range coder |
| R2 | Random access at storage granularity | Independently decodable units aligned to what you'd actually fetch from S3 |
| R3 | Reproducible GPU ≈ CPU output, within a stated tolerance | Float transform for speed; cross-backend equality is tolerance-based, not bit-exact (§3.2) |
| R4 | Guaranteed error bounds (max, and percentile) | Explicit correction layer, not "quality knob and hope" |
| R5 | High ratio (target 20–60× visually lossless on scroll `u8`) | Entropy-coding overhead must be amortized over ≫ one chunk |
| R6 | Cheap low-resolution decode for viewers | Frequency-band-ordered bitstream (optional profile) |
| R7 | No 64-bit numeric types | Supported dtypes: `u8 s8 u16 s16 u32 s32 f32`. No `f64`/`u64`/`s64`. All codec math is `f32` or `i32`. (File *offsets* remain 64-bit — a multi-TB volume leaves no choice.) |

---

## 2. Two-level structure: chunk and brick

This is the central structural decision, and it falls directly out of R5.

```
brick  = 128³ voxels = 8×8×8 chunks   ← entropy-coding unit, I/O unit, random-access unit
chunk  =  16³ voxels = 4096 coeffs    ← transform + quantization unit
```

### Why the transform unit is 16³
- 4096 voxels = 16 KB as `f32`, fits comfortably in CUDA shared memory.
- Separable 3D transform = 3 passes × 256 length-16 1D transforms. 256 threads = one
  CUDA block, one pencil per thread. Clean mapping, no divergence.
- 16 lanes = exactly one AVX-512 `zmm` of `f32`/`i32`, two AVX2 `ymm`, four NEON `q`.
  The CPU path vectorizes *across* pencils with zero shuffles for the z and y passes.
- Matches 3ddct, so cross-comparison of ratio/quality is apples-to-apples.

### Why the entropy unit is 128³ (the part that isn't obvious)

An interleaved rANS coder must flush its interleaved states at the end of a stream.
With `S` states of 32 bits, that is `4S` bytes of pure overhead per stream.

Budget at a 20× ratio on `u8` data:

| Entropy unit | Raw bytes | Compressed | 32-state flush | Overhead |
|---|---|---|---|---|
| chunk 16³ | 4 KB | 205 B | 128 B | **62 %** ✗ |
| 2×2×2 chunks (32³) | 32 KB | 1.6 KB | 128 B | 7.8 % ✗ |
| brick 128³ | 2 MB | 100 KB | 128 B | **0.13 %** ✓ |

So: chunks are the transform unit, but they do **not** carry their own entropy stream.
A brick carries `P` interleaved rANS streams (default `P = 4`, each 32-way interleaved,
one warp per stream), total flush overhead `P × 128 B` ≈ 0.5 % of a brick. Chunk `k` is
assigned to stream `k mod P`, so the streams stay balanced.

Consequences, stated plainly:
- **Random access granularity is 128³, not 16³.** This is fine — and arguably correct.
  It is the granularity you fetch from object storage anyway, it matches Fenix's 64³
  storage philosophy, and it maps 1:1 onto a Zarr chunk.
- **Intra-brick parallelism is `P` warps** for the entropy stage. For bulk decode this
  is irrelevant (a 1 TB volume has ~500 K bricks — parallelism comes from bricks). For
  a single-brick viewer fetch, `P = 4` warps is enough to hide latency. `P` is a
  per-archive header field, so a "low-latency" archive can raise it and pay ~2 %.

---

## 3. Pipeline

```
ENCODE                                         DECODE
  volume (u8/u16/i16/f32)                        brick bytes (from disk/S3/GDS)
    ↓ optional DC-plane extraction                 ↓ kernel 1: rANS → sparse (index, level)
    ↓ level shift / range normalize                ↓ kernel 2: scatter → dequant → integer IDCT
    ↓ forward integer DCT-II (separable ×3)        ↓ clamp, level unshift, store
    ↓ frequency-weighted dead-zone quant           ↓ optional deblock (needs neighbours)
    ↓ optional RDO level decision                  ↓
    ↓ optional bounded-error correction pass       volume
    ↓ significance hierarchy + rANS (static tables)
  brick bytes
```

### 3.1 Level shift and per-chunk normalization

Per **chunk**, store an 8-bit exponent `e` and scale coefficients by `2^-e`, chosen from
the chunk's measured value range. This is 3ddct's "per-chunk value range" idea, but
restricted to powers of two: a power-of-two scale is exact in `f32` (it only touches the
exponent field), so applying and undoing it introduces zero rounding error and costs one
byte per chunk instead of four. Air/background chunks collapse to a single all-zero flag.

`f32` input is handled by an up-front affine map to a 16-bit fixed-point domain, with the
scale/offset stored per brick; residual float precision beyond that is recovered (if
requested) by the correction layer of §3.5.

### 3.2 Float DCT-II, 16-point

**The transform is `f32` throughout** — forward and inverse, on every backend. This is
the fastest option on both GPU (FMA pipe, no integer-shift chains) and modern CPUs, and
it costs nothing in coding gain since there is no integer-approximation error.

Factorization: 16-point DCT-II decomposes (Lee) into an 8-point DCT-II on the even part
plus an 8-point DCT-IV on the odd part. ~31 multiplies / 81 adds vs 256 multiplies for
the naive matrix product — an 8× reduction. On GPU this is the difference between
"memory bound" and "arguably compute bound", so we take it. `tools/gen_transform`
emits the constants and validates orthogonality, coding gain against the ideal DCT on an
AR(1) ρ=0.95 source, and round-trip error against a `double` reference.

**Consequence, stated plainly:** GPU and CPU output are *not* byte-identical. Float
addition is not associative, the SIMD and CUDA kernels use different operation orders,
and FMA contraction differs by backend, so reconstructed values drift by a few ULP —
which occasionally flips the rounding of an output voxel. Therefore:

- **Cross-backend equivalence is tolerance-based** (QUALITY.md §4.2): decoded outputs
  must agree within ±1 LSB of the output dtype, with the fraction of differing voxels
  below 1e-4. Any voxel differing by ≥2 LSB is a bug.
- **Bitstreams produced by different encoder backends may differ**, since a coefficient
  sitting exactly on a quantizer boundary can land either side of it. Both are valid and
  both decode within tolerance. Golden-file tests therefore pin the *scalar CPU
  encoder's* output specifically.
- The one thing that *is* bit-exact everywhere is the entropy layer: rANS is pure
  integer arithmetic, so a given bitstream always yields exactly the same quantized
  levels on every backend. All reconstruction drift is confined to dequant + IDCT.

To keep the drift small and predictable, the reference scalar path compiles with
`-ffp-contract=off` and no fast-math; the optimized paths may contract, and the
tolerance test is what holds them honest.

**Stretch experiment, not baseline:** a 16³ chunk's per-pass transform is literally an
`m16n16k16` tensor-core tile. TF32's 10-bit mantissa is too imprecise for the quality
targets, but a split-high/low (Ozaki-style) MMA path could be worth benchmarking once
the scalar path is correct. Deferred to M8.

### 3.3 Quantization

Dead-zone quantizer, float:

```
level = sign(c) · floor(|c| · rq[u,v,w] + offset)      // rq = 1/q, precomputed
value = level · q[u,v,w]                               // dequant, exact-ish in f32
```

- `rq` is the precomputed reciprocal of the quant matrix, so the inner loop is one
  multiply and one `floor` — no division, and it vectorizes and runs on the GPU FMA pipe.
- `offset` implements the dead zone; default `0.34` (H.264 intra-like — a
  widened dead zone buys ~5 % rate for negligible distortion because the zeroed
  coefficients were noise).
- The quant matrix is parametric and isotropic: `q(u,v,w) = q_base · (1 + a·r^b)`
  with `r = ‖(u,v,w)‖/‖(15,15,15)‖`. Unlike image codecs there's no human visual
  system to model here and the data is isotropic, so radial weighting is the honest
  choice. `(a, b)` come from a **profile**:
  - `archival` — near-flat (`a` small): preserves high-frequency fiber texture, best
    for downstream ML / ink detection. Lower ratio.
  - `balanced` — default.
  - `viewing` — aggressive HF rolloff, best ratio, for human inspection and previews.
  A custom 4096-entry matrix can be embedded in the archive header instead.

Optional **RDO level decision** (encoder-only, doesn't affect the format): for each
coefficient, compare `J = D + λR` for the quantized level, the level−1, and zero, using
the static rANS tables to estimate `R` exactly. Fully parallel (no trellis, no
dependency between coefficients), so it runs on-GPU during encode. Typically 5–9 %
rate reduction at fixed PSNR. `--effort` selects it.

### 3.4 Significance hierarchy and entropy coding

4096 coefficients per chunk, overwhelmingly zero at useful rates. The representation
must be decodable by a warp in lockstep, so: **hierarchical 64-bit masks**, not
zigzag+run-length (which is inherently serial).

```
L0   1 bit    chunk has any nonzero AC coefficient
L1   64 bits  which of the 64 sub-blocks (4×4×4 each) are significant
L2   64 bits  per significant sub-block: which of its 64 coefficients are nonzero
VAL  per nonzero: |level|−1 as a symbol (0..14, escape → Exp-Golomb), then a raw sign bit
```

Decoding masks is `popcount` + warp prefix-sum — the operation GPUs are best at. The
coefficient positions for a sub-block fall out of `__popc` on the mask below the lane's
bit, giving each lane its own coefficient with no divergence.

**Contexts are static, never adaptive** (R1). This is the ratio sacrifice we make for
parallelism, and we claw most of it back by conditioning on a lot:
- L1 mask bits: context = sub-block's radial frequency band (4 bands).
- L2 mask bits: context = band × number of already-significant neighbours in the mask
  (0,1,2,3+) — computed from bits below the current one, so it's causal within a lane's
  own scan and still parallel across lanes.
- Level symbols: context = band × min(3, magnitude of previously decoded neighbour).
- Sign bits: bypass (raw), they're incompressible.

Tables are trained offline (§`tools/train_tables`) on a corpus of scroll data and
shipped in the binary. An archive may override any table in its header (costs ~2–8 KB,
amortized over the whole archive) — this is how we handle a genuinely different modality
without a code change.

**rANS details:** 32-bit state, **8-bit** renormalization, 12-bit probability precision
(`total = 4096`, so the symbol lookup is a 4096-entry table — one shared-memory lookup,
no search). Encode in reverse, decode forward. 32 interleaved states per stream.

**Correction to an earlier version of this document.** It claimed a warp would
decode one stream in lockstep, one interleaved state per lane, in the
DietGPU/nvCOMP-ANS style. That is not possible for this format, and the reason is
worth stating because it constrains anything built on top:

> Lockstep lane decoding requires knowing which model each symbol uses *before*
> decoding it. DietGPU can do this because it codes fixed-size items under a
> single model. Our parse is data-dependent — how many level symbols follow a
> sub-block is determined by the mask that precedes them, and a level's context
> depends on the magnitude decoded just before it. Lane *i* cannot pick its model
> until lanes 0..*i*-1 have finished.

Two ways out, and we take the second:

1. Split the stream by model class (masks, levels, signs each in their own
   sub-stream) so every symbol's model follows from its position, and drop the
   neighbour-magnitude context. This restores lockstep decoding but costs ratio
   and multiplies the per-brick flush overhead by the number of sub-streams.
2. **Decode one stream per thread, serially, and take parallelism from bricks and
   streams instead.** The 32 interleaved states live in that thread's registers,
   where they still buy instruction-level parallelism. A 1 TB volume has ~500 K
   bricks; at the default `P = 4` that is two million independent threads, which
   saturates any GPU several times over.

The cost of (2) is that total device parallelism is `bricks × P`, so `P` is no
longer just a latency knob — it sets whether the GPU has any work at all. This
was measured, on a 512³ volume (64 bricks) decoded on an RTX 5080:

| `P` | GPU threads | ratio | GPU decode | CPU decode (16 threads) |
|---|---|---|---|---|
| 4  |   256 | 19.47× | 121 MB/s | 1148 MB/s |
| 16 | 1024  | 19.22× | 651 MB/s | 1239 MB/s |
| 32 | 2048  | 18.90× | 709 MB/s | 1184 MB/s |

`P = 4` starves the device completely. Raising it to 16 is a 5× GPU speedup for
1.3 % of ratio, which is the trade almost any GPU-targeted archive should take;
the default stays at 4 because it is the right choice for CPU decode and for
storage.

Two implementation notes that cost 7× between them, and would cost the same
again in any reimplementation:

- The 32 interleaved states must live in **shared memory**, not in a thread-local
  array. An array indexed by a runtime value cannot stay in registers, so
  `state[next]` became a local-memory round trip on every symbol.
- The grid must be flattened to one thread per `(brick, stream)` pair. A 2-D grid
  keyed on the stream index wastes most of every warp whenever `P` is smaller
  than the block width, which is the common case.

Two further caveats worth stating before anyone quotes the GPU number: the table
above is warm, and the first decode in a process pays one to two seconds of CUDA
context creation and kernel JIT (~49 MB/s all-in for a 128 MB volume). Run-to-run
variance on the warm path is also wide, roughly 560–710 MB/s.

**Where this leaves the GPU path.** It does not yet beat a 16-core CPU on this
workload, and the reason is structural rather than a missing optimization: one
serial parse per stream is simply less parallelism than the device wants. Option
(1) above — per-model sub-streams enabling true lane-parallel decode — is the
real fix, and it is a format change that should be made before any archives ship
if GPU decode is the priority. The honest summary today is that the GPU is a
useful offload, not a speedup.

Byte rather than word renormalization, for a reason worth recording: the encoder
replaces its per-symbol integer division with a 31-bit reciprocal multiply, and that
reciprocal only reproduces `x / freq` exactly while the state stays below 2^31. Byte
renormalization keeps the state in `[2^23, 2^31)`; 16-bit renormalization lets it reach
2^32, where the reciprocal is wrong for roughly one symbol in a million — often enough
to corrupt a real archive, rarely enough to pass a small test. The division was
measured to be the single largest cost in the codec, larger than the transform by an
order of magnitude, so removing it correctly matters more than it looks.

### 3.5 Bounded error and percentile targets

This is what separates a scientific codec from a media codec, and it maps directly onto
the metrics you named (p90/p95/p99/max).

After the lossy layer, the encoder decodes its own output and measures the per-voxel
error field. It then emits an optional **correction layer**:

- `--max-error τ` (absolute, in input units) or `--rel-error ε` (relative to the chunk's
  measured range, 3ddct-style): every voxel with `|err| > τ` gets an entry. Guarantees
  p100 ≤ τ.
- `--p99 τ99 --max-error τmax`: **percentile-bounded mode**. Correct only enough voxels,
  worst-first, to satisfy each percentile constraint. This is much cheaper than a hard
  max bound (the tail is a tiny fraction of the volume) and is usually what people
  actually want.

The correction layer is coded as a per-chunk significance mask over voxel positions plus
rANS-coded deltas, in its own stream so it can be *skipped entirely* by a decoder that
only wants the fast lossy reconstruction. Layered, not baked in.

### 3.6 The DC plane (optional, and a free LOD)

The chunk DC coefficients of a brick form an 8³ array; across the volume they form an
exact 16× downsampled version of it. Storing that plane separately, and coding chunk DCs
as residuals against it:
- improves ratio (scroll data has strong inter-chunk DC correlation),
- preserves chunk independence (no DPCM chain — the predictor is a side array),
- **hands a viewer a free level-4 mipmap** it can decode without touching the residuals.

The DC plane is itself a volume, so it is stored as a nested gpudct archive. Recursion
bottoms out when a level fits in one brick.

### 3.7 Frequency-band progressive profile (removed)

This was specified and never implemented, and the half-built state was worse than
either alternative: `EncodeOptions::progressive` set a header flag that nothing
acted on, and `inspect` reported it back, so an archive could claim a property it
did not have. A container that can lie about its own contents is a defect, not a
missing feature. Both the flag and the option are gone; header flag bit 1 is left
permanently unused rather than reassigned, so no archive written earlier can be
misread by a later decoder.

The underlying idea remains sound and is recorded here in case it is wanted: the
DCT's low-frequency corner *is* a downsampled signal, so decoding only the
`2³ / 4³ / 8³` corner of each chunk yields an exact 8× / 4× / 2× downsampled
volume. Ordering the bitstream band-major with per-band offsets in the brick
header would give a viewer progressive refinement and cheap LODs, at a cost of
`bands × P` rANS flushes instead of `P` (≈ 2 % at 4 bands, `P=4`) -- which is why
it would have to be optional. Implementing it means doing the band ordering, the
per-band offsets, and an LOD decode path together, and re-introducing a flag only
once all three exist.

### 3.8 Deblocking

Hard quantization at 16³ produces visible seams, which hurt SSIM and, more importantly,
confuse downstream segmentation. Two options were considered:

- Lapped/overlapped transform — no seams by construction, but needs neighbouring chunks
  *at decode*, breaking independent decode. Rejected.
- **Decoder-side deblocking filter** (chosen) — an H.264-style conditional filter across
  chunk faces, strength derived from the quantization step and the local gradient. It is
  applied when neighbours are available (bulk decode: always, since we decode whole
  bricks; brick faces need a 2-voxel halo from the adjacent brick). It is *not* in the
  normative reconstruction path, so a decoder may skip it and still be bit-exact with
  respect to the format. Filter strength is signalled in the header as a hint.

An encoder-side pre-filter (3ddct's `predeblock`) is also worth having as an
`--denoise` option: scroll CT is noisy, and mild pre-filtering improves both ratio and
apparent quality. Off by default — it is not lossless-in-intent and shouldn't be silent.

---

## 4. GPU implementation (CUDA first)

### Kernel split

Not fused, deliberately, at least initially:

**K1 — entropy decode.** Grid = `bricks × P`, block = 32 threads (one warp = one stream).
Decodes masks + levels, writes **sparse** `(u16 chunk-local index, i16 level)` pairs to a
global scratch buffer plus a per-chunk count. Sparse, not dense: at 20× a chunk has
~100–300 nonzeros (4 B each ≈ 1.2 KB) versus 8 KB dense — a 7× traffic saving.

**K2 — inverse transform.** Grid = one block per chunk, block = 256 threads, 16 KB
shared memory. Zero shared memory, scatter the sparse levels, dequantize, three IDCT
passes (each thread owns one pencil), clamp and store. With 100 KB opt-in shared memory
that's 6 blocks/SM = 1536 threads/SM ≈ 75 % occupancy.

Traffic accounting per 16³ chunk of `u16` output: write 1.2 KB sparse + read 1.2 KB +
write 8 KB = 10.4 KB for 8 KB of output. At ~1 TB/s that is ~770 GB/s of decoded volume
on a 4090-class part — far beyond any storage system that would feed it, so fusion is a
later optimization, not a launch requirement. It gets revisited in M8 with real profiles.

Encode mirrors this: K1' = forward transform + quantize + (optional) RDO → sparse levels;
K2' = rANS encode, which runs *backwards* over the symbol sequence (rANS is LIFO), which
is why the sparse intermediate exists in the first place.

### Toolchain note

nvcc tops out at C++20. Since we're LLVM-23-only anyway, **compile CUDA with
`clang++ -x cuda`**, which unifies the toolchain and allows C++26 in host code adjacent
to device code. Risk: clang's CUDA support occasionally lags on the newest
architectures. Mitigation: the CUDA sources stay C++20-clean so an nvcc fallback build
remains possible; CI builds both.

### Other backends

Deferred, but the design is backend-neutral: the entropy format assumes only
`popcount`, `ballot`/subgroup-prefix-sum, and 32-bit integer math — all available in
Vulkan/SPIR-V (`GL_KHR_shader_subgroup_*`), HIP, and WebGPU (which lacks subgroups on
some targets; there a 1-lane-per-stream fallback with larger `P` works). A WebGPU decoder
is genuinely valuable for browser-side scroll viewers and is the most likely second
backend.

---

## 5. CPU implementation

Same bitstream, same bit-exact results.

- **Entropy**: 8- or 16-way interleaved rANS decoded with AVX2/AVX-512 gather for the
  symbol lookup (Ryg's `rans_word_sse41` structure, widened). The format's 32 states
  per stream are simply processed 8 or 16 at a time.
- **Transform**: vectorize *across* pencils. With `x` contiguous, the z-pass and y-pass
  are pure vertical vector ops on 16 contiguous `i32` — zero shuffles. The x-pass needs a
  16×16 in-register transpose (the standard AVX-512 `unpack`/`permute` ladder, or four
  4×4 NEON `zip` stages).
- **Dispatch**: runtime CPU detection into `scalar / sse4 / avx2 / avx512 / neon /
  sve2` variants, one translation unit per ISA, function-pointer table resolved once.
  C++26's `std::simd` is used where libc++ 23 supports it, behind our own `vec<T,N>`
  wrapper so we can drop to intrinsics per-ISA without touching call sites.
- **Threading**: parallel over bricks. Lock-free, no shared mutable state, so it scales
  linearly. Target ≥ 1 GB/s/core decode (3ddct reports 121–657 MB/s/core with an
  adaptive coder; static rANS + integer transform should beat that comfortably).

---

## 6. Repository layout

```
include/gpudct/         public C++26 API (headers, not modules — CUDA interop)
  gpudct.hpp              encode/decode entry points, std::mdspan volume views
  types.hpp               dtype, profile, quality, error-bound structs
  gpudct.h                stable C ABI (for Python/Rust/Fenix consumers)
src/
  core/                   backend-independent: format, headers, quant, tables, models
  cpu/                    scalar reference + per-ISA SIMD kernels + dispatch
  cuda/                   K1/K2 kernels, stream/batching, GDS path
  api/                    C ABI shim, error handling
tools/
  gpudct/                 CLI: compress, decompress, bench, metrics, inspect
  train_tables/           offline static-model training
  gen_transform/          derives + validates the integer DCT matrix
tests/                    unit, property/fuzz, golden bitstreams, conformance
bench/                    rate-distortion sweeps, throughput, baseline comparisons
docs/                     DESIGN.md, FORMAT.md, QUALITY.md, ROADMAP.md
```

## 7. Design decisions we consciously rejected

| Rejected | Why |
|---|---|
| Adaptive binary range coder (3ddct's choice) | ~4–8 % better ratio, but strictly serial per chunk → one GPU thread per chunk, wasting the SM. Fails R1. |
| Per-chunk independent entropy streams | 60 % state-flush overhead at target ratios. Fails R5. |
| Integer/fixed-point DCT | Would give byte-identical GPU/CPU output, but costs coding gain, needs a dynamic-range proof, and is slower on the FMA-heavy hardware we target. We take `f32` and a tolerance-based equivalence test instead (§3.2). |
| `f64` anywhere, and 64-bit dtypes | Not needed for CT/MRI/simulation volumes at these rates, and `f64` is 1/64th rate on consumer NVIDIA parts. `f32`/`i32` only (R7). |
| Lapped/overlapped transform | Needs neighbours at decode. Fails R2. |
| Wavelet (CDF 9/7, as Fenix uses) | Better at very low rates and naturally multiresolution, but block-DCT has better random access, far simpler GPU mapping, and better ratio in the visually-lossless regime we care about. Revisit only if RD curves say otherwise — `bench/` compares against it. |
| Zigzag + run-length + EOB | Serial. The 64-bit mask hierarchy is the parallel equivalent. |
| Global DC DPCM across chunks | Serial dependency chain. The DC plane (§3.6) gets the same win without it. |
