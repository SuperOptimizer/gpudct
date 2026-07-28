# gpudct — Quality metrics and test strategy

Two separate concerns that are easy to conflate:

- **Correctness** — the codec does what the format says, on every backend, for every
  input. Binary pass/fail, enforced in CI.
- **Quality** — the rate–distortion tradeoff is actually good. Continuous, tracked over
  time, compared against baselines.

---

## 1. Quality metrics

All implemented in `tools/gpudct metrics` and in `bench/`, computed on the full volume
(not sampled — sampling hides exactly the tail behaviour we care about).

### 1.1 Error distribution

The headline set, in input units and normalized by data range:

- **MAE** — mean absolute error.
- **RMSE**, **MSE**.
- **Absolute-error percentiles**: p50, p90, p95, p99, p99.9, p99.99, **p100 (max)**.
  Computed exactly via a full histogram over the integer error domain (for `u8`/`u16`
  the error range is small enough that an exact histogram is trivial and streams in one
  pass; for `f32` we use a two-pass exact method, not a t-digest — approximations at
  p99.99 are worthless).
- **Relative-error percentiles** — same set, `|err| / max(|x|, floor)`, for `f32` data.
- **Signed error mean (bias)** — a nonzero bias indicates a quantizer offset bug, and it
  matters for downstream averaging/segmentation. Should be ≈ 0.

### 1.2 Fidelity

- **PSNR** — with peak set from the dtype (255 / 65535) *and* a variant using the actual
  data range, since scroll volumes rarely span the full dtype. Both reported; the
  data-range one is the honest number.
- **3D SSIM** — a genuine volumetric SSIM with an 11³ Gaussian window (σ=1.5), not
  slice-averaged 2D SSIM. Slice-averaged 2D SSIM is blind to z-axis artifacts, which is
  exactly the failure mode a 3D transform can have.
- **3D MS-SSIM** — 5 scales, standard weights.

### 1.3 Structure preservation (the ones that actually predict downstream utility)

Scroll data is consumed by segmentation and ink-detection models, not by human eyes.
PSNR is a poor proxy for that, so:

- **Gradient error** — MAE and p99 of the error in 3D Sobel gradient magnitude. Catches
  over-smoothing that PSNR forgives.
- **Radially-averaged power spectrum ratio** — `P_dec(f) / P_orig(f)` binned by radial
  frequency. A codec that kills papyrus fiber texture shows a clean rolloff here while
  scoring fine on PSNR. This is our best early-warning metric.
- **Blockiness index** — mean `|Δ|` across chunk/brick faces divided by mean `|Δ|` in
  the interior. 1.0 = no seams. Directly measures whether deblocking is doing its job.
- **Histogram distance** — EMD between original and decoded intensity histograms.
- **Laplacian / high-pass energy ratio** — `‖∇²x_dec‖² / ‖∇²x_orig‖²`. A single scalar
  that says "how much detail did we destroy", cheap enough to report on every run.
- **Local-variance preservation** — error in per-8³-block standard deviation, reported
  as a map and as p50/p99. Catches the specific failure where flat regions are perfect
  and textured regions are mush, which whole-volume PSNR averages away.
- **Isosurface displacement** — for a set of thresholds spanning the histogram, the
  Hausdorff and mean displacement of the extracted isosurface. This is the metric that
  actually corresponds to "does the segmentation still land in the same place".
- **Error autocorrelation** — the error field should look like white noise. Structured
  (spatially correlated) error means we're removing *signal*, not noise, and shows up
  here long before it shows up in PSNR.

### 1.3b Anisotropy checks

A 3D transform can fail asymmetrically, and slice-wise viewing hides it:

- **Per-axis error profile** — MAE and p99 computed separately along x, y, z, plus per
  slice-index. A z-axis-specific artifact (the classic failure of a separable 3D
  transform on anisotropically-sampled data) is invisible in aggregate numbers.
- **Per-axis gradient error** — same split for the Sobel metric.
- **Directional power-spectrum ratio** — the §1.3 spectrum metric, split into axial vs
  in-plane frequency bands.

### 1.3c Worst-region reporting

Aggregate metrics over a 100 GB volume are dominated by air. Every run also reports:

- The **worst 100 bricks** by RMSE, p99 error, and SSIM, with their coordinates, so they
  can be inspected directly rather than inferred.
- Metrics **conditioned on intensity band** (air / low / mid / high) and on local
  gradient magnitude — "error in high-gradient regions" is the number that matters for
  scroll data, and it is 10–50× the whole-volume average.

### 1.4 Rate

- **bits per voxel (bpv)** and compression ratio.
- Component breakdown: masks / levels / signs / DC plane / correction layer / headers.
  Knowing which stage is spending the bits is what makes tuning possible.

### 1.6 Throughput (reported alongside quality, never separately)

Ratio and speed trade against each other, so a quality report that omits speed is
misleading. Every run reports encode and decode GB/s (of *uncompressed* volume) for each
available backend, plus GPU kernel time split between K1 (entropy) and K2 (transform),
achieved memory bandwidth, and occupancy. A ratio win that costs 3× decode time should
be visible as such in the same table.

### 1.5 Reported as

`bench/` produces RD curves (bpv vs each metric) over a quality sweep, plus a single
summary table per corpus. CI stores results as JSON so regressions are diffable.

---

## 2. Baselines

We only get to claim "high ratio" relative to something. Every RD sweep runs against:

| Baseline | Why |
|---|---|
| **3ddct** | The closest prior art and the direct comparison the project exists to beat. Same 16³ chunks. |
| **ZFP** (fixed-accuracy + fixed-rate) | The standard bounded-error scientific volume codec. |
| **SZ3** | The other standard, prediction-based. Usually wins on smooth simulation data, loses on textured CT — worth knowing where the crossover is. |
| **blosc2 + zstd/blosclz** (lossless) | The ratio floor and what Zarr users have today. |
| **JPEG-XL / AVIF, slicewise** | Sanity check that 3D is actually buying us something over strong 2D. If it isn't on some dataset, that's important to know. |
| **CDF 9/7 wavelet** (Fenix's approach) | Validates the DESIGN.md §7 rejection of wavelets, or overturns it. |

Success criterion for M8: beat 3ddct's ratio at equal 3D-SSIM, at ≥ 5× its throughput
on CPU and ≥ 50× on GPU.

---

## 3. Test corpus

`tests/data/` holds small committed volumes; large ones are fetched by
`tools/fetch_corpus` and cached.

**Real:**
- Vesuvius scroll 1/2/5 sub-volumes at several depths (surface, interior, air gaps,
  ink-bearing regions) — `u8` and `u16`, from the public dl.ash2txt.org data.
- A microCT scan and an MRI volume (different noise/texture statistics).
- An `f32` simulation field (smooth — the case where SZ3 wins; we should not embarrass
  ourselves).

**Synthetic (for correctness, not ratio):**
- Constant, impulse, single-frequency sinusoid per axis, sharp step, random uniform
  noise, Shepp–Logan 3D phantom, fractal/Perlin noise, gradient ramp.
- Adversarial: values at dtype extremes, one nonzero voxel per chunk, alternating
  ±max checkerboard (worst case for dynamic range), all-zero, all-max.

---

## 4. Correctness tests

### 4.1 Unit

- **Transform**: forward+inverse round-trip error vs a `double` naive DCT reference,
  over the full input range. Orthogonality, coding gain, and frequency-response checks
  on the generated integer matrix (`tools/gen_transform` asserts these too).
- **Dynamic range**: interval-arithmetic proof over the factorization that no
  intermediate overflows `int32` for the full `u16`/`i16` input range. This is a *test*,
  not a comment, because it's the thing most likely to silently break when the
  factorization is optimized.
- **Quantize/dequantize**: reciprocal-multiplier path must equal the exact division
  path for every `(coefficient, q)` pair in range. Exhaustive, not sampled.
- **rANS**: round-trip on random symbol streams; adversarial distributions (one symbol
  with probability 4095/4096, uniform, single-symbol alphabet); verify interleaved
  encode/decode matches the scalar reference exactly; verify the encoder never emits a
  state outside its renormalization window.
- **Mask hierarchy**: round-trip over random sparsity patterns, including all-zero,
  all-nonzero, and one-bit-set-per-level.

### 4.2 Cross-backend equivalence — the load-bearing test

Because the transform is `f32` (DESIGN.md §3.2), backends are equivalent within a
tolerance rather than byte-identical. The test has three tiers, from strictest to
loosest, and each tier catches a different bug class:

**Tier 1 — entropy layer, bit-exact.** For a fixed bitstream, every backend must decode
*identical quantized levels*. rANS is pure integer arithmetic, so any difference here is
a real bug with no float excuse. Compared by hashing the sparse `(index, level)` arrays.

```
levels_cpu_scalar(A) == levels_cpu_simd(A) == levels_cuda(A)   // exact
```

**Tier 2 — reconstruction, tolerance-based.** For a fixed bitstream:

```
assert max |decode_X(A) − decode_scalar(A)| ≤ 1 LSB     for all backends X
assert  count(differing voxels) / total < 1e-4
assert  RMS difference < 0.05 LSB
```

A voxel differing by ≥2 LSB means the operation order diverged more than float
non-associativity can explain — a real bug. The RMS bound is the one that catches slow
drift: a subtly wrong constant passes the max-difference test on most volumes but moves
RMS immediately.

**Tier 3 — encoder agreement.** Different encoder backends may emit different bitstreams
(a coefficient on a quantizer boundary can land either side). So we assert the *quality*
matches rather than the bytes:

```
assert |PSNR(enc_X) − PSNR(enc_scalar)| < 0.01 dB
assert |size(enc_X)  − size(enc_scalar)| / size < 1e-3
assert fraction of differing quantized levels < 1e-5
```

Golden-file tests (§4.4) pin the **scalar CPU encoder** specifically, since it is the
only one defined to be reproducible byte-for-byte across machines (built with
`-ffp-contract=off`, no fast-math).

Runs on every PR (CPU) and nightly (GPU). Any Tier-1 divergence is a P0.

### 4.3 Property / fuzz

- Round-trip on randomly generated volumes with random dims (including non-multiples of
  16 and 128 — padding/edge handling is the classic bug farm), dtypes, and quality
  settings; assert declared error bounds hold.
- **Bounded-error invariant**: for `--max-error τ`, assert `p100 ≤ τ` on every voxel of
  every corpus volume. For percentile modes, assert each declared percentile holds.
  This is the codec's core promise; it is checked exhaustively, not spot-checked.
- **Decoder fuzzing** (libFuzzer + ASan/UBSan/MSan) on malformed and truncated
  bitstreams. These files get shared between researchers, so a decoder that can be made
  to read out of bounds is a real vulnerability, not a theoretical one. Corpus seeded
  from valid streams with bit flips. Zero crashes, zero sanitizer reports, gated in CI.
- Decoder must reject: bad magic, unsupported version, inconsistent dims, offsets past
  end-of-buffer, brick sizes that don't sum, out-of-range table entries.

### 4.4 Golden / conformance

Checked-in reference bitstreams with SHA-256 hashes, plus their expected decoded
hashes. Any change to them requires an explicit format-version bump and a changelog
entry. This is what stops accidental format drift once real data exists in the wild.

A `conformance/` set of hand-built streams exercising each format feature (all-zero
brick, escape-coded levels, custom tables, correction layer present/absent, each
profile) so a third-party decoder can be validated.

### 4.5 Performance regression

Benchmarked in CI on fixed hardware, JSON-recorded, with thresholds:
encode/decode GB/s (CPU 1-thread, CPU all-thread, GPU), and ratio + 3D-SSIM at fixed
quality. A >3 % ratio regression or >5 % throughput regression fails the build. Without
this, performance work silently rots.

---

## 5. Sanitizers and static analysis

- ASan + UBSan on all CPU tests; MSan on the decoder specifically (uninitialized reads
  from a partially-parsed bitstream are the likely bug).
- `compute-sanitizer` (memcheck, racecheck, synccheck, initcheck) on all CUDA tests —
  racecheck in particular, since the warp-synchronous rANS decode relies on implicit
  lockstep assumptions that are wrong on Volta+ without explicit `__syncwarp`.
- clang-tidy + `-Wall -Wextra -Wconversion` as errors.

## Measured against 3ddct

3ddct is the closest prior art and the codec this project exists to beat.
Measured on a real full-resolution 128^3 brick from `PHerc0332`, using BD-rate --
the average bitrate difference at *equal quality*, which is the only fair way to
compare two codecs whose quality knobs are on different scales:

| volume | default | `--effort high` |
|---|---|---|
| PHerc0332 L0 (a) | -12.94% | -14.22% |
| PHerc0332 L0 (b) | -27.65% | -28.92% |
| PHerc1667 L0 | -16.31% | -18.21% |
| PHerc0500P2 L0 | -12.03% | -13.53% |
| PHerc0332 L3 (8x downsampled) | -8.29% | -9.98% |
| **mean** | **-15.44%** | **-16.97%** |

BD-rate against PSNR; the mean against 3D-SSIM at `--effort high` is **-17.10%**.
Negative means gpudct uses fewer bits at equal quality.

Spread matters more than the mean here. The margin is largest on high-contrast
full-resolution material and nearly vanishes on the 8x-downsampled level-3
volume, where downsampling has already removed the redundancy the transform
feeds on. A single-volume number would have been misleading either way: the
first volume measured was one of the *worst* cases.

Throughput, single-threaded, best-of-8 interleaved, at comparable ratio
(gpudct 16.45x / 3ddct 15.51x):

| build | gpudct enc | 3ddct enc | gpudct dec | 3ddct dec |
|---|---|---|---|---|
| matched flags (`-O2`, no fast-math) | **179 MB/s** | 90 MB/s | 113 MB/s | **144 MB/s** |
| each at its own recommended flags   | 179 MB/s | **198 MB/s** | 113 MB/s | **202 MB/s** |

Two things worth separating there. Under matched build discipline gpudct encodes
2x faster and decodes 0.79x as fast, so the implementations are closer than the
headline suggests -- much of 3ddct's advantage is that it is built with
`-ffast-math -march=native`, which it is designed for and which gpudct
deliberately refuses (it would break the bit-reproducibility the golden tests
pin). Under each codec's intended build, which is what a user actually gets,
3ddct decodes ~1.8x faster.

So: **gpudct wins on ratio and on encode, and loses on decode.** Closing the
decode gap is the outstanding work; roughly 40% of it is the inverse transform
and the rest the entropy stage.

### A note on benchmarking method

Timings on a laptop vary by 40% run to run, enough that several "improvements"
measured here initially reversed under repetition. Everything above is
best-of-N with the two codecs interleaved in the same loop, which cancels
thermal drift; single runs, and A/B comparisons taken minutes apart, produced
confidently wrong answers more than once. An isolated microbenchmark showed AVX2
39% ahead on the transform while an interleaved end-to-end A/B showed no gain at
all -- the microbenchmark was measuring something real that did not survive in
context.

### Context modelling: what did not work

Three attempts to strengthen the entropy contexts all measured worse, and the
reason is the same each time -- the tables are **static and global**, so every
extra context splits one fixed model set more thinly:

| change | BD-rate at `--effort high` |
|---|---|
| baseline (3 level contexts, 2 mask contexts) | **-7.49%** |
| mask context on popcount (2 -> 4 buckets), levels on magnitude sum (3 -> 5) | -6.89% |
| ... additionally trained on mixed normal/high effort | -6.69% |
| magnitude sum with only 3 buckets | -5.69% |

The last row is the instructive one. A running magnitude sum is the better
statistic in principle and is what HEVC-style coders use -- but capped at three
buckets it saturates after two or three coefficients, so every later coefficient
lands in the top bucket and it discriminates *less* than the previous magnitude
does. More context is not better when the bucket count cannot express it.

**This has since been implemented** (`src/core/cluster.hpp`,
`src/core/brick_tables.hpp`), following fenix: each brick measures its own symbol
statistics, greedily merges raw contexts into at most a handful of tables with
the table-signalling cost inside the merge objective, and ships them in its
payload. A brick that cannot pay for its tables keeps the global ones, so the
change cannot regress -- the worst case is one flag byte.

It was worth **-5.6% BD-rate on its own** (mean over five volumes went from
-8.71% to -14.35% at default effort), and the largest gain landed on the volume
where the global tables fitted worst (-2.56% to -9.27%).

Re-running the failed context experiments afterwards then *did* pay: with
per-brick tables, going back to four mask contexts and five level contexts
improved the mean from -15.94% to -16.41%. The contexts were never the problem.

Also measured and rejected: training the tables on a mix of normal and RDO
output. RDO changes the coefficient statistics enough that the mixed tables are
worse for both modes than tables trained on plain quantization alone.

### The dead-zone depends on whether RDO is running

`level = floor(|c|/q + offset)`, so the zero bin has half-width `(1-offset)*q`
and a *lower* offset means a wider dead zone. Swept over three real volumes,
BD-rate against 3ddct:

| offset | 0.20 | 0.34 | 0.45 | 0.55 | 0.70 |
|---|---|---|---|---|---|
| plain quantization | -3.13% | **-7.16%** | -6.15% | +2.75% | +52.45% |
| with RDO | -4.48% | -8.73% | -9.47% | **-9.49%** | -9.49% |

The optimum moves from 0.34 to ~0.6, and the plain-quantization curve falls off a
cliff exactly where the RDO curve plateaus. That is not a coincidence: without
RDO the dead zone is the only mechanism discarding noise coefficients, so it has
to be wide; with RDO the discarding is already being done optimally, and a wide
dead zone only destroys coefficients RDO would have kept. The encoder now picks
per effort level, which is worth ~0.8% BD-rate at `--effort high`.

Worth noting against fenix, whose `dz_frac = 0.80` looks like a much wider dead
zone than ours: that parameter is the zero-bin *width*, which corresponds to an
offset of 0.20 here -- the worst value in both rows above. The parameterizations
are inverses of each other, and comparing them directly would have led us the
wrong way.

### Per-brick tables on the GPU

Both backends build them. The device histograms its symbols (`ke2c_histogram`),
the host clusters, and the resulting tables are uploaded before the range-coding
kernel runs; on decode the compact frequency tables are uploaded and expanded
into slot tables by a kernel rather than being built on the host and shipped.

Two things about this were slower than expected and worth recording:

- **Clustering dominated encode.** The greedy merge recomputed every pair's cost
  after every merge -- O(n^3) cross-entropy evaluations over a 256-symbol
  alphabet, roughly 100M operations per brick. Caching the pairwise deltas and
  refreshing only the merged cluster's row and column, and running the per-brick
  loop across threads, took GPU encode from 350 MB/s back to 1552.
- **The device path had to parallelize it explicitly.** The CPU encoder got this
  for free because it already runs bricks through `parallel_for`; the CUDA host
  code was a serial loop, and that single difference was the whole 4x gap.

### The level context: four wrong answers and the right one

The AC level models carry ~45% of all coded bits, so their context was worth
four attempts. BD-rate at `--effort high`, five volumes:

| context | result |
|---|---|
| previous magnitude, 3 buckets | -14.35% (with per-brick tables) |
| previous magnitude, 5 buckets | **-16.41%** |
| running scan sum, 6 buckets | -15.57% |
| **3D causal neighbour sum, 6 buckets** | **-16.97%** |

The three losers were all the same mistake in different clothes: a *running total
over the scan* rather than a *local neighbourhood*. A running total answers "how
much energy has this sub-block spent so far", which saturates almost immediately
and says little about the next coefficient. The neighbour sum -- the already-coded
(u-1,v,w), (u,v-1,w), (u,v,w-1) -- answers "is this coefficient in an active
region", which is what actually predicts it.

fenix describes its context as a "neighbour-magnitude-sum", and it was read here
first as a scan-order sum. Two of the failures above are that misreading. The
scan order makes all three neighbours causal for free, since each has a strictly
smaller bit index within the sub-block, so the correct version costs nothing
extra to compute on either backend.

### Profiling the encoder: three wrong guesses

GPU encode sat at 1.33 GB/s with per-brick tables enabled, and the first three
things blamed for it were all wrong:

| change | expected | measured |
|---|---|---|
| neighbour scratch to shared memory | large | nothing |
| warp-aggregated histogram atomics | large | nothing |
| fast approximate log2 in clustering | large | nothing, and **-1.8% ratio** |

The instrumentation was the problem: the event marking the end of the
symbol-building kernel sat *after* the host clustering, so "ke2a" had been
reporting 250 ms of work that belonged to a stage which was not being measured at
all. Splitting the timers correctly showed the actual costs -- histogram kernel
156 ms, clustering 111 ms, model construction ~94 ms -- and each then had an
obvious fix:

- The per-brick table builder materialized a whole 75-model `ModelSet` per brick,
  each carrying a 4096-entry slot table that an encoder never reads. Skipping it
  on the encode path: 106 ms -> 50 ms.
- The histogram was a second kernel re-reading the symbol array. Each thread
  walks its own chunk's buffer, so a warp's 32 reads land ~6 KB apart and are
  fully uncoalesced. Folding the counting into the kernel that already has the
  symbols in registers removed 154 ms and cost 6 ms.

Net: 1.33 -> 1.77 GB/s. The lesson is the one this file keeps recording -- the
measurement has to be right before the optimization can be.

Also worth keeping: the approximate logarithm was rejected on *ratio*, not speed.
Clustering decisions are sensitive enough that ~1e-4 of error in the entropy
estimate changes which contexts merge, for 1.8% of BD-rate. Heuristics that feed
a merge objective are not automatically safe places to approximate.

### The quantizer matrix is a PSNR/SSIM trade, not free ratio

`q(u,v,w) = base * (1 + a * r^b)` with `r` the normalized radial frequency. The
profile values for `a` and `b` were M1 placeholders -- picked by intuition and
never measured, which ROADMAP M4 had flagged. They are measured now.

Self-BD-rate against the shipped balanced profile (`a=3, b=2`), five real
volumes, `--profile balanced`, at `b=2`:

| a | BD-PSNR | BD-SSIM |
|---|---|---|
| 3.0 (shipped) | 0.00% | 0.00% |
| 2.0 | -1.03% | +0.38% |
| 1.0 | -1.94% | +1.61% |
| 0.5 | **-2.21%** | +2.80% |
| 0.0 | -2.10% | +4.76% |

Flattening the matrix buys PSNR and costs SSIM, monotonically and with no
crossover, so there is no free 2% here -- only a choice about which metric the
default profile should favour. That it is a clean trade is itself the useful
result: it means the radial weighting is doing exactly what it was meant to do,
and the shipped `a=3` sits at the structure-preserving end of a real frontier
rather than at an arbitrary point off it.

Two cautions this measurement produced:

- **A single volume said the opposite.** On the first volume looked at, the
  archival profile appeared to match balanced on SSIM while beating it on PSNR,
  which reads as a free win. Across five volumes the SSIM cost is unambiguous.
  One volume is not a measurement; this file already says so about throughput,
  and it is just as true of ratio.
- PSNR is minimized by a flat matrix almost by construction -- the transform is
  orthonormal, so uniform quantization is the MSE optimum at high rate. A sweep
  scored only on PSNR would have driven `a` to zero and quietly destroyed the
  high-frequency content the fibre-texture use case depends on. The band-energy
  columns in `gpudct eval` are there to make that visible.

The `b` (radial exponent) sweep is not finished. What exists so far, at `a=1`:
`b=1.0` gives -1.87%/+2.09%, `b=1.5` gives -1.92%/+1.50%, `b=2.0` gives
-1.94%/+1.61% -- i.e. `b` moves things far less than `a` does, and no value
tested escapes the frontier.

Reproduce with `gpudct eval --profile balanced --qamp A --qshape B`; `--qamp`
and the `--profile` narrowing of `eval` exist for this sweep.

### Profiling the encoder, again: the remainder was 60% of the time

The lesson from the previous round -- that a stage timer covering only the
kernels sends optimization at the wrong stage -- was recorded but not fully
applied. The CUDA timers still measured only kernel spans, so they accounted for
**40% of decode and 32% of encode**, and the rest was invisible. `StageTimer` now
carries a host wall clock and prints an explicit `unaccounted` column, so the
report can be checked against its own total instead of believed.

The corrected profile, 1024^3 at quality 1.0, per gigabyte:

| decode | ms | encode | ms |
|---|---|---|---|
| readback | 92 | **tables (host clustering)** | **231** |
| host prep | 54 | ke2b (rANS) | 138 |
| k1 (entropy) | 45 | ke2a (symbols) | 48 |
| unaccounted | 46 | payload assembly | 33 |
| k2 (transform) | 11 | ke1 (transform) | 11 |

The kernels are **22% of decode and 6% of encode**. Every intuition about where
to optimize this codec on the GPU was wrong: K1 and K2 are not the problem, and
neither is the transform. Encode is host clustering; decode is transfer and
per-batch host preparation.

### An exact logarithm that is also fast

Clustering was the largest single cost in encode, and essentially all of it was
`std::log2`. The approximate logarithm rejected earlier (-1.8% BD-rate) was
solving the right problem the wrong way. Two exact changes:

- `-log2(c/total)` is `log2(total) - log2(c)`. The first term is loop-invariant,
  and the second is the logarithm of an *integer count* -- so it can be looked up
  in a precomputed table holding exactly what `std::log2` returns.
- `merge_delta` allocated a vector and walked the 256-symbol alphabet six times
  (build `merged`, two `cross_entropy_bits` calls that each recompute the total,
  then `distinct`). Wherever the merged count is nonzero both inputs pay the same
  per-symbol cost, so their counts add before multiplying and it becomes one pass
  with no allocation.

Together: encode **977 -> 1086 MB/s**, with the golden test still passing --
the bitstream is byte-identical. Approximation was never required; the operands
were integers all along.

### Four more things that measured nothing

| change | expected | measured |
|---|---|---|
| overlapped slab readback on a second stream | ~25% of decode | nothing |
| cached per-cluster totals | one fewer pass | nothing |
| branchless merge inner loop | fewer mispredicts | indistinguishable |
| `--streams` 16 -> 32 | large (thread starvation) | +3% decode, -0.2% ratio |

The readback overlap is the interesting failure. It worked exactly as designed --
the measured download went from 92 ms to 0.01 ms -- and bought no end-to-end time
at all, twice, interleaved. Whatever the transfer was contending with, it was not
on the critical path. Roughly 60 lines, a second stream, and a pinned-lifetime
invariant were reverted for it.

The stream sweep is worth keeping as a number: 4 -> 32 streams is +16% decode for
0.3% of ratio, and 64 streams is *worse* than 32 on both. The hypothesis being
tested was that 512 bricks x 16 streams = 8192 threads badly under-occupies the
GPU. It does not; the codec is not parallelism-starved at this size.

### Two measurement traps in the bench harness

**`decode()` versus `decode_into()`.** `decode()` sizes the output with
`std::vector::resize`, which value-initializes -- so a fresh 1 GB output costs a
full zero-fill plus page faults, measured at more than every other decode stage
combined. That is real for a caller who needs a new buffer, but it is allocator
behaviour rather than codec throughput, and the API already offers `decode_into`.
The bench now reuses one buffer. Any decode number measured before this change is
not comparable to one measured after it.

**The noise floor.** Encode over three interleaved pairs spanned 892 to 1107 MB/s
on identical code -- 24%. Nothing smaller than about 10% can be resolved by this
harness on this machine, which is why the branchless loop above is recorded as
"indistinguishable" rather than as a small win or a small loss.

### Sub-brick random access, and what `streams_per_brick` really controls

The brick is the independently decodable unit, so "the chunk is never addressable
on its own" was the standing description. That was true of the API and false of
the format. Chunk `ci` is coded into stream `ci % P`, and each stream's chunks
appear in increasing index order, so reaching one chunk needs **one stream's
prefix** -- about `ci / P` chunk decodes -- not all 512. The prefix must be
entropy-decoded, because rANS state carries from chunk to chunk within a stream
and that coupling is what buys the ratio; but it does not need the inverse
transform, which runs once for the target chunk only.

`decode_chunk` does this. Measured on a 512^3 scroll volume, 200 random reads:

| P | ratio | `decode_chunk` | `decode_brick` | cheaper by |
|---|---|---|---|---|
| 1 | 13.25x | 3.97 ms | 20.7 ms | 5.2x |
| 4 | 13.25x | 1.87 ms | 20.8 ms | 11.1x |
| 8 | 13.24x | 1.62 ms | 20.8 ms | 12.8x |
| 16 | 13.23x | 1.29 ms | 20.8 ms | **16.2x** |
| 32 | 13.21x | 1.20 ms | 20.7 ms | 17.3x |

Even at `P=1`, where the prefix is the whole brick up to the target, it is 5.2x
cheaper -- because skipping 511 inverse transforms is most of the saving. The
transform is a larger share of brick decode than the entropy stage is.

This reframes `P`. It had been treated as a GPU-parallelism knob costing ratio;
it is simultaneously the random-access granularity knob, and the two want the
same thing. The default moves from 4 to 16 on this evidence, measured on the same
volume:

| | 4 -> 16 |
|---|---|
| ratio | **-0.15%** |
| GPU decode | **+13%** |
| random chunk read | **1.45x cheaper** |
| CPU decode | unchanged |
| CPU encode | **-8%** |

A volume is encoded once and read many times, so trading 8% of encode for 13% of
GPU decode and a much cheaper cache miss is the right side of that deal. Drop
back to 4 when archive size is the only thing that matters -- it is still only
0.15%.

The comparison that prompted this: fenix decodes a 64^3 brick and caches 16^3
chunks, so a cache miss costs a 256 KiB decode. A 128^3 brick makes that 2 MiB,
8x worse -- which was a real deficiency until the prefix walk existed. With it, a
miss costs 1.29 ms against fenix's brick decode rather than 20.8 ms.

### Decode latency, which is a different problem from decode throughput

The target use case is a viewer keeping the volume *compressed* in VRAM and
decoding the visible working set on demand, so effective VRAM is `ratio` times
larger. What matters there is not GB/s on a gigabyte batch but the latency of
decoding a handful of bricks. Those turn out to be almost unrelated numbers, and
every optimization in this file up to this point was aimed at the wrong one.

Decode of a batch, K1 time, at the default 16 streams:

| batch | K1 | end to end |
|---|---|---|
| 8 bricks (16 MiB) | 34.7 ms | 386 MB/s |
| 512 bricks (1 GiB) | 45.4 ms | 1992 MB/s |

K1 barely changes across a 64x change in batch size, because its wall time is set
by the **per-thread serial chain**, not by total work: each thread decodes
`512/P` chunks one after another, and a chunk is a data-dependent rANS parse. At
8 bricks there are `8 * P` threads on a device with ~10,000 lanes, so none of the
per-symbol memory latency is hidden.

Two consequences, both measured.

**`P` is a latency knob, and a strong one.** K1 time falls as `1/P` almost
exactly, because it divides the serial chain:

| P (8 bricks) | K1 | ratio |
|---|---|---|
| 16 | 34.7 ms | 13.23x |
| 32 | 18.2 ms | 13.21x |
| 64 | 8.9 ms | 13.16x |

4x lower latency for 0.5% of ratio. This is invisible in the throughput benchmark
-- at 1024^3 the three values are within noise of each other -- which is why it
went unnoticed for so long. `P=64` is the practical end of the lever: one stream
per chunk would cost roughly 6% of ratio in flush overhead.

**Slot tables belong in shared memory when the batch is small.** The symbol
decode's first dependent load is `slot[table][state & 0xfff]` -- 4096 bytes per
table, randomly indexed, nothing to prefetch. `k1_entropy_decode_shared` gives
each block one brick and stages that brick's slot tables in shared memory:

| bricks | shared | flat |
|---|---|---|
| 8 | **5.52 ms** | 8.89 ms |
| 27 | **6.02 ms** | 9.87 ms |
| 64 | 11.96 ms | **9.90 ms** |

Below the crossover it is ~39% faster; above it, the flat launch's occupancy
wins. The host picks by brick count (`kSharedLaunchMaxBricks = 32`).

Together, decoding 8 bricks went from **34.7 ms to 5.52 ms of K1**, and 386 to
1078 MB/s end to end.

Also tried and measured neutral: packing the per-brick `freq`/`cum` into one word
the way the global tables already do. It removes a load per symbol and changed
nothing (8.93 vs 8.91 ms), which says those two arrays were already cache-
resident -- 16 KB per brick, shared by every thread in the block. The slot table
is 64 KB and is not. The change was kept anyway, because one packed array is
simpler than two parallel ones and symmetric with `DeviceModels`.

A measurement note: an early 27-brick reading showed 66 ms for the shared launch
and did not reproduce -- three clean reps gave 6.0 ms. It was taken immediately
after generating the test volume with a slow script, so the machine was still
busy. A number that disagrees with its neighbours by 10x is a bad measurement
until proven otherwise; it should not be reasoned about.
