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
- **Isosurface displacement** — how far the isosurface moved, in voxels. This is the
  metric that actually corresponds to "does the segmentation still land in the same
  place", and it is the one the others cannot stand in for: a codec can hold a fine
  PSNR while shifting a sheet a fraction of a voxel, and bias accumulates along a
  traced sheet in a way that jitter does not. Implemented; see below for what it
  reports and what it measures on real data.
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

### 2.1 NVENC / NVDEC — the fixed-function baseline

A GPU already contains silicon that compresses 8-bit images. If a volume is fed to it
as video (z becomes time), how much does a purpose-built 3D codec actually add? Run
`bench/nvcodec_bench.py`; measured on an RTX 5080 Laptop, driver 610.88, against
`big512.raw` (512³ u8).

**Rate-distortion is close to a wash, and the test volume decides the sign.** Three
datasets, BD-rate of gpudct's convex hull against HEVC at a 512^3 clip:

| data | BD-rate | |
|---|---|---|
| synthetic `big512.raw` | +1.1% | tie |
| real, PHerc0332 **L3** (8x downsampled, 25% masked zeros) | **+23.5%** | HEVC wins clearly |
| real, PHercParis4 **L0** (2.4um full-res, 100% content) | **+0.1%** | tie |

The L3 number is the outlier and must not be quoted as the real-data verdict: 8x
downsampling has already removed the high-frequency content the transform feeds on. The
3ddct comparison below shows the identical collapse on the identical volume (-8.29% at L3
against -27.65% at full resolution), which is what makes downsampled data a systematically
misleading benchmark for this codec rather than merely a harder one.

On full-resolution scroll CT -- the data that matters -- the crossover is what to
remember, not the aggregate:

| PSNR | gpudct | HEVC 512^3 | |
|---|---|---|---|
| 32 dB | 212x | **271x** | HEVC +28% |
| 36 dB | 86.7x | 93.7x | HEVC +8% |
| 39 dB | **47.4x** | 46.7x | +2% |
| 42 dB | **27.4x** | 24.5x | **+12%** |
| 45 dB | **16.2x** | 13.4x | **+21%** |

We win where the archive would actually sit and lose below 36 dB, which is lossy enough
that downstream ink detection is likely compromised regardless.

Detail from the synthetic volume follows; its per-codec table is retained because the
settings sweeps and the AV1 range bug were found on it.

BD-rate of gpudct against each hardware codec on `big512.raw`, negative meaning gpudct
needs fewer bits:

| codec | brick | overlap | BD-rate |
|---|---|---|---|
| HEVC | 256³ | 26.9–46.6 dB | −2.0% |
| HEVC | 512³ | 26.9–46.6 dB | **+1.1%** |
| AV1 | 256³ | 42.4–54.3 dB | −10.2% |
| AV1 | 512³ | 42.4–54.3 dB | −9.8% |

The single number hides a crossover, which is the more useful result — HEVC wins below
~36 dB, gpudct wins above it, and the gap widens in both directions:

| PSNR | gpudct | HEVC 512³ | AV1 512³ |
|---|---|---|---|
| 30 dB | 73.1× | **88.4×** | — |
| 36 dB | 28.6× | 29.5× | — |
| 42 dB | **13.4×** | 11.6× | — |
| 48 dB | **6.5×** | — | 5.8× |

So on ratio alone, hardware HEVC is a genuine peer and beats us at viewing quality.
What gpudct wins on is everything else:

**Speed.** Both sides must be measured *saturated*, which is the trap here: a single
ffmpeg process uses one of the card's engines and is additionally limited by ffmpeg's
per-frame CPU work, so it understates the hardware badly. Two independent signatures show
this. Decode throughput scales with frame *area* while frames/second stays pinned near
3000–5500 (0.24 GB/s at 256², 1.46 at 512², 3.14 at 1024²) — a fixed per-frame cost, not
a pixel rate. And running N clips concurrently keeps climbing to N≈4:

| N concurrent | NVDEC 1024³ | NVENC 512³ |
|---|---|---|
| 1 | 1.93 GB/s | 0.41 GB/s |
| 2 | 3.43 | 0.75 |
| 4 | 4.11 | 0.90 |
| 8 | **4.31** | **0.90** |

Saturation at ~2.2× single-stream, plateauing at N=4, is consistent with 2 NVENC and 2
NVDEC engines on this part. N=8 raised no session errors, so the driver's encode-session
cap is not what binds. Against those ceilings, device-resident in both cases (NVDEC via
`-hwaccel cuda`, gpudct via `DeviceVolume`, neither copying back):

| | rate | vs saturated ASIC |
|---|---|---|
| **decode** — gpudct 32×128³ bricks, 6.5 ms | 10.4 GB/s | **2.4× faster** than NVDEC's 4.31 |
| **encode** — gpudct | 608 MB/s | **1.5× slower** than NVENC's 0.90 GB/s |

So the hardware wins on encode and loses on decode. An earlier revision of this section
claimed the opposite for encode, by comparing gpudct's whole-GPU throughput against a
single NVENC session; the fixed-function encoder is genuinely faster than our kernels
once both engines are in use. Decode latency for a single 128³ brick is 5.5 ms, which
NVDEC cannot match at any batch size because it cannot decode a 128³ brick at all — see
below.

**Granularity costs a video codec far less than expected**, which is worth recording
because it contradicts the premise this comparison started from — that video codecs earn
their ratio from long GOPs, so their advantage comes from giving up random access.
Varying one axis at a time on the same voxels:

| lever | qp24 | qp36 |
|---|---|---|
| frame size, 256² → 1024² (16× the pixels) | +1.9% | +7.6% |
| clip length, 32 → 1024 frames (32× longer) | +5.6% | +16.0% |

Both are weak, and length saturates past ~128 frames. The 64× granularity sacrifice buys
HEVC 3–10%, not the large win the framing assumed — so a video codec gives up much less
by working in small units than expected, and correspondingly gains much less from being
handed the whole volume.

That does *not* make the two even, though, which an earlier revision of this section
concluded by comparing our 128³ brick against HEVC clips of 256³ and up. Held to the same
128³ unit, HEVC manages 17.83× at 37.64 dB against our ~23.2× — see the granularity note
below. The differentiators:

- **NVENC will not encode a frame below 129×129** (145 for H.264) and **NVDEC will not
  decode one below 144×144** (142² fails, 144² works; depth is unconstrained). A 128³
  brick therefore cannot be a frame. It *can* be padded to 144² and trimmed after
  decoding, and that costs only **2.6%** (18.31× → 17.83× at qp24), so this is a real
  workaround rather than a wall — an earlier revision of this section overstated it as
  one. What it does mean is that the comparison at *matched* 128³ granularity is
  **17.83× at 37.64 dB for HEVC against ~23.2× for gpudct, about 30% in our favour**.
  The apparent tie above comes from letting HEVC use 256³ and larger units.

  An apron of real neighbouring voxels *is* required, and an earlier revision of this
  section concluded the opposite from a metric that could not see the effect. Comparing
  edge PSNR against core PSNR inside one unit asks whether the codec degrades *near* a
  border; it does not. A seam is a discontinuity *across* the join between two units
  coded without knowledge of each other — a derivative artifact — and per-unit PSNR is
  blind to it by construction. HEVC deblocks and applies SAO within a frame, but nothing
  filters across clips.

  Measured with the blockiness index of §1.3 (`bench/seam_test.py`), tiling 512³ into
  128³ units:

  | config | ratio | PSNR | seam excess over original |
  |---|---|---|---|
  | replicated pad | 31.32× | 39.80 | **+40.8%** |
  | apron 8 | 24.33× | 39.75 | +9.0% |
  | apron 16 | 20.74× | 39.77 | +11.8% |

  At qp30 plain padding reaches +62.7%. An 8-voxel apron removes about 78% of the excess
  and costs 20–22% of ratio; 16 is no better than 8, so 8 is the size to use.
- **Access within a clip is still sequential.** Reaching slice 400 means decoding 400
  frames regardless of how short the clip is, whereas a gpudct brick is one dispatch and
  `decode_chunk` reaches 1/P of a brick.
- **Decode throughput**, 2.4× as above.

**Seams, and why the comparison changes once both codecs are tiled.** gpudct's own
blocking is *worse* than HEVC's when untreated — +69.8% seam excess at 128³ brick faces
and +45.9% at 16³ chunk faces, at balanced/0.5 — but the deblocking filter is decode-side
and costs no bits at all, where HEVC's apron costs 20–22% of bitrate:

| config | ratio | PSNR | seam excess |
|---|---|---|---|
| gpudct, no deblock | 37.88× | 40.25 | +69.8% |
| **gpudct `--deblock`** | **37.88×** | **40.38** | **−39.9%** |
| HEVC, replicated pad | 31.32× | 39.80 | +40.8% |
| HEVC, apron 8 | 24.33× | 39.75 | +9.0% |

The tie reported against a single undivided 512³ clip is therefore not a usable
comparison: that configuration has no random access. Tiled into units and paying for
seam-free joins, at the operating points in use:

| target ratio | gpudct `--deblock` | HEVC + apron |
|---|---|---|
| 25× | ~42.7 dB | ~39.7 dB |
| 50× | ~39.0 dB | ~36.4 dB |

About 3 dB in our favour, on top of 2.4× decode throughput.

Two consequences for this codec. `--deblock` is **off by default** while improving PSNR,
removing seams, and costing nothing, which makes the default wrong for every use except
bit-exactness testing. And the calibrated strength of 1.0 is wrong — see below.

### 2.2 The deblocking filter is calibrated too strong

`DeblockThresholds::from` derives its thresholds from the quantizer matrix, on the
argument that a dead-zone quantizer with step q leaves error variance q²/12, so one
strength setting should work across every profile and quality. Swept against the
blockiness index rather than against PSNR alone, it does not. Real full-resolution
PHerc scroll CT, 512³, seam excess over the original volume's own 0.9987 (16³ faces):

| strength | q=0.25, 69× | q=0.5, 37.9× | q=1.0, 21.8× |
|---|---|---|---|
| off | +80.1% | +45.9% | +25.2% |
| 0.25 | **+3.1%** | +21.7% | +21.1% |
| 0.40 | −24.6% | **−3.0%** | +13.1% |
| 0.50 | −31.2% | −14.1% | +4.7% |
| 0.60 | −33.5% | −21.7% | **−1.8%** |
| 1.00 (shipped) | −34.7% | −34.7% | −20.3% |

The index is a ratio, so both directions are failures: above 1.0 the seams are visible,
below it the filter is smoothing chunk faces *more* than the interior, which is signal
removal precisely where a downstream gradient or segmentation operator will look. The
shipped strength overshoots at every rate, and the overshoot is not free — at q=1.0 it
costs PSNR outright (43.26 dB against 43.36 dB with the filter off, where strength 0.5
gives 43.38 dB). Seam-optimal strength is roughly 0.25 / 0.40 / 0.60 across the three
rates, i.e. it *rises* with quality.

That trend is the diagnosis. The q²/12 model is scale-free, so if it were right the
optimal multiplier would be constant. It rises with quality because the model
overpredicts at coarse quantizers: real reconstruction error cannot exceed the signal's
own local variation, so it saturates where the model keeps growing linearly in q. The
threshold is therefore too large exactly where the filter is most aggressive.

**The fix.** The two limits are combined as a reciprocal sum rather than a minimum:

    1/rms = 1/model_rms + 1/(1.7 * activity)

where `activity` is the mean absolute first difference along the filtered axis, over
interior steps of the decoded volume — the same quantity the blockiness index divides by,
measured per axis because scroll data is not isotropic. The form reduces to `model_rms`
when the quantizer is fine and to `1.7 * activity` when it is coarse, with no
discontinuity between. A hard `min` was tried first and is wrong: it switches abruptly at
the crossover and cut the filter to almost nothing across the whole useful rate range
(+80.1% seam excess at q=0.25, i.e. barely better than off).

The gain constant was fitted by back-solving the measured optima above; it comes out at
1.74 / 1.64 / 1.78 across a 4× span of `model_rms`, constant to within the measurement,
which is the evidence that the functional form is right rather than the fit. Strength 1.0
is now correct at every rate:

| quality | seam excess before | after | PSNR off → on |
|---|---|---|---|
| 0.25 | −34.7% | **+5.9%** | 37.10 → 37.36 |
| 0.5 | −34.7% | **−3.6%** | 40.25 → 40.41 |
| 1.0 | −20.3% | **−1.1%** | 43.36 → 43.35 |

Confirmed on a second volume with quite different statistics (5–17× rather than 22–69×),
where the same constant gives +10.2% / +4.4% / +1.0% and PSNR improves at every rate. The
residual error is on the under-filtered side at the coarsest rates, which is the safe
direction: a faint remaining seam costs less than smoothing away real structure.

`GPUDCT_DEBLOCK_DEBUG=1` prints the two magnitudes per axis.


**Would a 256³ brick be better?** Measured, and no. Rebuilding with `kBrickChunks = 16`
and encoding the same volume gives identical PSNR at every rate point and a ratio gain of
+6.6% at q=0.125, +1.9% at q=1.0, +1.1% at q=4.0 — the gain is per-brick entropy tables
amortized over 8× more chunks, so it is largest where tables are the biggest share. That
~2% at working quality costs 8× coarser random access (2 MB → 16 MB per brick), breaks
the brick = one Vesuvius Zarr chunk mapping so every access becomes an 8× fetch
amplification, and does not survive on the CUDA backend as a constant change (every rate
point failed on device while the CPU path was fine). Note the asymmetry: 256³ helps HEVC
*more* than it helps us (+2.8% at qp24), because their mechanism is GOP length and ours
is table amortization. Neither is worth the granularity.

Caveats, so these numbers are not over-read: the harness drives ffmpeg, whose per-frame
CPU work means the NVDEC figures are a **floor on the silicon, not its ceiling**; encode
throughput repeated at ±30% across runs (p7 measured 170 and 230 MB/s on two occasions),
so only the order of magnitude is meaningful; and `big512.raw` is isotropic
(slice-to-slice MAE 8.97 ≈ in-plane 8.94), which is fair to a video codec but is not real
scroll CT. Re-run on the Vesuvius corpus before quoting any of this externally.

**Settings were tuned, not defaulted** — two BD-rate sweeps over 25 configurations. Three
results worth keeping:

- **Monochrome is a trap.** `-pix_fmt gray` (HEVC Rext) is accepted by NVENC and decoded
  by NVDEC, and costs **+85% bitrate** at equal PSNR. Its mono path is far worse than
  4:2:0, whose two flat chroma planes are nearly free; yuv444p ties 4:2:0 to within 0.01%.
- **Hierarchical B-frames are the biggest lever.** Disabling them costs +13%; referencing
  half of them (`-b_ref_mode middle`) rather than all is worth −12.8%.
- **`-tune uhq` is incompatible with B-frames** on this driver at every resolution up to
  512×512, and B-frames are worth far more, so `-tune hq` is correct.
- **Per-frame-type QP offsets** are worth −1.83% at ±4 (`-init_qpI qp-4 -init_qpB qp+4`).
  Plain `constqp` quantizes every frame in the B pyramid identically, which is not what a
  hierarchical coder wants. ±2 gives −1.57%, ±6 only −0.48%, ±8 is a net loss.

**Tuning is exhausted.** Across three sweep rounds and ~35 configurations, only two
settings are worth anything: `-b_ref_mode middle` (−12.8%) and QP offsets (−1.83%).
Everything else lands inside ±1%: VBR constant-quality −0.29%, `-multipass fullres`
−0.06% (−0.73% combined with VBR), larger `-dpb_size` a small *loss* (+0.21% at 8, +0.65%
at 16), and disabling SEI/metadata exactly zero. Container choice is a real but tiny
effect — mp4 costs a flat ~7 KB against a raw elementary stream, which is 0.22% at qp24
and 6.1% at qp48, so it is under 0.5% anywhere in the 10–100× band and only matters at
extreme ratios. HEVC's numbers here therefore reflect a genuinely tuned encoder, and the
~1.8% still available to it would move the crossover against us by a fraction of a dB
without changing any conclusion.

One measurement bug is recorded here because it nearly became a published conclusion:
`av1_nvenc` does not preserve the full-range flag that HEVC does, so the decoder expands
16–235 to 0–255. That is an affine distortion, not a coding loss, and it pinned AV1's
apparent PSNR at 28.9 dB *regardless of QP* while ratio moved 5×→13×. A quantizer that
changes rate without changing distortion is impossible; that impossibility is what
exposed it. Fitting `orig = 0.863·dec + 15.5` confirmed the 255/219 range ratio, and
`-vf scale=in_range=full:out_range=full` recovered the missing **18 dB**.

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

Runs on every PR (CPU) and nightly (GPU). Any Tier-1 divergence is a P0.

### 4.2b Floating-point policy, and the one place it costs something

The build enables fast floating point: FMA contraction and reassociation. Measured on
real full-resolution scroll, that is worth **+25% CPU decode** (1724 → 2159 MB/s) and
**+2.6% encode** at *identical* ratio — 21.84× and 0.3663 bpv either way. The cost is
that no backend is byte-reproducible any more, which is why §4.2's tiers and the
golden tests (§4.4) assert tolerances rather than hashes. Divergence between the scalar
and SIMD backends measures at most **1 LSB on 0.27% of voxels**, with archive sizes
identical to the byte.

`-fno-finite-math-only` is kept. The `f32` dtype means caller data can legitimately
contain NaN or Inf, and assuming their absence is not a rounding difference — it turns a
representable input into undefined behaviour.

**The exception is the bounded-error modes**, and it is not cosmetic. `--max-abs` and its
relatives are a *hard* guarantee, and the correction layer establishes it by computing
residuals against the decoder's own reconstruction. That only works if encoder and
decoder reconstruct identically. Within one build they do, and the bound holds exactly:

| bound | max error, same build | max error, encode strict / decode fast |
|---|---|---|
| 0.5 (lossless) | 0 | **1** (84 voxels of 134M) |
| 2 | 2 | **3** (9 voxels) |

So across builds the declared bound is exceeded, rarely and by ~1 LSB, but exceeded. A
bound that is occasionally wrong is not a bound. Two honest positions, and the choice is
per workflow rather than global:

- **Lossy archives** — use the default. The guarantee that matters is statistical and
  fast math does not move it.
- **Bounded-error or lossless archives that outlive the build that wrote them** — build
  with `-DGPUDCT_STRICT_FP=ON`. It restores exact reproducibility, and the 25% decode is
  the price of a guarantee that is actually a guarantee.

The conformance corpus (§4.4) is what would catch a regression here, and it is now
required rather than optional: with a non-reproducible encoder, "old archives still
decode correctly" can only be tested against archives that are genuinely old.

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

### Encoder-side pre-filtering loses, and the reason generalizes

`--denoise` was on the roadmap on 3ddct's argument that scroll CT is noisy, a transform
codec spends real bits coding that noise, and removing it first should improve both ratio
and apparent quality. Tested on real full-resolution PHerc by filtering the volume,
compressing the filtered version, and scoring against the **original** — scoring against
the filtered volume would credit the filter for its own distortion, which is how a
pre-filter can be made to look arbitrarily good.

| pre-filter | ratio | PSNR | 3D-SSIM |
|---|---|---|---|
| none | 37.88× | 40.41 | 0.97421 |
| gauss σ=0.4 | 39.35× | 40.06 | 0.97245 |
| gauss σ=0.6 | 45.27× | 38.43 | 0.96208 |
| gauss σ=0.8 | 50.78× | 36.91 | 0.94936 |

The ratio gains are real — σ=0.8 buys 34% — but they are never worth their cost.
Interpolated to a matched 45× the unfiltered codec gives ≈39.5 dB and 0.967 SSIM against
σ=0.6's 38.43 dB and 0.962. The same holds at every rate tested from 22× to 160×, and on
SSIM as well as PSNR, so it is not an artifact of PSNR's known hostility to denoising.

The reason is worth stating because it applies to any fixed pre-filter: the quantizer is
already a rate-distortion-optimal denoiser. It discards high-frequency content in the
order that costs the least distortion per bit saved, having seen the actual coefficients.
A Gaussian discards it in a fixed order chosen in advance. Doing the codec's job for it,
worse, and then also paying for the residual, cannot come out ahead.

What this does *not* rule out is an **edge-preserving** pre-filter — bilateral, NLM,
anisotropic diffusion. Those remove noise while leaving the fibre boundaries the
transform codes efficiently anyway, which is a different proposition from an isotropic
blur. That remains untested, and it is the only version of this idea still worth trying.

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

### DeviceVolume: the archive stays in VRAM

The ordinary decode entry points are the wrong shape for a viewer that keeps a
scroll compressed in device memory. Every call uploads the archive, allocates
scratch, expands entropy tables, parses every brick header on the host, and
copies the result back to system memory -- and for a batch of eight bricks that
fixed cost *is* the time, while the copy back is pure waste when the consumer is
a renderer on the same device.

`DeviceVolume` pays it once at open. Measured on a 512^3 volume at P=64, 20
iterations after a warm-up, decoding into device memory with no readback:

| batch | latency | rate |
|---|---|---|
| 1 brick (2 MiB) | 5.52 ms | 381 MB/s |
| 8 bricks (16 MiB) | 5.74 ms | 2.93 GB/s |
| 32 bricks (64 MiB) | 6.45 ms | **10.4 GB/s** |

The shape is the whole story: latency is nearly flat in batch size, because it is
the K1 serial chain (`512/P` chunks per thread) and almost nothing else. A caller
should therefore batch as much as it can -- one brick and thirty-two cost the
same 6 ms.

Against where this started, for the eight-brick case a viewer actually issues:

| | 8 bricks |
|---|---|
| session start (P=16, flat K1, decode to host) | 44 ms |
| + P=64 and shared-memory slot tables | 17.9 ms |
| + DeviceVolume (resident archive, no readback) | **5.7 ms** |

**A scratch bug worth recording.** The first version sized the sparse coefficient
buffers for the format's worst case, 4096 nonzeros per chunk. That is 8.4 MB of
device memory per brick of batch, and the reported footprint came out at 526 MiB
for a volume whose *compressed* form is 10 MiB. On a class whose entire purpose
is to make VRAM go further, the scratch dwarfing the archive is not a tuning
detail, it is the feature failing. Sizing for typical occupancy and growing once
on overflow -- which the buffers being persistent makes cheap -- brought it to
62.5 MiB with no change in latency.

That number is the one to quote: a 100 GB scroll at 13x holds ~7.7 GB of
compressed archive in VRAM plus ~50 MiB of decode scratch.

### `streams_per_brick` is a budget, not a constant

The section above moved the default from 4 to 16 and left it there, on the basis
that P costs ratio and buys parallelism. Both halves of that are right; what was
missing is that the exchange rate is not a constant, and a fixed default is
therefore wrong on most archives.

The GPU entropy kernel runs one thread per (brick, stream). A 512^3 volume is 64
bricks, so at P=16 the whole kernel is 1024 threads -- and it is 92% of GPU
decode time, with the inverse transform at 1.8 ms and everything else in the
noise. Raising P is the only lever on it that does not change the format.

Measured on 512^3 scroll, decode-side, entropy kernel only:

| P | q=1 archive | k1 (entropy) |
|---|---|---|
| 8 | 26,000,587 | 1031.8 ms |
| 16 | 26,009,787 | 498.1 ms |
| 32 | 26,028,144 | 171.6 ms |
| 64 | 26,065,081 | **107.0 ms** |

4.65x for +0.213%. But run the same sweep on a sparser archive and the second
column behaves completely differently:

| volume, quality | mean brick payload | 16 -> 64 size | 16 -> 64 bytes |
|---|---|---|---|
| scroll512 q=4 | 819 KB | +0.106% | +864 B/brick |
| scroll512 q=1 | 406 KB | +0.213% | +864 B/brick |
| scroll512 q=0.25 | 120 KB | +0.718% | +864 B/brick |
| scrollL0 q=4 | 269 KB | +0.321% | +864 B/brick |
| scrollL0 q=1 | 96 KB | +0.901% | +864 B/brick |
| scrollL0 q=0.25 | 30 KB | **+2.844%** | +864 B/brick |

The last column is the whole story. The absolute cost is **identical to the
byte** across a 27x span of archive size, because a stream's cost is fixed and
has nothing to do with what it carries: a 16-byte rANS flush (`kRansStates`
32-bit states) plus a 4-byte entry in the brick's size table. 864 bytes is
48 extra streams at 18 bytes -- 20 gross, less two recovered because shorter
streams renormalize slightly less. Only the denominator moves.

So the same change is a bargain at 1835 ms saved per percent of ratio on dense
data and a bad deal at 12 ms per percent on sparse data, and no single default
can be right for both. `streams_per_brick = 0`, now the default, makes the
encoder decide: probe four bricks, measure the mean payload, and take the largest
P whose extra bytes stay under 0.35% of it. Because the cost is exactly linear in
P, the probe needs no second encode -- it measures the payload at P=16 and the
model supplies the rest.

What it chooses, against the ground truth above:

| volume, quality | chosen P | realized cost | entropy kernel |
|---|---|---|---|
| scroll512 q=4 | 64 | +0.106% | 153.9 ms |
| scroll512 q=1 | 64 | +0.213% | 498 -> **106.4 ms** |
| scroll512 q=0.25 | 32 | +0.239% | 69.9 ms |
| scrollL0 q=4 | 64 | +0.321% | 61.4 ms |
| scrollL0 q=1 | 32 | +0.300% | 105 -> **35.6 ms** |
| scrollL0 q=0.25 | 16 | 0 | 41.9 ms |

Every realized cost is at or below 0.32%, P reaches 32 or 64 in five of six, and
the one case left at 16 is the one where raising it would have spent 0.95% to
save 28 ms. The floor is the old default, so the automatic path can only ever
raise P; an archive whose bricks cannot afford more streams comes out exactly as
it always did.

Two things the probe had to get right. It must not depend on the thread count,
or the archive would depend on the machine that wrote it -- so the sample size is
a constant, not a function of hardware concurrency, and `parallel_for` runs it
like any other pass. And it must not sample evenly: bricks are indexed x fastest,
so striding by `brick_count / nprobe` lands on the same (x, y) column in every z
layer, which on a cylindrical scroll is either always air or always dense.
Evenly spaced picks put the 120 KB case at P=16; a golden-ratio sequence, which
spreads across all three axes, puts it at 32 where it belongs.

The cost is encode time, and it is a whole extra round of brick encodes: +6% on a
512-brick volume, +37% on a 64-brick one, where four bricks is a small fraction
of the work but still a full round of latency on a many-core machine. A volume is
encoded once and decoded many times, and `--streams N` opts out entirely.

### GPU decode is slower than CPU decode, and that is not the point

Worth recording plainly, because the numbers invite the wrong conclusion. On
512^3 scroll at q=1, end to end:

| | decode |
|---|---|
| CPU (SIMD, all cores) | 0.13 s |
| CUDA, P=16 | 0.60 s |
| CUDA, P=64 | 0.23 s |

Even after 4.65x on the entropy kernel the GPU does not win on throughput. It
should not be expected to: rANS decoding is a serial byte-at-a-time walk with a
data-dependent table lookup per symbol, which is close to the worst possible
shape for a GPU and close to the best for an out-of-order core with a large
cache. Parallelism across streams is the only thing the device has, and the
format caps it at `bricks * P`.

What the CUDA path is for is decoding *into VRAM* without a PCIe round trip --
`DeviceVolume`, where the archive stays resident and the decoded voxels are
already where the renderer wants them -- and freeing the CPU while it happens.
Measured against those goals it is worth having. Measured as a throughput
accelerator it is not, and no amount of kernel tuning changes that; it would take
a different entropy layer.

### Isosurface displacement, measured

The metric walks every grid edge, finds where the original crosses the isovalue by
linear interpolation between the two samples, finds where the reconstruction crosses
the same edge, and reports the distance between them. Four numbers, and the last two
are the ones that matter:

- **along edge** — the raw shift of the crossing along the axis. This is an *upper
  bound* on how far the surface moved: a surface oblique to the axis reads a larger
  shift along the edge than it actually travelled.
- **along normal** — the same shift projected onto the surface normal, estimated from
  the original's gradient. This is the honest displacement.
- **bias** (the signed mean) — a surface that consistently moves one way is far worse
  than one that jitters, because bias accumulates over a traced sheet while jitter
  averages out. This is why a signed mean is reported alongside the absolute one.
- **topology change** — edges where exactly one of the two volumes crosses, over edges
  where either does. The surface appeared or vanished rather than moved, which breaks
  a trace instead of bending it.

The isovalue defaults to Otsu's threshold on the *original*, which on scroll CT lands
in the valley between air and papyrus, and is always reported. `--iso V` overrides it.

Measured on a 512³ real scroll volume, balanced profile, deblocking on. Otsu picked
77.70, crossing 15.6% of edges:

| quality | archive | PSNR | normal mean | normal p99 | bias | topology |
|---|---|---|---|---|---|---|
| 0.25 | 9.3 MB | 26.7 dB | 0.097 vx | 0.405 vx | **-0.0186** | 0.442 |
| 1 | 31.0 MB | 34.7 dB | 0.040 vx | 0.188 vx | -0.0066 | 0.182 |
| 4 | 62.5 MB | 45.7 dB | 0.011 vx | 0.060 vx | -0.0017 | 0.049 |

Two things worth recording. Displacement falls roughly with the quantizer, as it
should. But the **bias is consistently negative at every rate** — the surface moves
inward, toward the low side of the isovalue, and at q=0.25 by nearly 2% of a voxel on
average. That is a systematic direction, not noise, and it is exactly the failure mode
PSNR cannot see. It is small enough not to be alarming and consistent enough to be
real; whether it comes from the dead-zone quantizer (which biases coefficients toward
zero, and so biases reconstruction toward the local mean) has not been established.
That is the obvious hypothesis and it is untested.

The topology fraction at q=0.25 — 44% of surface edges gaining or losing a crossing —
is the number that should discourage using that rate for segmentation work, and it is
far more legible than "26.7 dB".
