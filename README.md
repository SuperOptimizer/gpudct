# gpudct

A lossy and bounded-error 3D DCT codec for massive volumetric data, built for
Vesuvius Challenge scroll CT and comparable microCT / MRI / simulation volumes.

Two structural units (`docs/DESIGN.md` §2):

```
brick = 128³ voxels   entropy-coding, I/O, and random-access unit
chunk =  16³ voxels   transform and quantization unit
```

A 128³ brick is exactly one chunk of the Vesuvius open-data Zarr arrays, so an
archive maps 1:1 onto the storage layout the data already uses.

## Build

Needs CMake ≥ 3.28, Ninja, and a C++23-or-later compiler (developed against
clang 22 / MSVC 14.51; the code targets C++26 and falls back cleanly).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest
```

Options: `-DGPUDCT_ENABLE_CUDA=ON` (needs a CUDA toolkit; RTX 50xx is sm_120 and
wants CUDA 13+), `-DGPUDCT_ENABLE_SIMD=ON/OFF`, `-DGPUDCT_WERROR=ON`.

## Use

```sh
# raw volume in, archive out
gpudct compress scroll.raw scroll.gdct --dims 512,512,512 --dtype u8 --quality 0.5

gpudct decompress scroll.gdct out.raw --deblock
gpudct inspect scroll.gdct

# rate-distortion sweep with the full quality metric set
gpudct eval scroll.raw --dims 512,512,512 --dtype u8

# compare two volumes on every metric, plus the worst bricks
gpudct metrics orig.raw decoded.raw --dims 512,512,512 --dtype u8
```

Dtypes: `u8 s8 u16 s16 u32 s32 f32`. No 64-bit numeric types, by design.

### Error bounds

The feature that separates this from a media codec. Bounds are enforced by a
correction layer that clips the error tail, and they are checked exhaustively in
`tests/test_bounds.cpp` — every voxel, every corpus volume.

```c++
EncodeOptions opts;
opts.bounds.max_abs = 2.0f;   // hard: no voxel is ever off by more than 2
opts.bounds.p99     = 4.0f;   // percentile: 99% within 4, tail clipped only as needed
```

A percentile bound costs roughly 2.3× less than the equivalent hard bound,
because it only has to correct the voxels that actually breach the budget.

## Where it stands

Implemented and tested: the scalar reference codec, the float Lee-factorized
transform, static-model rANS with division-free encoding, entropy tables trained
on real scroll data, the raw-brick fallback, the bounded-error layer, the
deblocking filter, the full metric suite, and a bit-identical SIMD transform.
63 tests across 6 binaries, all green.

The CUDA decode backend is implemented and validated against the CPU reference
on an RTX 5080 (sm_120, CUDA 13.3): agreement is byte-identical on most volumes
and within 1 LSB on real scroll data, with under 1e-5 of voxels differing.

The GPU and CPU backends are at full feature parity, per-brick entropy tables
included. Under plain quantization the two encoders produce byte-identical
archives; under RDO they produce archives of identical size that decode to the
same voxels, and the difference is worth understanding -- see below.

Not done: DC-plane prediction (measured worthless -- DC is 0.1-0.5% of coded
bits), the progressive profile, Zarr integration, and comparisons against
ZFP / SZ3.

Measured on a real full-resolution 128³ brick from `PHerc0332`, balanced profile
(one brick = one Zarr chunk of the open dataset):

| quality | ratio | PSNR | 3D-SSIM | p99 err | max err |
|---|---|---|---|---|---|
| 0.125 | 51.4× | 31.1 dB | 0.925 |  20 | 57 |
| 0.25  | 31.1× | 34.7 dB | 0.963 |  13 | 36 |
| 0.5   | 19.5× | 38.4 dB | 0.983 |   8 | 25 |
| 1.0   | 12.5× | 42.1 dB | 0.992 |   5 | 16 |
| 2.0   |  8.1× | 45.6 dB | 0.996 |   3 | 12 |

Sampling resolution matters more than anything else here. The same codec at
quality 0.5 gets 19.5× on a full-resolution brick and 5.7× on a level-3
(8× downsampled) brick of the same scroll: downsampling has already removed the
redundancy the transform feeds on. Benchmark on the resolution you intend to
store.

Throughput on a 512³ volume (quality 0.5, 16-core CPU / RTX 5080):

| path | encode | decode |
|---|---|---|
| CPU, all threads | 2639 MB/s | 1000–1258 MB/s |
| CUDA (`--streams 32`), warm | — | 560–709 MB/s |
| CUDA, first call in a process | — | ~49 MB/s |

The cold-start row is not a rounding detail: creating a CUDA context and JITting
the kernels costs one to two seconds, which swamps a single decode entirely. The
GPU path is worth using from a long-running process that decodes many volumes,
and is a pessimization for one-shot CLI use.

The GPU does not beat the CPU here, and the cause is structural: the bitstream's
parse is data-dependent, so each rANS stream must be decoded serially by one
thread, and total device parallelism is `bricks × streams_per_brick`. Use
`--streams 16` or higher for GPU-targeted archives — it is a 5× GPU speedup for
1.3 % of ratio. `docs/DESIGN.md` §3.4 explains the constraint and what a real
fix would cost.


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

## GPU backend

Both encode and decode run on the GPU. Encode output is **byte-identical** to the
CPU encoder; decode matches within 1 LSB (the transform is `f32`, so the two
backends round a handful of voxels differently -- see docs/QUALITY.md 4.2).

Measured on an RTX 5080 laptop / Core Ultra 9 275HX, a 1 GiB volume (1024^3),
quality 0.5, `--streams 16`, best of 3. Driver startup is excluded from the
timer and reported separately: creating a CUDA context costs ~1.2 s once per
process, which a library caller pays once and which otherwise swamps any
single-volume measurement.

| path | encode | decode |
|---|---|---|
| CUDA, default | 2.06 GB/s | **3.21 GB/s** |
| CUDA, `--effort high` | **2.13 GB/s** | **2.98 GB/s** |
| CPU, default | **2.46 GB/s** | 2.03 GB/s |
| CPU, `--effort high` | 1.61 GB/s | 1.91 GB/s |

Decode is measured through `decode_into` on a reused buffer, not `decode` on a
fresh `std::vector`. The difference is not small: `std::vector::resize`
value-initializes, so a fresh 1 GiB output costs a full zero-fill plus page
faults -- measured at more than every other stage of decode put together. That
cost is real for a caller who needs a new buffer, but it is allocator behaviour
rather than codec throughput, and the API offers `decode_into` for callers who
can reuse one. Decode numbers taken before this distinction was drawn are not
comparable to these.

GPU decode is ahead in every configuration; GPU encode is ahead at `--effort
high` (where RDO costs the CPU more than the GPU) and behind at default effort,
because per-brick tables need a round trip to the host for clustering before the
range-coding kernel can start -- that clustering is, by a wide margin, the
largest single cost in GPU encode.

The kernel row is measured with CUDA events and is stable. The host-delivered
rows are wall-clock and vary by more than 2x run to run on this machine: writing
1 GiB outputs repeatedly fills the page cache, and the next run's 1 GiB
allocation then pays for reclaim. The CPU figures are also depressed by whatever
else the machine is doing. Treat the wall-clock rows as ranges, not numbers.

The kernel row is the one that matters for anything keeping its data on the
device -- rendering, ML inference. For host delivery, decode moves 156 ms of its
~500 ms in the readback against 31 ms of kernels: **it is PCIe- and
memory-bound, not compute-bound.**

### Scale matters more than anything else

Device parallelism is `bricks x streams_per_brick`, so a small volume starves the
GPU completely:

| volume | bricks | GPU threads | GPU encode | GPU decode |
|---|---|---|---|---|
| 256^3  |   8 |   256 |    6 MB/s |   65 MB/s |
| 512^3  |  64 |  2048 |   91 MB/s |  420 MB/s |
| 1024^3 | 512 | 16384 | 2121 MB/s | 1748 MB/s |

Benchmark the GPU on at least a few hundred bricks, or the number means nothing.
`--streams 16` is the sweet spot at that scale; 64 costs ratio without buying
throughput.

### Why RDO archives are not byte-identical across backends

Under plain quantization the two encoders agree byte for byte. Under RDO they do
not, and the mechanism is amplification rather than error: an RDO decision is a
float comparison, so a coefficient sitting exactly on the boundary between two
levels can fall either way on GPU and CPU. Before per-brick tables that cost one
symbol. Now it changes the brick's symbol histogram, which changes its clustered
tables, which changes every byte coded after it.

The archives stay the same size to within a rounding error and decode to the same
voxels, which is what the tests assert. It is worth knowing about because it
makes byte-comparison useless as a debugging tool for RDO archives -- compare
sizes and reconstructions instead.

### Rate-distortion optimization

`--effort high` is worth -2.8% BD-rate on PSNR and -4.7% on 3D-SSIM against plain
quantization, which is most of the margin over 3ddct. Two stages, both parallel:
per-coefficient level choice, and a per-sub-block decision to drop the whole
sub-block -- the second matters more, because zeroing a sub-block also removes
its 64-bit significance mask, a saving the per-coefficient stage cannot see.

The Lagrange scale was measured rather than assumed:

| lambda scale | 0.04 | 0.08 | **0.12** | 0.20 | 0.35 | 0.60 |
|---|---|---|---|---|---|---|
| BD-rate (PSNR) | -2.06% | -2.70% | **-2.81%** | -1.85% | +1.89% | +8.94% |

RDO costs the CPU about 25% of its encode throughput and the GPU almost nothing,
because it is per-sub-block parallel: at `--effort high` the GPU encoder is
*faster* than the CPU one (1854 vs 1782 MB/s on a 1 GiB volume).

### What actually moved the numbers

Mostly not the things that looked like the bottleneck:

- **Right-sizing scratch, not tuning kernels.** Both paths allocated for the
  worst case -- 4096 nonzero coefficients per chunk on decode, 6144 symbols per
  chunk on encode. At 512 bricks that is 6.4 GB of `cudaMalloc` feeding kernels
  that run in 54 ms. Sizing for typical occupancy with a retry on overflow was
  the single largest encode win.
- **Pinning both directions.** A pageable PCIe copy runs at about half rate.
  Pinning the readback halved it; pinning the upload took encode from 1896 to
  2100 MB/s.
- **Moving the brick crop onto the GPU**, which removed ~1M short host memcpys.
- **Parallelising symbol generation per chunk** rather than per stream: 2048
  threads scanning 4096 coefficients twice became 32768 threads scanning once,
  worth 3.2x on encode.

And two that were wrong:

- Compacting encode output on device cut PCIe traffic from 268 MB to 7 MB and
  moved throughput not at all. It stayed in -- it cuts memory pressure and allows
  larger batches -- but it bought no speed. The serial per-stream rANS is the
  limit, not the bus.
- Removing a redundant 1 GiB zero-fill of the output buffer made decode *slower*,
  1723 -> 971 MB/s. The zero-fill was accidentally serving as a page pre-fault:
  without it, pinning a gigabyte of cold pages cost 255 ms instead of 3, and the
  DMA then faulted its way through at a quarter speed (428 ms against 93). The
  first touch has to happen somewhere. Doing it deliberately, across threads,
  costs 30 ms and is the reason `decode_into` exists -- so a caller can own the
  allocation and reuse it, paying that once rather than per volume.

## Documentation

- `docs/DESIGN.md` — format, algorithms, and the decisions that were rejected
- `docs/QUALITY.md` — what "quality" is measured to mean, and the test strategy
- `docs/ROADMAP.md` — milestones and current status

## License

MIT. See `LICENSE`.
