// Cross-backend equivalence for the CUDA decoder (docs/QUALITY.md section 4.2).
//
// The transform is f32, so the GPU is not required to be byte-identical to the
// CPU -- float addition is not associative and the two use different operation
// orders. It *is* required to agree within a stated tolerance, and the entropy
// layer beneath it is required to be exact, because that part is pure integer
// arithmetic and has no excuse.
//
// The whole file no-ops when the build has no CUDA backend or the machine has no
// device, so it is safe to run everywhere.

#include <algorithm>
#include <cmath>

#include "gpudct/device_volume.hpp"
#include "gpudct/gpudct.hpp"
#include "gpudct/metrics.hpp"
#include "test.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct_test;

namespace {

bool cuda_ready() { return backend_available(Backend::cuda); }

struct Agreement {
  double max_abs = 0;
  double rms = 0;
  double differing_fraction = 0;
  bool ok = false;
};

Agreement compare_backends(const Volume& v, DType t, const EncodeOptions& opts) {
  Agreement a;
  const std::vector<std::uint8_t> raw = to_typed(v, t);
  std::vector<std::uint8_t> archive;
  if (encode(raw.data(), v.dims, t, opts, archive) != Status::ok) return a;

  // Deblocking off, deliberately. It is not in the normative reconstruction path
  // (docs/DESIGN.md 3.8) and it is a *conditional* filter, so a legal 1-LSB
  // difference between backends can land either side of a threshold and make the
  // filter fire on one and not the other. That turns a 1-LSB divergence into one
  // the size of the filter's clip -- measured at 5 LSB on u16 once deblocking
  // became the default. Amplified threshold noise is not a backend disagreement,
  // and testing through it would hide the thing this test exists to catch.
  DecodeOptions dopts;
  dopts.deblock = false;

  std::vector<std::uint8_t> cpu, gpu;
  VolumeInfo ic, ig;
  if (decode(archive, dopts, cpu, ic, Backend::cpu_scalar) != Status::ok) return a;
  if (decode(archive, dopts, gpu, ig, Backend::cuda) != Status::ok) return a;
  if (cpu.size() != gpu.size()) return a;

  const std::vector<float> fc = from_typed(cpu, t, v.data.size());
  const std::vector<float> fg = from_typed(gpu, t, v.data.size());

  double sq = 0;
  std::size_t diff = 0;
  for (std::size_t i = 0; i < fc.size(); ++i) {
    const double e = std::fabs(static_cast<double>(fc[i]) - fg[i]);
    a.max_abs = std::max(a.max_abs, e);
    sq += e * e;
    if (e != 0.0) ++diff;
  }
  a.rms = std::sqrt(sq / static_cast<double>(fc.size()));
  a.differing_fraction = static_cast<double>(diff) / static_cast<double>(fc.size());
  a.ok = true;
  return a;
}

}  // namespace

TEST(cuda_decode_agrees_with_cpu_within_tolerance) {
  if (!cuda_ready()) {
    std::printf("       (no CUDA device; skipped)\n");
    return;
  }
  // A high stream count so the GPU has threads to work with; the agreement
  // itself does not depend on it.
  for (std::uint8_t p : {std::uint8_t{4}, std::uint8_t{16}, std::uint8_t{32}}) {
    for (float q : {0.25f, 1.0f, 4.0f}) {
      const Volume v = scroll_like_volume({256, 192, 160});
      EncodeOptions opts;
      opts.quality = q;
      opts.streams_per_brick = p;
      const Agreement a = compare_backends(v, DType::u8, opts);
      REQUIRE(a.ok);
      std::printf("       P=%2u q=%.2f  max %.0f  rms %.5f  differing %.2e\n", p,
                  static_cast<double>(q), a.max_abs, a.rms, a.differing_fraction);
      // A voxel off by 2 or more cannot be explained by float non-associativity.
      CHECK(a.max_abs <= 1.0);
      CHECK(a.differing_fraction < 1e-4);
      CHECK(a.rms < 0.05);
    }
  }
}

TEST(cuda_handles_every_dtype) {
  if (!cuda_ready()) return;
  const Volume base = scroll_like_volume({144, 144, 144});
  for (DType t : {DType::u8, DType::s8, DType::u16, DType::s16, DType::f32}) {
    Volume v = base;
    const float hi = std::min(dtype_max(t), 30000.0f);
    const float lo = std::max(dtype_min(t), -30000.0f);
    for (float& f : v.data) f = lo + (f / 255.0f) * (hi - lo);

    EncodeOptions opts;
    opts.quality = 1.0f;
    opts.streams_per_brick = 16;
    const Agreement a = compare_backends(v, t, opts);
    REQUIRE(a.ok);
    // One LSB of the dtype, which for f32 is measured against its own scale.
    const double lsb = (t == DType::f32) ? (hi - lo) / 65535.0 * 2.0 : 1.0;
    if (a.max_abs > lsb)
      std::printf("       dtype %s max %.6f (lsb %.6f)\n", std::string(dtype_name(t)).c_str(),
                  a.max_abs, lsb);
    CHECK(a.max_abs <= lsb);
  }
}

// The equivalence tests above deliberately bypass deblocking, so this is what
// keeps the default path covered. The bar is looser and it has to be: the filter
// is conditional, so where the two backends differ by a legal 1 LSB the filter
// can fire on one and not the other, and the resulting difference is bounded by
// its clip rather than by the reconstruction difference. What must still hold is
// that this stays rare and bounded -- a systematic divergence would show up as a
// large differing fraction, not as a handful of amplified voxels.
TEST(cuda_deblocked_output_agrees_within_the_filter_bound) {
  if (!cuda_ready()) return;
  const Volume v = scroll_like_volume({144, 144, 144});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions opts;
  opts.quality = 1.0f;
  opts.streams_per_brick = 16;
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::ok);

  std::vector<std::uint8_t> cpu, gpu;
  VolumeInfo ic, ig;
  REQUIRE(decode(archive, DecodeOptions{}, cpu, ic, Backend::cpu_scalar) == Status::ok);
  REQUIRE(decode(archive, DecodeOptions{}, gpu, ig, Backend::cuda) == Status::ok);
  REQUIRE(cpu.size() == gpu.size());

  double maxd = 0;
  std::size_t diff = 0;
  for (std::size_t i = 0; i < cpu.size(); ++i) {
    const double e = std::fabs(static_cast<double>(cpu[i]) - static_cast<double>(gpu[i]));
    maxd = std::max(maxd, e);
    if (e != 0.0) ++diff;
  }
  const double frac = static_cast<double>(diff) / static_cast<double>(cpu.size());
  std::printf("       deblocked: max %.0f  differing %.2e\n", maxd, frac);
  CHECK(maxd <= 8.0);
  CHECK(frac < 1e-3);
}

// Dimensions that leave partial bricks exercise the crop path in the CUDA host
// code, which is separate from the CPU one and easy to get wrong by a row.
TEST(cuda_handles_partial_bricks) {
  if (!cuda_ready()) return;
  for (Dims d : {Dims{130, 66, 34}, Dims{129, 129, 129}, Dims{1, 1, 1}, Dims{200, 40, 300}}) {
    const Volume v = scroll_like_volume(d);
    EncodeOptions opts;
    opts.quality = 1.0f;
    opts.streams_per_brick = 16;
    const Agreement a = compare_backends(v, DType::u8, opts);
    REQUIRE(a.ok);
    CHECK(a.max_abs <= 1.0);
  }
}

// An archive carrying a bounded-error correction layer must not be decoded by a
// path that ignores corrections. The GPU declines it and the caller falls back,
// so the bound still holds.
TEST(cuda_falls_back_rather_than_breaking_a_bound) {
  if (!cuda_ready()) return;
  const Volume v = scroll_like_volume({160, 160, 160});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions opts;
  opts.quality = 0.25f;
  opts.bounds.max_abs = 2.0f;
  opts.streams_per_brick = 16;
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::ok);

  std::vector<std::uint8_t> out;
  VolumeInfo info;
  REQUIRE(decode(archive, DecodeOptions{}, out, info, Backend::cuda) == Status::ok);

  const std::vector<float> orig = from_typed(raw, DType::u8, v.data.size());
  const std::vector<float> dec = from_typed(out, DType::u8, v.data.size());
  double worst = 0;
  for (std::size_t i = 0; i < orig.size(); ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(orig[i]) - dec[i]));
  CHECK(worst <= 2.0);
}

// Incompressible regions become raw bricks, which the GPU path does not decode.
// The fallback must still produce the right answer.
TEST(cuda_falls_back_on_raw_bricks) {
  if (!cuda_ready()) return;
  const Volume v = noise_volume({128, 128, 128}, 0.0f, 255.0f);
  EncodeOptions opts;
  opts.quality = 16.0f;  // past the point where storing beats coding
  opts.streams_per_brick = 16;
  const Agreement a = compare_backends(v, DType::u8, opts);
  REQUIRE(a.ok);
  CHECK_EQ(a.max_abs, 0.0);
}

// Rate-distortion optimized quantization runs on both backends and must reach
// the same decisions.
//
// Per-brick entropy tables are disabled here on purpose. The device encoder does
// not build them -- its kernels code against one uploaded table set -- so with
// them on, the two encoders differ in entropy modelling rather than in RDO, and
// this test would be measuring the wrong thing. The size gap that creates is
// pinned separately below.
TEST(cuda_rdo_matches_cpu_rdo) {
  if (!cuda_ready()) return;
  const Volume v = scroll_like_volume({256, 192, 160});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);

  for (float q : {0.25f, 1.0f}) {
    EncodeOptions opts;
    opts.quality = q;
    opts.effort = Effort::high;
    opts.streams_per_brick = 16;
    opts.per_brick_tables = false;

    std::vector<std::uint8_t> cpu, gpu;
    REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, cpu, Backend::cpu_scalar) == Status::ok);
    REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, gpu, Backend::cuda) == Status::ok);

    const double dsize = std::fabs(static_cast<double>(cpu.size()) - gpu.size()) /
                         static_cast<double>(cpu.size());
    std::printf("       q=%.2f  cpu %zu B  gpu %zu B  (%.4f%% apart)\n",
                static_cast<double>(q), cpu.size(), gpu.size(), dsize * 100.0);
    CHECK(dsize < 1e-3);

    // Both must decode to the same reconstruction as well.
    std::vector<std::uint8_t> dc, dg;
    VolumeInfo ic, ig;
    REQUIRE(decode(cpu, DecodeOptions{}, dc, ic, Backend::cpu_scalar) == Status::ok);
    REQUIRE(decode(gpu, DecodeOptions{}, dg, ig, Backend::cpu_scalar) == Status::ok);
    double worst = 0;
    for (std::size_t i = 0; i < dc.size(); ++i)
      worst = std::max(worst, std::fabs(static_cast<double>(dc[i]) - dg[i]));
    CHECK(worst <= 1.0);
  }
}

// Encode parity, per-brick tables included.
//
// The device encoder now builds per-brick tables the same way the CPU one does:
// it histograms the symbols on the device, clusters on the host, and uploads the
// tables before the range-coding kernel.
//
// Plain quantization is byte-identical between the backends. RDO is not, and the
// reason is worth stating: an RDO decision is a float comparison, so a
// coefficient sitting exactly on the boundary between two levels can fall either
// way. Before per-brick tables that cost at most one symbol. Now it changes the
// brick's symbol histogram, which changes its clustered tables, which changes
// every byte coded after it -- so a single tie flips the whole archive while
// leaving its size essentially unchanged. That is amplification, not error: what
// must hold is that both archives are the same size and decode to the same
// voxels, which is what this checks.
TEST(cuda_encode_matches_cpu) {
  if (!cuda_ready()) return;
  const Volume v = scroll_like_volume({256, 192, 160});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);

  for (float q : {0.25f, 1.0f, 4.0f}) {
    EncodeOptions opts;
    opts.quality = q;
    opts.streams_per_brick = 16;

    // Plain quantization: identical bytes.
    std::vector<std::uint8_t> cpu, gpu;
    REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, cpu, Backend::cpu_scalar) == Status::ok);
    REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, gpu, Backend::cuda) == Status::ok);
    if (cpu != gpu)
      std::printf("       q=%.2f: cpu %zu B, gpu %zu B\n", static_cast<double>(q), cpu.size(),
                  gpu.size());
    CHECK(cpu == gpu);

    // RDO: same size, same reconstruction.
    opts.effort = Effort::high;
    std::vector<std::uint8_t> cpu_r, gpu_r;
    REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, cpu_r, Backend::cpu_scalar) ==
            Status::ok);
    REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, gpu_r, Backend::cuda) == Status::ok);
    const double dsize = std::fabs(static_cast<double>(cpu_r.size()) - gpu_r.size()) /
                         static_cast<double>(cpu_r.size());
    std::printf("       q=%.2f rdo: cpu %zu B, gpu %zu B (%.4f%% apart)\n",
                static_cast<double>(q), cpu_r.size(), gpu_r.size(), dsize * 100.0);
    CHECK(dsize < 2e-3);

    std::vector<std::uint8_t> dc, dg;
    VolumeInfo ic, ig;
    REQUIRE(decode(cpu_r, DecodeOptions{}, dc, ic, Backend::cpu_scalar) == Status::ok);
    REQUIRE(decode(gpu_r, DecodeOptions{}, dg, ig, Backend::cpu_scalar) == Status::ok);
    REQUIRE(dc.size() == dg.size());
    double worst = 0;
    for (std::size_t i = 0; i < dc.size(); ++i)
      worst = std::max(worst, std::fabs(static_cast<double>(dc[i]) - dg[i]));
    CHECK(worst <= 2.0);
  }
}

// And the archives those produce decode identically on the device, per-brick
// tables and all.
TEST(cuda_decode_handles_per_brick_tables) {
  if (!cuda_ready()) return;
  const Volume v = scroll_like_volume({256, 192, 160});
  EncodeOptions opts;
  opts.quality = 0.5f;
  opts.effort = Effort::high;
  opts.streams_per_brick = 16;
  opts.per_brick_tables = true;

  const Agreement a = compare_backends(v, DType::u8, opts);
  REQUIRE(a.ok);
  std::printf("       max %.0f  rms %.5f  differing %.2e\n", a.max_abs, a.rms,
              a.differing_fraction);
  CHECK(a.max_abs <= 1.0);
  CHECK(a.differing_fraction < 1e-4);
}

TEST_MAIN()

// DeviceVolume: the compressed archive stays in VRAM and decoding writes into
// device memory. It must produce exactly what the ordinary GPU decoder does --
// it runs the same two kernels, so any difference is a bug in the persistent
// setup (table expansion, descriptor gathering, batch slicing), which is
// precisely the part that is new.
TEST(device_volume_matches_full_decode) {
  if (!cuda_ready()) return;
  const Volume v = scroll_like_volume({300, 260, 280});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions opts;
  opts.streams_per_brick = 64;  // the setting a VRAM-resident reader would use
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::ok);

  std::vector<std::uint8_t> full;
  VolumeInfo info;
  REQUIRE(decode(archive, DecodeOptions{}, full, info, Backend::cuda) == Status::ok);

  std::unique_ptr<DeviceVolume> dv;
  const Status os = DeviceVolume::open(archive, dv);
  if (os == Status::backend_unavailable) return;
  REQUIRE(os == Status::ok);
  REQUIRE(dv->brick_count() > 1);

  const Dims grid = dv->brick_dims();
  const std::size_t brick_vox = static_cast<std::size_t>(kBrickDim) * kBrickDim * kBrickDim;

  // Ask for the bricks out of order and with a repeat: the output is defined to
  // be dense in the order given, not in brick order, and nothing about the
  // decode may depend on the request being sorted or unique.
  std::vector<std::uint32_t> want;
  for (std::uint32_t i = static_cast<std::uint32_t>(dv->brick_count()); i-- > 0;)
    want.push_back(i);
  want.push_back(0);

  std::uint8_t* d_out = nullptr;
  REQUIRE(DeviceVolume::device_malloc(want.size() * brick_vox,
                                      reinterpret_cast<void**>(&d_out)) == Status::ok);
  REQUIRE(dv->decode_bricks(want, d_out) == Status::ok);
  std::vector<std::uint8_t> got(want.size() * brick_vox);
  REQUIRE(DeviceVolume::device_to_host(got.data(), d_out, got.size()) == Status::ok);
  DeviceVolume::device_free(d_out);

  for (std::size_t k = 0; k < want.size(); ++k) {
    const std::uint32_t bi = want[k];
    const std::uint32_t bx = static_cast<std::uint32_t>(bi % grid.x);
    const std::uint32_t by = static_cast<std::uint32_t>((bi / grid.x) % grid.y);
    const std::uint32_t bz = static_cast<std::uint32_t>(bi / (grid.x * grid.y));
    for (int z = 0; z < kBrickDim; ++z) {
      const std::uint32_t gz = bz * kBrickDim + static_cast<std::uint32_t>(z);
      if (gz >= v.dims.z) break;
      for (int y = 0; y < kBrickDim; ++y) {
        const std::uint32_t gy = by * kBrickDim + static_cast<std::uint32_t>(y);
        if (gy >= v.dims.y) break;
        for (int x = 0; x < kBrickDim; ++x) {
          const std::uint32_t gx = bx * kBrickDim + static_cast<std::uint32_t>(x);
          if (gx >= v.dims.x) break;
          const std::size_t li =
              k * brick_vox +
              (static_cast<std::size_t>(z) * kBrickDim + static_cast<std::size_t>(y)) * kBrickDim +
              static_cast<std::size_t>(x);
          const std::size_t fi =
              (static_cast<std::size_t>(gz) * v.dims.y + gy) * v.dims.x + gx;
          if (got[li] != full[fi]) {
            CHECK_EQ(static_cast<int>(got[li]), static_cast<int>(full[fi]));
            return;
          }
        }
      }
    }
  }
}
