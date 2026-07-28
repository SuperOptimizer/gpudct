// Deblocking filter behaviour (docs/DESIGN.md section 3.8).
//
// The filter is only worth having if it removes seams without eating real
// structure, so both halves of that claim are tested: blockiness must fall, and
// content that happens to sit on a chunk boundary must survive.

#include <cmath>

#include "gpudct/gpudct.hpp"
#include "gpudct/metrics.hpp"
#include "test.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct_test;

namespace {

struct Result {
  Metrics off, on;
};

Result run(const Volume& v, float quality, float strength = 1.0f) {
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  const std::vector<float> orig = from_typed(raw, DType::u8, v.data.size());

  EncodeOptions eo;
  eo.quality = quality;
  std::vector<std::uint8_t> archive;
  if (encode(raw.data(), v.dims, DType::u8, eo, archive) != Status::ok) return {};

  Result r;
  for (int pass = 0; pass < 2; ++pass) {
    DecodeOptions dopts;
    dopts.deblock = (pass == 1);
    dopts.deblock_strength = strength;
    std::vector<std::uint8_t> out;
    VolumeInfo info;
    if (decode(archive, dopts, out, info) != Status::ok) return {};
    const std::vector<float> dec = from_typed(out, DType::u8, v.data.size());
    Metrics m = compute_metrics(orig.data(), dec.data(), v.dims, DType::u8);
    (pass == 0 ? r.off : r.on) = m;
  }
  return r;
}

}  // namespace

TEST(deblocking_reduces_seams_on_smooth_content) {
  // A smooth ramp has no real structure at chunk boundaries, so every seam the
  // codec introduces there is pure artifact and the filter should remove it.
  const Volume v = ramp_volume({128, 128, 128}, 10.0f, 240.0f);
  const Result r = run(v, 0.25f);
  std::printf("       blockiness %.3f -> %.3f, ssim %.5f -> %.5f, psnr %.2f -> %.2f\n",
              r.off.blockiness, r.on.blockiness, r.off.ssim, r.on.ssim, r.off.psnr_range,
              r.on.psnr_range);
  CHECK(r.on.blockiness < r.off.blockiness);
  // Removing an artifact should not cost fidelity on content this smooth.
  CHECK(r.on.psnr_range >= r.off.psnr_range - 0.5);
}

TEST(deblocking_reduces_seams_on_scroll_like_content) {
  const Volume v = scroll_like_volume({128, 128, 128});
  const Result r = run(v, 0.25f);
  std::printf("       blockiness %.3f -> %.3f, ssim %.5f -> %.5f, psnr %.2f -> %.2f\n",
              r.off.blockiness, r.on.blockiness, r.off.ssim, r.on.ssim, r.off.psnr_range,
              r.on.psnr_range);
  CHECK(r.on.blockiness < r.off.blockiness);
}

// The filter must not fire where there is a genuine edge, whether or not that
// edge lands on a multiple of 16. This is the failure mode that would make it
// worse than useless.
TEST(deblocking_preserves_a_real_edge_on_a_chunk_boundary) {
  Volume v = make({128, 128, 128});
  for (std::uint32_t z = 0; z < 128; ++z)
    for (std::uint32_t y = 0; y < 128; ++y)
      for (std::uint32_t x = 0; x < 128; ++x) v.at(x, y, z) = (x < 64) ? 20.0f : 230.0f;

  // Encoded at high quality, so the step is reproduced nearly exactly and any
  // damage the filter does is visible rather than lost in quantization noise.
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions eo;
  eo.quality = 8.0f;
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, eo, archive) == Status::ok);

  DecodeOptions dopts;
  dopts.deblock = true;
  std::vector<std::uint8_t> out;
  VolumeInfo info;
  REQUIRE(decode(archive, dopts, out, info) == Status::ok);

  // The step sits exactly on the boundary at x = 64. It must still be a step.
  double worst = 0;
  for (std::uint32_t z = 0; z < 128; z += 7)
    for (std::uint32_t y = 0; y < 128; y += 7) {
      const std::size_t i = (static_cast<std::size_t>(z) * 128 + y) * 128;
      const double lo = out[i + 63], hi = out[i + 64];
      worst = std::max(worst, 210.0 - (hi - lo));
    }
  CHECK(worst < 12.0);
}

TEST(deblock_strength_zero_is_a_no_op) {
  const Volume v = scroll_like_volume({64, 64, 64});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions eo;
  eo.quality = 0.5f;
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, eo, archive) == Status::ok);

  std::vector<std::uint8_t> a, b;
  VolumeInfo ia, ib;
  DecodeOptions off, zero;
  zero.deblock = true;
  zero.deblock_strength = 0.0f;
  REQUIRE(decode(archive, off, a, ia) == Status::ok);
  REQUIRE(decode(archive, zero, b, ib) == Status::ok);
  CHECK(a == b);
}

TEST(deblocking_is_deterministic_across_thread_counts) {
  const Volume v = scroll_like_volume({160, 144, 130});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions eo;
  eo.quality = 0.5f;
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, eo, archive) == Status::ok);

  std::vector<std::uint8_t> a, b;
  VolumeInfo ia, ib;
  DecodeOptions o1, o8;
  o1.deblock = o8.deblock = true;
  o1.threads = 1;
  o8.threads = 8;
  REQUIRE(decode(archive, o1, a, ia) == Status::ok);
  REQUIRE(decode(archive, o8, b, ib) == Status::ok);
  CHECK(a == b);
}

TEST(deblocking_handles_volumes_smaller_than_a_chunk) {
  for (Dims d : {Dims{1, 1, 1}, Dims{3, 3, 3}, Dims{16, 16, 16}, Dims{17, 2, 33}}) {
    const Volume v = scroll_like_volume(d);
    const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
    EncodeOptions eo;
    std::vector<std::uint8_t> archive;
    REQUIRE(encode(raw.data(), d, DType::u8, eo, archive) == Status::ok);
    DecodeOptions dopts;
    dopts.deblock = true;
    std::vector<std::uint8_t> out;
    VolumeInfo info;
    REQUIRE(decode(archive, dopts, out, info) == Status::ok);
    CHECK_EQ(out.size(), raw.size());
  }
}

TEST_MAIN()
