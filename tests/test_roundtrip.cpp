#include <algorithm>
#include <cmath>

#include "gpudct/gpudct.hpp"
#include "test.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct_test;

namespace {

struct Stats {
  double mae = 0, rmse = 0, max_abs = 0, psnr = 0;
  double ratio = 0;
};

Stats compare(const std::vector<float>& a, const std::vector<float>& b, double peak,
              std::size_t raw_bytes, std::size_t coded_bytes) {
  Stats s;
  double sum = 0, sq = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double e = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    sum += e;
    sq += e * e;
    s.max_abs = std::max(s.max_abs, e);
  }
  const double n = static_cast<double>(a.size());
  s.mae = sum / n;
  s.rmse = std::sqrt(sq / n);
  s.psnr = (s.rmse > 0) ? 20.0 * std::log10(peak / s.rmse) : 1e9;
  s.ratio = static_cast<double>(raw_bytes) / static_cast<double>(coded_bytes);
  return s;
}

// Encodes, decodes, and returns the reconstruction alongside its statistics.
Stats round_trip(const Volume& v, DType t, const EncodeOptions& opts,
                 std::vector<float>* decoded_out = nullptr) {
  const std::vector<std::uint8_t> raw = to_typed(v, t);
  std::vector<std::uint8_t> archive;
  const Status es = encode(raw.data(), v.dims, t, opts, archive);
  if (es != Status::ok) {
    ::gpudct_test::fail(__FILE__, __LINE__, std::string("encode: ") +
                                                std::string(status_message(es)));
    return {};
  }

  std::vector<std::uint8_t> out;
  VolumeInfo info;
  const Status ds = decode(archive, DecodeOptions{}, out, info);
  if (ds != Status::ok) {
    ::gpudct_test::fail(__FILE__, __LINE__, std::string("decode: ") +
                                                std::string(status_message(ds)));
    return {};
  }
  if (!(info.dims == v.dims)) {
    ::gpudct_test::fail(__FILE__, __LINE__, "decoded dims differ");
    return {};
  }

  const std::vector<float> orig = from_typed(raw, t, v.data.size());
  const std::vector<float> dec = from_typed(out, t, v.data.size());
  if (decoded_out) *decoded_out = dec;

  const double peak = (t == DType::f32)
                          ? (*std::max_element(orig.begin(), orig.end()) -
                             *std::min_element(orig.begin(), orig.end()))
                          : static_cast<double>(dtype_max(t)) - dtype_min(t);
  return compare(orig, dec, peak, raw.size(), archive.size());
}

}  // namespace

// A constant volume has nothing but DC. If this is not near-exact, either the
// transform normalization or the dead zone is wrong.
//
// Sized to exactly one full brick: a smaller volume still pays for all 512
// chunks of its brick plus the rANS state flush, so its ratio measures padding
// overhead rather than compression. That overhead is real and worth knowing
// about, but it belongs in its own test, below.
TEST(constant_volume_survives_nearly_exactly) {
  const Volume v = constant_volume({128, 128, 128}, 100.0f);
  EncodeOptions opts;
  const Stats s = round_trip(v, DType::u8, opts);
  CHECK(s.max_abs <= 1.0);
  // 4096 identical voxels per chunk collapse to one DC coefficient.
  CHECK(s.ratio > 200.0);
}

// A volume much smaller than a brick pays for the whole brick anyway. This is
// the documented cost of making the brick the entropy unit (docs/DESIGN.md
// section 2); the test pins it so a regression shows up as a number, not a
// surprise.
TEST(sub_brick_volumes_pay_a_bounded_padding_overhead) {
  const Volume v = constant_volume({40, 33, 37}, 100.0f);
  EncodeOptions opts;
  const Stats s = round_trip(v, DType::u8, opts);
  std::printf("       padded ratio=%.2fx\n", s.ratio);
  CHECK(s.max_abs <= 1.0);
  // Loose on purpose. The entropy tables are trained on real scroll data, where
  // a perfectly constant chunk essentially never occurs, so this synthetic case
  // is off-distribution and its ratio is not something to optimize for. The
  // check exists to catch padding overhead exploding, not to pin a number.
  CHECK(s.ratio > 8.0);
}

TEST(zero_volume_survives_exactly) {
  const Volume v = constant_volume({20, 20, 20}, 0.0f);
  EncodeOptions opts;
  const Stats s = round_trip(v, DType::u8, opts);
  CHECK_EQ(s.max_abs, 0.0);
}

TEST(smooth_volume_round_trips_with_low_error) {
  const Volume v = ramp_volume({64, 48, 40}, 10.0f, 240.0f);
  EncodeOptions opts;
  const Stats s = round_trip(v, DType::u8, opts);
  CHECK(s.psnr > 35.0);
  CHECK(s.ratio > 5.0);
}

TEST(scroll_like_volume_round_trips) {
  const Volume v = scroll_like_volume({96, 80, 72});
  EncodeOptions opts;
  opts.quality = 2.0f;
  const Stats s = round_trip(v, DType::u8, opts);
  std::printf("       psnr=%.2f dB  mae=%.3f  max=%.0f  ratio=%.1fx\n", s.psnr, s.mae,
              s.max_abs, s.ratio);
  CHECK(s.psnr > 30.0);
  CHECK(s.ratio > 2.0);
}

// Dimensions that are not multiples of the chunk or brick size exercise the
// edge-replication path, which is where off-by-one bugs live.
TEST(non_multiple_dimensions_round_trip) {
  for (Dims d : {Dims{1, 1, 1}, Dims{17, 3, 129}, Dims{130, 127, 5}, Dims{16, 16, 16},
                 Dims{128, 128, 128}, Dims{129, 1, 1}}) {
    const Volume v = scroll_like_volume(d);
    EncodeOptions opts;
    opts.quality = 4.0f;
    const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
    std::vector<std::uint8_t> archive;
    REQUIRE(encode(raw.data(), d, DType::u8, opts, archive) == Status::ok);
    std::vector<std::uint8_t> out;
    VolumeInfo info;
    REQUIRE(decode(archive, DecodeOptions{}, out, info) == Status::ok);
    CHECK_EQ(out.size(), raw.size());
    CHECK(info.dims == d);
  }
}

TEST(every_dtype_round_trips) {
  const Volume base = scroll_like_volume({48, 40, 36});
  for (DType t : {DType::u8, DType::s8, DType::u16, DType::s16, DType::u32, DType::s32,
                  DType::f32}) {
    Volume v = base;
    // Rescale into each dtype's natural range so the test is measuring the
    // codec rather than saturation.
    const float lo = dtype_min(t) == -3.402823466e38f ? -1.0f : dtype_min(t);
    const float hi = dtype_max(t) == 3.402823466e38f ? 1.0f : dtype_max(t);
    const float span = std::min(hi - lo, 65535.0f);
    for (float& f : v.data) f = lo + (f / 255.0f) * span;

    EncodeOptions opts;
    opts.quality = 4.0f;
    const Stats s = round_trip(v, t, opts);
    if (s.psnr <= 25.0)
      std::printf("       dtype %s psnr=%.2f\n", std::string(dtype_name(t)).c_str(), s.psnr);
    CHECK(s.psnr > 25.0);
  }
}

TEST(higher_quality_means_lower_error_and_worse_ratio) {
  const Volume v = scroll_like_volume({64, 64, 64});
  EncodeOptions lo, hi;
  lo.quality = 0.5f;
  hi.quality = 8.0f;
  const Stats a = round_trip(v, DType::u8, lo);
  const Stats b = round_trip(v, DType::u8, hi);
  CHECK(b.psnr > a.psnr);
  CHECK(b.ratio < a.ratio);
}

TEST(profiles_order_as_designed) {
  const Volume v = scroll_like_volume({64, 64, 64});
  EncodeOptions a, b, c;
  a.profile = Profile::archival;
  b.profile = Profile::balanced;
  c.profile = Profile::viewing;
  const Stats sa = round_trip(v, DType::u8, a);
  const Stats sb = round_trip(v, DType::u8, b);
  const Stats sc = round_trip(v, DType::u8, c);
  // archival keeps the most detail and compresses least; viewing is the reverse.
  CHECK(sa.psnr > sb.psnr);
  CHECK(sb.psnr > sc.psnr);
  CHECK(sa.ratio < sb.ratio);
  CHECK(sb.ratio < sc.ratio);
}

TEST(stream_count_does_not_change_the_decoded_result) {
  const Volume v = scroll_like_volume({80, 64, 64});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  std::vector<float> reference;
  for (std::uint8_t p : {std::uint8_t{1}, std::uint8_t{2}, std::uint8_t{4}, std::uint8_t{8},
                         std::uint8_t{16}}) {
    EncodeOptions opts;
    opts.streams_per_brick = p;
    std::vector<std::uint8_t> archive;
    REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::ok);
    std::vector<std::uint8_t> out;
    VolumeInfo info;
    REQUIRE(decode(archive, DecodeOptions{}, out, info) == Status::ok);
    const std::vector<float> dec = from_typed(out, DType::u8, v.data.size());
    if (reference.empty()) reference = dec;
    else CHECK(reference == dec);
  }
}

// Threading must not affect the bytes produced or the voxels recovered.
TEST(thread_count_does_not_change_results) {
  const Volume v = scroll_like_volume({96, 80, 80});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  std::vector<std::uint8_t> single, multi;
  EncodeOptions o1, o8;
  o1.threads = 1;
  o8.threads = 8;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, o1, single) == Status::ok);
  REQUIRE(encode(raw.data(), v.dims, DType::u8, o8, multi) == Status::ok);
  CHECK(single == multi);

  std::vector<std::uint8_t> d1, d8;
  VolumeInfo i1, i8;
  DecodeOptions p1, p8;
  p1.threads = 1;
  p8.threads = 8;
  REQUIRE(decode(single, p1, d1, i1) == Status::ok);
  REQUIRE(decode(single, p8, d8, i8) == Status::ok);
  CHECK(d1 == d8);
}

// The brick is the random-access unit, so a single-brick fetch must agree with
// the corresponding region of a full decode.
TEST(single_brick_decode_matches_full_decode) {
  const Volume v = scroll_like_volume({200, 150, 140});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions opts;
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::ok);

  std::vector<std::uint8_t> full;
  VolumeInfo info;
  REQUIRE(decode(archive, DecodeOptions{}, full, info) == Status::ok);

  const Dims grid = brick_grid(v.dims);
  for (std::uint32_t bz = 0; bz < grid.z; ++bz)
    for (std::uint32_t by = 0; by < grid.y; ++by)
      for (std::uint32_t bx = 0; bx < grid.x; ++bx) {
        std::vector<std::uint8_t> brick;
        REQUIRE(decode_brick(archive, bx, by, bz, brick) == Status::ok);
        REQUIRE(brick.size() == static_cast<std::size_t>(kBrickDim) * kBrickDim * kBrickDim);
        for (int z = 0; z < kBrickDim; ++z) {
          const std::uint32_t gz = bz * kBrickDim + static_cast<std::uint32_t>(z);
          if (gz >= v.dims.z) break;
          for (int y = 0; y < kBrickDim; ++y) {
            const std::uint32_t gy = by * kBrickDim + static_cast<std::uint32_t>(y);
            if (gy >= v.dims.y) break;
            for (int x = 0; x < kBrickDim; ++x) {
              const std::uint32_t gx = bx * kBrickDim + static_cast<std::uint32_t>(x);
              if (gx >= v.dims.x) break;
              const std::size_t bi =
                  (static_cast<std::size_t>(z) * kBrickDim + static_cast<std::size_t>(y)) *
                      kBrickDim + static_cast<std::size_t>(x);
              const std::size_t fi =
                  (static_cast<std::size_t>(gz) * v.dims.y + gy) * v.dims.x + gx;
              if (brick[bi] != full[fi]) {
                CHECK_EQ(static_cast<int>(brick[bi]), static_cast<int>(full[fi]));
                return;
              }
            }
          }
        }
      }
}

// Sub-brick random access. A single-chunk fetch decodes only its stream's prefix
// rather than the whole brick, so this is the test that the prefix walk leaves
// the rANS state in exactly the place a full decode would have.
//
// Swept over streams_per_brick because P is what sets how the chunks are
// distributed across streams: P=1 puts every chunk in one stream (the prefix is
// the whole brick), P=16 spreads them, and a P that does not divide 512 evenly
// is the case where an off-by-one in the walk would hide.
TEST(single_chunk_decode_matches_full_decode) {
  const Volume v = scroll_like_volume({200, 150, 140});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);

  for (std::uint8_t P : {std::uint8_t{1}, std::uint8_t{4}, std::uint8_t{7}, std::uint8_t{16}}) {
    EncodeOptions opts;
    opts.streams_per_brick = P;
    std::vector<std::uint8_t> archive;
    REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::ok);

    std::vector<std::uint8_t> full;
    VolumeInfo info;
    REQUIRE(decode(archive, DecodeOptions{}, full, info) == Status::ok);

    const Dims cgrid{(v.dims.x + kChunkDim - 1) / kChunkDim,
                     (v.dims.y + kChunkDim - 1) / kChunkDim,
                     (v.dims.z + kChunkDim - 1) / kChunkDim};
    for (std::uint32_t cz = 0; cz < cgrid.z; ++cz)
      for (std::uint32_t cy = 0; cy < cgrid.y; ++cy)
        for (std::uint32_t cx = 0; cx < cgrid.x; ++cx) {
          std::vector<std::uint8_t> chunk;
          REQUIRE(decode_chunk(archive, cx, cy, cz, chunk) == Status::ok);
          REQUIRE(chunk.size() == static_cast<std::size_t>(kChunkVox));
          for (int z = 0; z < kChunkDim; ++z) {
            const std::uint32_t gz = cz * kChunkDim + static_cast<std::uint32_t>(z);
            if (gz >= v.dims.z) break;
            for (int y = 0; y < kChunkDim; ++y) {
              const std::uint32_t gy = cy * kChunkDim + static_cast<std::uint32_t>(y);
              if (gy >= v.dims.y) break;
              for (int x = 0; x < kChunkDim; ++x) {
                const std::uint32_t gx = cx * kChunkDim + static_cast<std::uint32_t>(x);
                if (gx >= v.dims.x) break;
                const std::size_t li =
                    (static_cast<std::size_t>(z) * kChunkDim + static_cast<std::size_t>(y)) *
                        kChunkDim + static_cast<std::size_t>(x);
                const std::size_t fi =
                    (static_cast<std::size_t>(gz) * v.dims.y + gy) * v.dims.x + gx;
                if (chunk[li] != full[fi]) {
                  CHECK_EQ(static_cast<int>(chunk[li]), static_cast<int>(full[fi]));
                  return;
                }
              }
            }
          }
        }
  }
}

// Out-of-range chunk coordinates must be rejected, not read out of bounds.
TEST(single_chunk_decode_rejects_bad_coordinates) {
  const Volume v = scroll_like_volume({64, 64, 64});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, EncodeOptions{}, archive) == Status::ok);

  std::vector<std::uint8_t> chunk;
  CHECK(decode_chunk(archive, 4, 0, 0, chunk) == Status::invalid_argument);
  CHECK(decode_chunk(archive, 0, 4, 0, chunk) == Status::invalid_argument);
  CHECK(decode_chunk(archive, 0, 0, 4, chunk) == Status::invalid_argument);
  CHECK(decode_chunk(archive, 0, 0, 0, chunk) == Status::ok);
}

TEST(inspect_reports_what_was_encoded) {
  const Volume v = scroll_like_volume({50, 60, 70});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u16);
  EncodeOptions opts;
  opts.profile = Profile::viewing;
  opts.streams_per_brick = 8;
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u16, opts, archive) == Status::ok);

  VolumeInfo info;
  REQUIRE(inspect(archive, info) == Status::ok);
  CHECK(info.dims == v.dims);
  CHECK(info.dtype == DType::u16);
  CHECK(info.profile == Profile::viewing);
  CHECK_EQ(static_cast<int>(info.streams_per_brick), 8);
  CHECK_EQ(info.brick_count, 1u);
}

TEST(encode_rejects_bad_arguments) {
  const Volume v = constant_volume({8, 8, 8}, 1.0f);
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  std::vector<std::uint8_t> archive;
  EncodeOptions opts;
  CHECK(encode(nullptr, v.dims, DType::u8, opts, archive) == Status::invalid_argument);
  CHECK(encode(raw.data(), Dims{0, 8, 8}, DType::u8, opts, archive) == Status::invalid_argument);
  // 0 is not an error: it is the default, and means the encoder picks P itself.
  opts.streams_per_brick = 0;
  CHECK(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::ok);
  opts.streams_per_brick = 65;  // above the format's ceiling
  CHECK(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::invalid_argument);
}

// What the encoder picks for P must be a property of the archive, not of the
// machine that made it -- the probe runs under parallel_for like everything
// else, and a sample whose size depended on the thread count would make the
// bytes depend on it too.
TEST(automatic_stream_count_is_deterministic) {
  const Volume v = scroll_like_volume({256, 192, 160});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  std::vector<std::uint8_t> prev;
  for (int t : {1, 2, 4, 8}) {
    EncodeOptions opts;
    opts.streams_per_brick = 0;
    opts.threads = t;
    std::vector<std::uint8_t> archive;
    REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::ok);
    if (!prev.empty()) CHECK(archive == prev);
    prev = std::move(archive);
  }
  // And it must land inside the format's range whatever it chose.
  VolumeInfo info;
  REQUIRE(inspect(prev, info) == Status::ok);
  CHECK(info.streams_per_brick >= 1);
  CHECK(info.streams_per_brick <= 64);
}

// Pure noise is incompressible. The raw-brick fallback means the codec must
// never make it bigger, and where the fallback triggers on an 8-bit dtype the
// working-domain mapping is an exact integer shift, so it must also be lossless.
TEST(incompressible_data_never_expands) {
  const Volume v = noise_volume({128, 128, 128}, 0.0f, 255.0f);
  for (float q : {0.25f, 1.0f, 4.0f, 16.0f}) {
    EncodeOptions opts;
    opts.quality = q;
    const Stats s = round_trip(v, DType::u8, opts);
    // Worst case is the raw fallback: one mode byte plus the file header and
    // brick index, over a 2 MiB brick.
    CHECK(s.ratio > 0.999);
  }
}

TEST(raw_fallback_is_lossless_for_8_and_16_bit_types) {
  for (DType t : {DType::u8, DType::s8, DType::u16, DType::s16}) {
    Volume v = noise_volume({128, 128, 128}, dtype_min(t), dtype_max(t));
    EncodeOptions opts;
    opts.quality = 64.0f;  // far past the point where coding beats storing
    const Stats s = round_trip(v, t, opts);
    CHECK_EQ(s.max_abs, 0.0);
  }
}

// Edge bricks are mostly padding. The fallback must store only the in-volume
// part, or a volume slightly larger than a brick would store 8x the padding it
// needs and the fallback would never trigger where it should.
TEST(raw_fallback_stores_only_in_volume_voxels) {
  const Volume v = noise_volume({130, 20, 20}, 0.0f, 255.0f);
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions opts;
  opts.quality = 64.0f;
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::ok);
  // Two bricks, one of them 2x20x20 of real data. If padding were stored the
  // archive would be several megabytes.
  CHECK(archive.size() < raw.size() + 4096);

  std::vector<std::uint8_t> out;
  VolumeInfo info;
  REQUIRE(decode(archive, DecodeOptions{}, out, info) == Status::ok);
  CHECK(out == raw);
}

TEST(checkerboard_volume_does_not_break_anything) {
  const Volume v = checkerboard_volume({32, 32, 32}, 0.0f, 255.0f);
  EncodeOptions opts;
  opts.quality = 8.0f;
  const Stats s = round_trip(v, DType::u8, opts);
  CHECK(s.max_abs <= 255.0);
}

TEST(impulse_volume_does_not_break_anything) {
  const Volume v = impulse_volume({32, 32, 32}, 0.0f, 255.0f);
  EncodeOptions opts;
  opts.quality = 16.0f;
  const Stats s = round_trip(v, DType::u8, opts);
  CHECK(s.max_abs <= 255.0);
}

TEST_MAIN()
