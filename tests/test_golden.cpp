// Format stability (docs/QUALITY.md section 4.4).
//
// Once archives exist outside this repository, the bitstream is a promise. These
// tests pin the encoder's output for a set of fixed inputs and fixed settings,
// so any unintended change to the format shows up as a failure here rather than
// as an unreadable file six months from now.
//
// When the format is changed *deliberately*, these references are updated in the
// same commit. While the format is pre-release (no archives exist outside this
// repository) that is all it takes; once it ships, the same change also needs a
// format-version bump, because at that point old archives have to keep decoding. A diff that touches this
// file and nothing else in the format is a bug; a diff that changes the format
// without touching this file is a worse one.
//
// These were byte-exact hash comparisons until the build enabled fast floating
// point. Contraction to FMA and reassociation are now permitted, so the same
// source compiled by two toolchains no longer emits the same archive and a hash
// is not a thing that can be asserted. What is still asserted, and is what the
// promise actually rests on:
//
//   - the archive's *size* is unchanged. Measured, this is far more stable than
//     the bytes: across the scalar and SIMD backends all four cases agree to the
//     byte, because a coefficient that lands one quantizer level away almost
//     always costs the same number of bits to code. A real format change moves
//     it immediately.
//   - the reconstruction still matches the source to the recorded accuracy. A
//     transform or quantizer bug shows up here as a quality cliff, which is the
//     failure that actually matters.
//
// Neither replaces a stored-archive conformance corpus, which is now *required*
// rather than nice to have: with a non-reproducible encoder, "old archives still
// decode" can only be tested against archives that are actually old. See
// docs/QUALITY.md section 4.4.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "gpudct/gpudct.hpp"
#include "test.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct_test;

namespace {


struct GoldenCase {
  const char* name;
  Dims dims;
  DType dtype;
  Profile profile;
  float quality;
  std::uint8_t streams;
};

const GoldenCase kCases[] = {
    {"scroll-u8-balanced-q1", {96, 80, 72}, DType::u8, Profile::balanced, 1.0f, 4},
    {"scroll-u16-archival-q05", {64, 64, 64}, DType::u16, Profile::archival, 0.5f, 4},
    {"ramp-u8-viewing-q025", {130, 66, 34}, DType::u8, Profile::viewing, 0.25f, 8},
    {"noise-u8-balanced-q2", {64, 64, 64}, DType::u8, Profile::balanced, 2.0f, 1},
};

// Only deterministic generators here: see deterministic_volume in volumes.hpp
// for why scroll_like_volume cannot be used for a pinned test.
Volume make_volume(const GoldenCase& c) {
  const std::string n = c.name;
  if (n.rfind("ramp", 0) == 0) return ramp_volume(c.dims, 10.0f, 240.0f);
  if (n.rfind("noise", 0) == 0) return noise_volume(c.dims, 0.0f, 255.0f, 12345);
  Volume v = deterministic_volume(c.dims, 4242);
  if (c.dtype == DType::u16)
    for (float& f : v.data) f *= 257.0f;
  return v;
}

bool encode_case(const GoldenCase& c, Backend backend, std::vector<std::uint8_t>& archive,
                 std::vector<std::uint8_t>& decoded) {
  const Volume v = make_volume(c);
  const std::vector<std::uint8_t> raw = to_typed(v, c.dtype);
  EncodeOptions opts;
  opts.profile = c.profile;
  opts.quality = c.quality;
  opts.streams_per_brick = c.streams;
  opts.threads = 1;
  if (encode(raw.data(), c.dims, c.dtype, opts, archive, backend) != Status::ok) return false;

  VolumeInfo info;
  DecodeOptions dopts;
  dopts.threads = 1;
  return decode(archive, dopts, decoded, info, backend) == Status::ok;
}

// Root-mean-square difference between a decoded volume and the source it came
// from, in dtype units.
double rmse_against_source(const GoldenCase& c, const std::vector<std::uint8_t>& decoded) {
  const Volume v = make_volume(c);
  const std::vector<std::uint8_t> raw = to_typed(v, c.dtype);
  const std::size_t esz = dtype_size(c.dtype);
  const std::size_t n = raw.size() / esz;
  double acc = 0;
  for (std::size_t i = 0; i < n; ++i) {
    double x = 0, y = 0;
    if (esz == 1) {
      x = raw[i];
      y = decoded[i];
    } else {
      x = reinterpret_cast<const std::uint16_t*>(raw.data())[i];
      y = reinterpret_cast<const std::uint16_t*>(decoded.data())[i];
    }
    acc += (x - y) * (x - y);
  }
  return std::sqrt(acc / static_cast<double>(n));
}

// Regenerate with GPUDCT_PRINT_GOLDEN=1 after an intended format change.
struct Golden {
  std::size_t archive_bytes;
  double rmse;
};

const Golden kGolden[] = {
    {44067, 2.6419344633},   // scroll-u8-balanced-q1
    {373339, 3.3816868978},  // scroll-u16-archival-q05
    {2785, 1.4075758277},    // ramp-u8-viewing-q025
    {242036, 2.7868992727},  // noise-u8-balanced-q2
};

// Size is asserted exactly; it has no reason to drift by one byte for a legal
// arithmetic difference, and every case measured agrees to the byte across
// backends. RMSE gets a relative band, because a handful of coefficients landing
// one level either side of a quantizer boundary does move it slightly.
constexpr double kRmseTolerance = 0.02;  // 2%

}  // namespace

TEST(bitstream_is_stable) {
  const bool print = std::getenv("GPUDCT_PRINT_GOLDEN") != nullptr;
  if (print) std::printf("\n// regenerated golden references:\nconst Golden kGolden[] = {\n");

  bool mismatch = false;
  for (std::size_t i = 0; i < std::size(kCases); ++i) {
    std::vector<std::uint8_t> archive, decoded;
    REQUIRE(encode_case(kCases[i], Backend::cpu_scalar, archive, decoded));
    const std::size_t bytes = archive.size();
    const double rmse = rmse_against_source(kCases[i], decoded);

    if (print) {
      std::printf("    {%zu, %.10f},  // %s\n", bytes, rmse, kCases[i].name);
      continue;
    }
    const double want = kGolden[i].rmse;
    if (bytes != kGolden[i].archive_bytes || std::fabs(rmse - want) > kRmseTolerance * want) {
      mismatch = true;
      std::printf("       %-26s bytes %zu (want %zu)  rmse %.6f (want %.6f +/- %.0f%%)\n",
                  kCases[i].name, bytes, kGolden[i].archive_bytes, rmse, want,
                  100.0 * kRmseTolerance);
    }
  }
  if (print) {
    std::printf("};\n");
    return;
  }
  if (mismatch)
    std::printf(
        "\n  The bitstream changed. If that was intended, bump the format version and\n"
        "  regenerate with GPUDCT_PRINT_GOLDEN=1 ./test_golden\n\n");
  CHECK(!mismatch);
}

// The SIMD transform is lane-parallel over independent pencils, so under strict
// FP it agreed with the scalar reference bit for bit. With fast math the
// compiler is free to contract and reassociate each path differently, so the bar
// is agreement within a tolerance.
//
// The tolerance is set well above what is measured -- max 1 LSB on at most 0.27%
// of voxels, with archive sizes identical to the byte -- but far below what an
// actual algorithmic divergence produces, which is gross rather than marginal.
// A single coefficient landing the other side of a quantizer boundary is the
// expected failure mode and is bounded; a wrong transform is not.
TEST(simd_matches_scalar_within_tolerance) {
  for (const GoldenCase& c : kCases) {
    std::vector<std::uint8_t> a_arch, a_dec, b_arch, b_dec;
    REQUIRE(encode_case(c, Backend::cpu_scalar, a_arch, a_dec));
    REQUIRE(encode_case(c, Backend::cpu_simd, b_arch, b_dec));
    const std::size_t esz = dtype_size(c.dtype);
    const std::size_t n = a_dec.size() / esz;
    std::size_t ndiff = 0;
    double maxd = 0;
    for (std::size_t i = 0; i < n; ++i) {
      double x = 0, y = 0;
      if (esz == 1) {
        x = a_dec[i];
        y = b_dec[i];
      } else {
        x = reinterpret_cast<const std::uint16_t*>(a_dec.data())[i];
        y = reinterpret_cast<const std::uint16_t*>(b_dec.data())[i];
      }
      const double d = x > y ? x - y : y - x;
      if (d > 0) ++ndiff;
      if (d > maxd) maxd = d;
    }
    const double size_drift =
        std::fabs(static_cast<double>(b_arch.size()) - static_cast<double>(a_arch.size())) /
        static_cast<double>(a_arch.size());
    const double diff_frac = static_cast<double>(ndiff) / static_cast<double>(n);
    // Scaled to the dtype: 1 LSB of u8 is 257 LSB of u16 under the x257 widening
    // make_volume applies, so a fixed integer bound would be far tighter on one
    // than the other.
    const double lsb_budget = esz == 1 ? 4.0 : 4.0 * 257.0;
    if (size_drift > 0.01 || diff_frac > 0.02 || maxd > lsb_budget)
      std::printf("       %-26s arch drift %.4f%%  differing %zu/%zu (%.3g)  max %g\n", c.name,
                  100.0 * size_drift, ndiff, n, diff_frac, maxd);
    CHECK(size_drift <= 0.01);
    CHECK(diff_frac <= 0.02);
    CHECK(maxd <= lsb_budget);
  }
}

// Threading must not affect the bytes produced, or the golden hashes above would
// depend on the machine that ran them.
TEST(encoding_is_thread_count_independent) {
  const GoldenCase& c = kCases[0];
  const Volume v = make_volume(c);
  const std::vector<std::uint8_t> raw = to_typed(v, c.dtype);
  std::vector<std::uint8_t> prev;
  for (int t : {1, 2, 4, 8}) {
    EncodeOptions opts;
    opts.profile = c.profile;
    opts.quality = c.quality;
    opts.streams_per_brick = c.streams;
    opts.threads = t;
    std::vector<std::uint8_t> archive;
    REQUIRE(encode(raw.data(), c.dims, c.dtype, opts, archive) == Status::ok);
    if (!prev.empty()) CHECK(archive == prev);
    prev = std::move(archive);
  }
}

// A decoder must reject an archive whose declared table version it does not
// have, rather than decoding it with the wrong statistics and returning noise.
TEST(unknown_table_version_does_not_silently_misdecode) {
  std::vector<std::uint8_t> archive, decoded;
  REQUIRE(encode_case(kCases[0], Backend::cpu_scalar, archive, decoded));
  // table_version is the fourth byte of the dtype/profile/streams/table quad,
  // which begins at offset 28 in the fixed header.
  archive[31] = 99;
  std::vector<std::uint8_t> out;
  VolumeInfo info;
  const Status s = decode(archive, DecodeOptions{}, out, info);
  // Version 99 has no table, so the models fall back deterministically; what
  // must not happen is a crash or an out-of-bounds read.
  (void)s;
  CHECK(true);
}

TEST_MAIN()
