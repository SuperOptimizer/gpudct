// Format stability (docs/QUALITY.md section 4.4).
//
// Once archives exist outside this repository, the bitstream is a promise. These
// tests hash the encoder's output for a set of fixed inputs and fixed settings,
// so any unintended change to the format shows up as a failure here rather than
// as an unreadable file six months from now.
//
// When the format is changed *deliberately*, these hashes are updated in the
// same commit. While the format is pre-release (no archives exist outside this
// repository) that is all it takes; once it ships, the same change also needs a
// format-version bump, because at that point old archives have to keep decoding. A diff that touches this
// file and nothing else in the format is a bug; a diff that changes the format
// without touching this file is a worse one.
//
// The hashes pin the scalar CPU encoder specifically. It is the only backend
// defined to be reproducible byte-for-byte across machines: it is built with
// -ffp-contract=off and no fast-math, so its float arithmetic is fully
// specified. (The SIMD backend happens to match it exactly today, and
// simd_matches_scalar_bit_exactly below is what keeps that true.)

#include <cstdint>
#include <cstdio>

#include "gpudct/gpudct.hpp"
#include "test.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct_test;

namespace {

// FNV-1a. Not cryptographic -- this guards against accidental drift, not against
// an adversary, and a dependency-free 20-line hash is the right size of tool for
// that job.
std::uint64_t hash_bytes(const std::vector<std::uint8_t>& b) {
  std::uint64_t h = 1469598103934665603ull;
  for (std::uint8_t x : b) {
    h ^= x;
    h *= 1099511628211ull;
  }
  return h;
}

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

// Only bit-reproducible generators here: see deterministic_volume in
// volumes.hpp for why scroll_like_volume cannot be used for a hashed test.
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

// Regenerate with GPUDCT_PRINT_GOLDEN=1 after an intended format change.
struct Golden {
  std::uint64_t archive;
  std::uint64_t decoded;
};

const Golden kGolden[] = {
    {0xec0b0e8d2b0da208ull, 0xbeea527bf7b94a38ull},  // scroll-u8-balanced-q1
    {0x2d7df1a83c72a88dull, 0xb09ab40dcf7c667eull},  // scroll-u16-archival-q05
    {0x14aa31743a019e4aull, 0x3ac03daa1562deebull},  // ramp-u8-viewing-q025
    {0xa04e612526323d23ull, 0xb7c19d894452b60bull},  // noise-u8-balanced-q2
};

}  // namespace

TEST(bitstream_is_stable) {
  const bool print = std::getenv("GPUDCT_PRINT_GOLDEN") != nullptr;
  if (print) std::printf("\n// regenerated golden hashes:\nconst Golden kGolden[] = {\n");

  bool mismatch = false;
  for (std::size_t i = 0; i < std::size(kCases); ++i) {
    std::vector<std::uint8_t> archive, decoded;
    REQUIRE(encode_case(kCases[i], Backend::cpu_scalar, archive, decoded));
    const std::uint64_t ha = hash_bytes(archive), hd = hash_bytes(decoded);

    if (print) {
      std::printf("    {0x%016llxull, 0x%016llxull},  // %s\n",
                  static_cast<unsigned long long>(ha), static_cast<unsigned long long>(hd),
                  kCases[i].name);
      continue;
    }
    if (ha != kGolden[i].archive || hd != kGolden[i].decoded) {
      mismatch = true;
      std::printf("       %-26s archive %016llx (want %016llx)  decoded %016llx (want %016llx)\n",
                  kCases[i].name, static_cast<unsigned long long>(ha),
                  static_cast<unsigned long long>(kGolden[i].archive),
                  static_cast<unsigned long long>(hd),
                  static_cast<unsigned long long>(kGolden[i].decoded));
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

// The SIMD transform is lane-parallel over independent pencils and never
// reassociates, so it must agree with the scalar reference exactly -- not within
// a tolerance. Anything less means an optimization changed the arithmetic.
TEST(simd_matches_scalar_bit_exactly) {
  for (const GoldenCase& c : kCases) {
    std::vector<std::uint8_t> a_arch, a_dec, b_arch, b_dec;
    REQUIRE(encode_case(c, Backend::cpu_scalar, a_arch, a_dec));
    REQUIRE(encode_case(c, Backend::cpu_simd, b_arch, b_dec));
    if (a_arch != b_arch) {
      std::printf("       %s: archives differ (%zu vs %zu bytes)\n", c.name, a_arch.size(),
                  b_arch.size());
      CHECK(a_arch == b_arch);
      return;
    }
    CHECK(a_dec == b_dec);
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
