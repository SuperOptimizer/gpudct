// Conformance: archives written by an older build still decode correctly.
//
// This is the test that outranks every other format check, and it became
// mandatory rather than optional when the build enabled fast floating point
// (docs/QUALITY.md 4.2b). Before that, "the format did not change" could be
// asserted by re-encoding and comparing hashes, because the encoder was
// bit-reproducible. It is not any more -- two toolchains legitimately produce
// different bytes -- so re-encoding can no longer prove anything about
// compatibility. Only decoding an archive that was genuinely produced earlier
// can.
//
// The archives in tests/data/conformance are therefore *inputs*, never
// regenerated as part of a normal change. Regenerating them to make this test
// pass defeats its entire purpose: it converts "we can still read old files"
// into "we can read the files we just wrote", which is not a property anyone
// needs. They are rewritten only alongside a deliberate format-version bump,
// and the old ones are kept so both versions stay covered.
//
// Each archive is paired with the generator that produced its source volume, so
// the decoded result is checked against real ground truth rather than against a
// stored copy of the output.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "gpudct/gpudct.hpp"
#include "test.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct_test;

namespace {

struct ConfCase {
  const char* file;
  Dims dims;
  DType dtype;
  // RMSE of the decode against the source volume, recorded when the archive was
  // created. A drifting decoder shows up here.
  double rmse;
};

const ConfCase kCases[] = {
    {"scroll-u8-balanced-q1.gdct", {96, 80, 72}, DType::u8, 2.6435507980},
    {"scroll-u16-archival-q05.gdct", {64, 64, 64}, DType::u16, 3.3813552361},
    {"ramp-u8-viewing-q025.gdct", {130, 66, 34}, DType::u8, 1.3918776745},
    {"noise-u8-balanced-q2.gdct", {64, 64, 64}, DType::u8, 2.7868883223},
};

// Mirrors the generators in test_golden.cpp. Kept in sync deliberately: if these
// diverge the test fails loudly rather than silently comparing against the wrong
// ground truth.
Volume make_volume(const std::string& n, Dims dims, DType dtype) {
  if (n.rfind("ramp", 0) == 0) return ramp_volume(dims, 10.0f, 240.0f);
  if (n.rfind("noise", 0) == 0) return noise_volume(dims, 0.0f, 255.0f, 12345);
  Volume v = deterministic_volume(dims, 4242);
  if (dtype == DType::u16)
    for (float& f : v.data) f *= 257.0f;
  return v;
}

std::string data_path(const char* file) {
  return std::string(GPUDCT_TEST_DATA_DIR) + "/conformance/" + file;
}

bool read_file(const std::string& path, std::vector<std::uint8_t>& out) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n <= 0) {
    std::fclose(f);
    return false;
  }
  out.resize(static_cast<std::size_t>(n));
  const std::size_t got = std::fread(out.data(), 1, out.size(), f);
  std::fclose(f);
  return got == out.size();
}

double rmse_against(const Volume& v, DType t, const std::vector<std::uint8_t>& decoded) {
  const std::vector<std::uint8_t> raw = to_typed(v, t);
  const std::size_t esz = dtype_size(t);
  const std::size_t n = raw.size() / esz;
  if (decoded.size() != raw.size()) return -1.0;
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

// Wider than the 2% the same-build golden test uses. This one has to survive a
// different compiler, a different backend and a different machine than the one
// that wrote the archive, and the point is to catch a decoder that is wrong, not
// one that rounds differently.
constexpr double kTolerance = 0.05;

}  // namespace

TEST(stored_archives_still_decode) {
  const bool print = std::getenv("GPUDCT_PRINT_CONFORMANCE") != nullptr;
  bool bad = false;
  for (const ConfCase& c : kCases) {
    std::vector<std::uint8_t> archive;
    if (!read_file(data_path(c.file), archive)) {
      std::printf("       missing corpus file: %s\n", data_path(c.file).c_str());
      bad = true;
      continue;
    }

    VolumeInfo info;
    std::vector<std::uint8_t> decoded;
    const Status s = decode(archive, DecodeOptions{}, decoded, info);
    if (s != Status::ok) {
      std::printf("       %-30s decode failed: %s\n", c.file,
                  std::string(status_message(s)).c_str());
      bad = true;
      continue;
    }
    if (info.dims.x != c.dims.x || info.dims.y != c.dims.y || info.dims.z != c.dims.z ||
        info.dtype != c.dtype) {
      std::printf("       %-30s header changed\n", c.file);
      bad = true;
      continue;
    }

    const Volume v = make_volume(c.file, c.dims, c.dtype);
    const double r = rmse_against(v, c.dtype, decoded);
    if (print) {
      std::printf("    {\"%s\", {%u, %u, %u}, DType::%s, %.10f},\n", c.file, c.dims.x, c.dims.y,
                  c.dims.z, std::string(dtype_name(c.dtype)).c_str(), r);
      continue;
    }
    if (r < 0 || std::fabs(r - c.rmse) > kTolerance * c.rmse) {
      std::printf("       %-30s rmse %.6f (want %.6f +/- %.0f%%)\n", c.file, r, c.rmse,
                  100.0 * kTolerance);
      bad = true;
    }
  }
  if (print) return;
  CHECK(!bad);
}

// A decoder that is handed a truncated or corrupted archive must fail, not read
// out of bounds. The conformance archives are convenient real inputs to mangle.
TEST(mangled_archives_are_rejected_not_misread) {
  std::vector<std::uint8_t> archive;
  REQUIRE(read_file(data_path(kCases[0].file), archive));

  // Truncation at every scale, including inside the header.
  for (std::size_t keep : {std::size_t{0}, std::size_t{1}, std::size_t{16}, std::size_t{31},
                           std::size_t{64}, archive.size() / 2, archive.size() - 1}) {
    std::vector<std::uint8_t> t(archive.begin(),
                               archive.begin() + static_cast<std::ptrdiff_t>(keep));
    std::vector<std::uint8_t> out;
    VolumeInfo info;
    const Status s = decode(t, DecodeOptions{}, out, info);
    // Any status is acceptable except a claim of success on a truncated file.
    if (s == Status::ok)
      std::printf("       truncation to %zu bytes reported ok\n", keep);
    CHECK(s != Status::ok);
  }

  // Single-byte corruption across the whole archive. This must not crash or read
  // out of bounds; a wrong-but-in-bounds decode is a legal outcome, since the
  // format carries no per-brick checksum.
  for (std::size_t i = 0; i < archive.size(); i += 997) {
    std::vector<std::uint8_t> m = archive;
    m[i] ^= 0xFF;
    std::vector<std::uint8_t> out;
    VolumeInfo info;
    (void)decode(m, DecodeOptions{}, out, info);
  }
  CHECK(true);
}

TEST_MAIN()
