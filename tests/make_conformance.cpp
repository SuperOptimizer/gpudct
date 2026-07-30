// Writes the conformance corpus (tests/data/conformance).
//
// Run deliberately and rarely: these archives are inputs to test_conformance,
// and regenerating them turns "we can still read old files" into "we can read
// what we just wrote". The only legitimate reason to run this is a deliberate
// format-version bump, and then the previous archives are kept rather than
// replaced so both versions stay covered.
//
// Built as a normal target but not registered with ctest.

#include <cstdio>
#include <string>
#include <vector>

#include "gpudct/gpudct.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct_test;

namespace {

struct Case {
  const char* file;
  Dims dims;
  DType dtype;
  Profile profile;
  float quality;
  std::uint8_t streams;
};

const Case kCases[] = {
    {"scroll-u8-balanced-q1.gdct", {96, 80, 72}, DType::u8, Profile::balanced, 1.0f, 4},
    {"scroll-u16-archival-q05.gdct", {64, 64, 64}, DType::u16, Profile::archival, 0.5f, 4},
    {"ramp-u8-viewing-q025.gdct", {130, 66, 34}, DType::u8, Profile::viewing, 0.25f, 8},
    {"noise-u8-balanced-q2.gdct", {64, 64, 64}, DType::u8, Profile::balanced, 2.0f, 1},
};

Volume make_volume(const std::string& n, Dims dims, DType dtype) {
  if (n.rfind("ramp", 0) == 0) return ramp_volume(dims, 10.0f, 240.0f);
  if (n.rfind("noise", 0) == 0) return noise_volume(dims, 0.0f, 255.0f, 12345);
  Volume v = deterministic_volume(dims, 4242);
  if (dtype == DType::u16)
    for (float& f : v.data) f *= 257.0f;
  return v;
}

}  // namespace

int main() {
  const std::string dir = std::string(GPUDCT_TEST_DATA_DIR) + "/conformance/";
  for (const Case& c : kCases) {
    const Volume v = make_volume(c.file, c.dims, c.dtype);
    const std::vector<std::uint8_t> raw = to_typed(v, c.dtype);
    EncodeOptions opts;
    opts.profile = c.profile;
    opts.quality = c.quality;
    opts.streams_per_brick = c.streams;
    opts.threads = 1;
    std::vector<std::uint8_t> archive;
    if (encode(raw.data(), c.dims, c.dtype, opts, archive, Backend::cpu_scalar) != Status::ok) {
      std::fprintf(stderr, "encode failed for %s\n", c.file);
      return 1;
    }
    const std::string path = dir + c.file;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
      std::fprintf(stderr, "cannot create %s\n", path.c_str());
      return 1;
    }
    std::fwrite(archive.data(), 1, archive.size(), f);
    std::fclose(f);
    std::printf("wrote %-32s %zu bytes\n", c.file, archive.size());
  }
  return 0;
}
