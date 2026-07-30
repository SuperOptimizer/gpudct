// Thread safety of the public API.
//
// The library spawns threads internally (parallel_for over bricks) and is also
// expected to survive being called from several of the caller's threads at once
// -- a viewer decoding four regions concurrently is the normal case, not an
// exotic one. Those are different properties and both are asserted here.
//
// This test exists mainly to give the sanitizer something to find. Run it under
// TSan (see .github/workflows/ci.yml); passing without a sanitizer only proves
// that nothing crashed on this particular interleaving, which is weak evidence.
// What makes it worth running unsanitized is the result comparison: a data race
// on shared encoder state usually corrupts output long before it segfaults.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "gpudct/gpudct.hpp"
#include "test.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct_test;

namespace {

constexpr int kThreads = 8;

Volume source() { return scroll_like_volume({144, 144, 144}); }

}  // namespace

// Many threads encoding at once must produce what one thread produces. Shared
// mutable state in the encoder -- a static table, a cached buffer -- shows up
// here as an archive that differs from the serial one.
TEST(concurrent_encodes_agree_with_serial) {
  const Volume v = source();
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);

  EncodeOptions opts;
  opts.quality = 1.0f;
  opts.streams_per_brick = 16;
  opts.threads = 1;  // one thread *inside* each encode, so the concurrency is ours

  std::vector<std::uint8_t> expected;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, expected) == Status::ok);

  std::vector<std::vector<std::uint8_t>> got(kThreads);
  std::atomic<int> failures{0};
  std::vector<std::thread> pool;
  for (int i = 0; i < kThreads; ++i)
    pool.emplace_back([&, i] {
      if (encode(raw.data(), v.dims, DType::u8, opts, got[i]) != Status::ok) ++failures;
    });
  for (auto& t : pool) t.join();

  CHECK(failures.load() == 0);
  for (int i = 0; i < kThreads; ++i) {
    if (got[i] != expected)
      std::printf("       thread %d produced %zu bytes, serial produced %zu\n", i, got[i].size(),
                  expected.size());
    CHECK(got[i] == expected);
  }
}

// Same for decode, which additionally has the deblocking pass writing over the
// output buffer after the parallel section.
TEST(concurrent_decodes_agree_with_serial) {
  const Volume v = source();
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions opts;
  opts.quality = 1.0f;
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::ok);

  DecodeOptions dopts;
  dopts.threads = 1;
  std::vector<std::uint8_t> expected;
  VolumeInfo info;
  REQUIRE(decode(archive, dopts, expected, info) == Status::ok);

  std::vector<std::vector<std::uint8_t>> got(kThreads);
  std::atomic<int> failures{0};
  std::vector<std::thread> pool;
  for (int i = 0; i < kThreads; ++i)
    pool.emplace_back([&, i] {
      VolumeInfo vi;
      if (decode(archive, dopts, got[i], vi) != Status::ok) ++failures;
    });
  for (auto& t : pool) t.join();

  CHECK(failures.load() == 0);
  for (int i = 0; i < kThreads; ++i) CHECK(got[i] == expected);
}

// Random access is the path a viewer actually hammers: many threads pulling
// different bricks out of one archive, with no coordination between them.
TEST(concurrent_brick_reads_agree_with_serial) {
  const Volume v = source();
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions opts;
  opts.quality = 1.0f;
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::ok);

  VolumeInfo info;
  REQUIRE(inspect(archive, info) == Status::ok);

  // 144^3 is two bricks per axis.
  std::vector<std::vector<std::uint8_t>> serial(8);
  int idx = 0;
  for (std::uint32_t z = 0; z < 2; ++z)
    for (std::uint32_t y = 0; y < 2; ++y)
      for (std::uint32_t x = 0; x < 2; ++x, ++idx)
        REQUIRE(decode_brick(archive, x, y, z, serial[idx]) == Status::ok);

  std::atomic<int> mismatches{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < kThreads; ++t)
    pool.emplace_back([&, t] {
      // Each thread walks the bricks from a different offset so they collide.
      for (int k = 0; k < 8; ++k) {
        const int j = (k + t) % 8;
        const std::uint32_t x = static_cast<std::uint32_t>(j % 2);
        const std::uint32_t y = static_cast<std::uint32_t>((j / 2) % 2);
        const std::uint32_t z = static_cast<std::uint32_t>(j / 4);
        std::vector<std::uint8_t> out;
        if (decode_brick(archive, x, y, z, out) != Status::ok || out != serial[j]) ++mismatches;
      }
    });
  for (auto& t : pool) t.join();
  CHECK(mismatches.load() == 0);
}

TEST_MAIN()
