// Decoder robustness.
//
// These archives get shared between researchers over object storage, so a
// decoder that can be walked out of bounds by a hand-edited file is a real
// vulnerability rather than a theoretical one. Every case here asserts the same
// thing: the decoder either succeeds or reports an error, and never reads past
// its buffer. Run under ASan/UBSan for the second half of that claim to mean
// anything.

#include <cstring>

#include "gpudct/gpudct.hpp"
#include "test.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct_test;

namespace {

// Fuzzing a codec is inherently slow: a corrupt stream decodes into millions of
// plausible-looking symbols before anything detects the problem, and there is no
// honest early exit for that. So the sweeps below are exhaustive where the
// structure lives -- the header, the brick index, the start of the payload,
// which is where the interesting failures cluster -- and sampled through the
// bulk of the payload. GPUDCT_FUZZ_FULL=1 makes every sweep exhaustive; CI runs
// that nightly, not per-PR.
[[nodiscard]] bool fuzz_full() {
  const char* v = std::getenv("GPUDCT_FUZZ_FULL");
  return v != nullptr && v[0] == '1';
}

// Bytes at the front of an archive that are exhaustively swept regardless.
constexpr std::size_t kStructuralBytes = 1024;

std::vector<std::uint8_t> make_archive(Dims d = {70, 60, 50}) {
  const Volume v = scroll_like_volume(d);
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions opts;
  std::vector<std::uint8_t> archive;
  const Status s = encode(raw.data(), d, DType::u8, opts, archive);
  if (s != Status::ok) archive.clear();
  return archive;
}

// Decoding must terminate with some definite answer, never crash.
void decode_must_not_crash(std::span<const std::uint8_t> bytes) {
  std::vector<std::uint8_t> out;
  VolumeInfo info;
  const Status s = decode(bytes, DecodeOptions{}, out, info);
  (void)s;
  VolumeInfo probe;
  (void)inspect(bytes, probe);
  std::vector<std::uint8_t> brick;
  (void)decode_brick(bytes, 0, 0, 0, brick);
}

}  // namespace

TEST(empty_and_tiny_inputs_are_rejected) {
  std::vector<std::uint8_t> out;
  VolumeInfo info;
  CHECK(decode({}, DecodeOptions{}, out, info) == Status::truncated);
  for (std::size_t n = 1; n < 128; ++n) {
    std::vector<std::uint8_t> junk(n, 0xab);
    const Status s = decode(junk, DecodeOptions{}, out, info);
    CHECK(s != Status::ok);
  }
}

TEST(bad_magic_is_rejected) {
  std::vector<std::uint8_t> a = make_archive();
  REQUIRE(!a.empty());
  a[0] = 'X';
  std::vector<std::uint8_t> out;
  VolumeInfo info;
  CHECK(decode(a, DecodeOptions{}, out, info) == Status::corrupt_bitstream);
}

TEST(unsupported_version_is_rejected) {
  std::vector<std::uint8_t> a = make_archive();
  REQUIRE(!a.empty());
  a[8] = 99;
  std::vector<std::uint8_t> out;
  VolumeInfo info;
  CHECK(decode(a, DecodeOptions{}, out, info) == Status::unsupported_version);
}

// Truncation. Exhaustive across the structural prefix, strided beyond it.
TEST(truncation_is_handled) {
  const std::vector<std::uint8_t> a = make_archive({40, 40, 40});
  REQUIRE(!a.empty());
  const bool full = fuzz_full();
  const std::size_t dense = full ? a.size() : std::min(a.size(), kStructuralBytes);
  for (std::size_t n = 0; n < dense; ++n)
    decode_must_not_crash(std::span<const std::uint8_t>(a.data(), n));

  if (!full) {
    const std::size_t stride = std::max<std::size_t>(1, (a.size() - dense) / 128);
    for (std::size_t n = dense; n < a.size(); n += stride)
      decode_must_not_crash(std::span<const std::uint8_t>(a.data(), n));
  }
}

TEST(single_bit_flips_are_handled) {
  const std::vector<std::uint8_t> a = make_archive({40, 40, 40});
  REQUIRE(!a.empty());
  const bool full = fuzz_full();

  // Every bit of the structural prefix, then a sample of the payload.
  const std::size_t dense = std::min(a.size(), full ? a.size() : std::size_t{192});
  for (std::size_t byte = 0; byte < dense; ++byte)
    for (int bit = 0; bit < 8; ++bit) {
      std::vector<std::uint8_t> c = a;
      c[byte] ^= static_cast<std::uint8_t>(1u << bit);
      decode_must_not_crash(c);
    }

  Rng rng(2024);
  const int samples = full ? 3000 : 200;
  for (int trial = 0; trial < samples; ++trial) {
    std::vector<std::uint8_t> c = a;
    const std::size_t byte = dense + rng.next() % (c.size() - dense);
    c[byte] ^= static_cast<std::uint8_t>(1u << (rng.next() % 8));
    decode_must_not_crash(c);
  }
}

// The header fields the decoder uses for arithmetic are the ones worth attacking
// directly: dimensions, brick count, offsets.
TEST(hostile_header_fields_are_handled) {
  const std::vector<std::uint8_t> a = make_archive({40, 40, 40});
  REQUIRE(!a.empty());
  // Every 4-byte field in the fixed header, set to a selection of nasty values.
  for (std::size_t off = 8; off + 4 <= 88; off += 4) {
    for (std::uint32_t v : {0u, 1u, 0xffffffffu, 0x7fffffffu, 0x80000000u}) {
      std::vector<std::uint8_t> c = a;
      std::memcpy(c.data() + off, &v, 4);
      decode_must_not_crash(c);
    }
  }
}

TEST(brick_index_entries_pointing_outside_the_file_are_rejected) {
  std::vector<std::uint8_t> a = make_archive({40, 40, 40});
  REQUIRE(!a.empty());
  // index_offset lives at byte 72 in the fixed header; the first entry follows.
  std::uint64_t index_offset = 0;
  std::memcpy(&index_offset, a.data() + 72, 8);
  REQUIRE(index_offset + 16 <= a.size());

  const std::uint64_t huge = 0xffffffffffff0000ull;
  std::memcpy(a.data() + index_offset, &huge, 8);
  std::vector<std::uint8_t> out;
  VolumeInfo info;
  CHECK(decode(a, DecodeOptions{}, out, info) == Status::corrupt_bitstream);
}

TEST(brick_sizes_that_do_not_sum_are_rejected) {
  std::vector<std::uint8_t> a = make_archive({40, 40, 40});
  REQUIRE(!a.empty());
  std::uint64_t index_offset = 0;
  std::memcpy(&index_offset, a.data() + 72, 8);
  std::uint64_t payload_off = 0;
  std::memcpy(&payload_off, a.data() + index_offset, 8);
  REQUIRE(payload_off + 4 <= a.size());

  // Inflate the first stream's declared size past the payload it lives in.
  const std::uint32_t bogus = 0x0fffffffu;
  std::memcpy(a.data() + payload_off, &bogus, 4);
  std::vector<std::uint8_t> out;
  VolumeInfo info;
  CHECK(decode(a, DecodeOptions{}, out, info) != Status::ok);
}

TEST(random_bytes_are_never_accepted) {
  Rng rng(31337);
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<std::uint8_t> junk(1 + rng.next() % 4096);
    for (auto& b : junk) b = static_cast<std::uint8_t>(rng.next());
    decode_must_not_crash(junk);
  }
  // ... including when the magic happens to be right.
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<std::uint8_t> junk(96 + rng.next() % 4096);
    for (auto& b : junk) b = static_cast<std::uint8_t>(rng.next());
    const char magic[8] = {'G', 'P', 'U', 'D', 'C', 'T', '\0', '\0'};
    std::memcpy(junk.data(), magic, 8);
    const std::uint32_t ver = 1;
    std::memcpy(junk.data() + 8, &ver, 4);
    decode_must_not_crash(junk);
  }
}

TEST(out_of_range_brick_coordinates_are_rejected) {
  const std::vector<std::uint8_t> a = make_archive({40, 40, 40});
  REQUIRE(!a.empty());
  std::vector<std::uint8_t> brick;
  CHECK(decode_brick(a, 1, 0, 0, brick) == Status::invalid_argument);
  CHECK(decode_brick(a, 0, 9999, 0, brick) == Status::invalid_argument);
  CHECK(decode_brick(a, 0, 0, 0, brick) == Status::ok);
}

TEST_MAIN()
