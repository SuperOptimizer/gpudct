// The .gdct container (docs/DESIGN.md section 2).
//
// Layout:
//   FileHeader                       fixed size, little-endian
//   [table blob]                     optional, only when flag_custom_tables is set
//   BrickEntry[brick_count]          offset + size per brick
//   brick payloads                   u32 stream_size[P], then P rANS blobs
//
// Everything multi-byte is little-endian and read through explicit byte shuffling
// rather than a struct memcpy, so the format does not depend on host endianness
// or on the compiler's padding decisions.
#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "gpudct/types.hpp"

namespace gpudct::detail {

inline constexpr char kMagic[8] = {'G', 'P', 'U', 'D', 'C', 'T', '\0', '\0'};
inline constexpr std::uint32_t kFormatVersion = 1;

inline constexpr std::uint32_t kFlagCustomTables = 1u << 0;
// Bit 1 was a progressive-band-ordering flag that nothing ever acted on; it is
// left unused rather than reassigned, so no old archive can be misread.
inline constexpr std::uint32_t kFlagRelativeScaling = 1u << 2;
// A bounded-error correction layer is present (docs/DESIGN.md section 3.5).
inline constexpr std::uint32_t kFlagCorrections = 1u << 3;

struct FileHeader {
  std::uint32_t version = kFormatVersion;
  std::uint32_t flags = 0;
  Dims dims{};
  DType dtype = DType::u8;
  Profile profile = Profile::balanced;
  std::uint8_t streams_per_brick = 4;
  std::uint8_t table_version = 2;
  // Quantization, stored explicitly rather than re-derived from the profile so
  // that retuning a profile never changes how an existing archive decodes.
  float q_base = 8.0f, q_a = 3.0f, q_b = 2.0f, deadzone = 0.34f;
  // work = raw * data_scale + data_offset
  float data_scale = 1.0f, data_offset = 0.0f;
  std::uint64_t table_offset = 0, table_size = 0;
  std::uint64_t index_offset = 0;
  std::uint64_t brick_count = 0;
  // Quantization step of the correction layer, in input units. 1.0 for integer
  // dtypes, which makes a correction exact; for f32 it is derived from the
  // requested bound. Appended after the 64-bit fields so that adding it did not
  // move any existing offset.
  float corr_step = 1.0f;

  static constexpr std::size_t kSize = 8 + 4 * 2 + 4 * 3 + 4 + 4 * 6 + 8 * 4 + 4;
};

struct BrickEntry {
  std::uint64_t offset = 0;
  std::uint32_t size = 0;
  std::uint32_t reserved = 0;
  static constexpr std::size_t kSize = 16;
};

// --- little-endian primitives ---------------------------------------------

inline void put_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
  b.push_back(static_cast<std::uint8_t>(v));
  b.push_back(static_cast<std::uint8_t>(v >> 8));
  b.push_back(static_cast<std::uint8_t>(v >> 16));
  b.push_back(static_cast<std::uint8_t>(v >> 24));
}

inline void put_u64(std::vector<std::uint8_t>& b, std::uint64_t v) {
  put_u32(b, static_cast<std::uint32_t>(v));
  put_u32(b, static_cast<std::uint32_t>(v >> 32));
}

inline void put_f32(std::vector<std::uint8_t>& b, float f) {
  std::uint32_t v;
  std::memcpy(&v, &f, 4);
  put_u32(b, v);
}

[[nodiscard]] inline std::uint32_t get_u32(std::span<const std::uint8_t> b, std::size_t p) {
  return static_cast<std::uint32_t>(b[p]) | (static_cast<std::uint32_t>(b[p + 1]) << 8) |
         (static_cast<std::uint32_t>(b[p + 2]) << 16) |
         (static_cast<std::uint32_t>(b[p + 3]) << 24);
}

[[nodiscard]] inline std::uint64_t get_u64(std::span<const std::uint8_t> b, std::size_t p) {
  return static_cast<std::uint64_t>(get_u32(b, p)) |
         (static_cast<std::uint64_t>(get_u32(b, p + 4)) << 32);
}

[[nodiscard]] inline float get_f32(std::span<const std::uint8_t> b, std::size_t p) {
  const std::uint32_t v = get_u32(b, p);
  float f;
  std::memcpy(&f, &v, 4);
  return f;
}

// --- header ---------------------------------------------------------------

void write_header(const FileHeader& h, std::vector<std::uint8_t>& out);
[[nodiscard]] Status read_header(std::span<const std::uint8_t> in, FileHeader& out);

}  // namespace gpudct::detail
