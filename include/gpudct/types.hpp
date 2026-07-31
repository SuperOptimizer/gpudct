// gpudct — public value types.
//
// No 64-bit numeric data types anywhere (see docs/DESIGN.md R7). All codec math
// is f32 or i32; only byte offsets into an archive are 64-bit, because a
// multi-TB volume leaves no choice.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace gpudct {

// --------------------------------------------------------------------------
// Structural constants. These are format-defining; changing one is a format
// version bump. See docs/DESIGN.md section 2 for why the transform unit and the
// entropy unit are different sizes.
// --------------------------------------------------------------------------
inline constexpr int kChunkDim = 16;                            // transform unit edge
inline constexpr int kChunkVox = kChunkDim * kChunkDim * kChunkDim;  // 4096
inline constexpr int kBrickChunks = 8;                          // chunks per brick edge
inline constexpr int kBrickDim = kChunkDim * kBrickChunks;      // 128, entropy/IO unit edge
inline constexpr int kChunksPerBrick = kBrickChunks * kBrickChunks * kBrickChunks;  // 512

// Significance hierarchy: a chunk's 4096 coefficients are 64 sub-blocks of 64.
inline constexpr int kSubDim = 4;
inline constexpr int kSubCount = 64;   // sub-blocks per chunk
inline constexpr int kSubVox = 64;     // coefficients per sub-block

// --------------------------------------------------------------------------

enum class DType : std::uint8_t {
  u8 = 0,
  s8 = 1,
  u16 = 2,
  s16 = 3,
  u32 = 4,
  s32 = 5,
  f32 = 6,
};

[[nodiscard]] constexpr std::size_t dtype_size(DType t) noexcept {
  switch (t) {
    case DType::u8:
    case DType::s8: return 1;
    case DType::u16:
    case DType::s16: return 2;
    case DType::u32:
    case DType::s32:
    case DType::f32: return 4;
  }
  return 0;
}

[[nodiscard]] constexpr bool dtype_is_float(DType t) noexcept { return t == DType::f32; }

[[nodiscard]] constexpr bool dtype_is_signed(DType t) noexcept {
  return t == DType::s8 || t == DType::s16 || t == DType::s32 || t == DType::f32;
}

[[nodiscard]] std::string_view dtype_name(DType t) noexcept;
[[nodiscard]] bool dtype_from_name(std::string_view name, DType& out) noexcept;

// Representable range, as f32. Used for clamping on reconstruction and for the
// PSNR peak. u32/s32 lose precision here by design: f32 has 24 mantissa bits, and
// a codec quantizing to a few hundred levels does not care about bit 25.
[[nodiscard]] constexpr float dtype_min(DType t) noexcept {
  switch (t) {
    case DType::u8: return 0.0f;
    case DType::s8: return -128.0f;
    case DType::u16: return 0.0f;
    case DType::s16: return -32768.0f;
    case DType::u32: return 0.0f;
    case DType::s32: return -2147483648.0f;
    case DType::f32: return -3.402823466e38f;
  }
  return 0.0f;
}

[[nodiscard]] constexpr float dtype_max(DType t) noexcept {
  switch (t) {
    case DType::u8: return 255.0f;
    case DType::s8: return 127.0f;
    case DType::u16: return 65535.0f;
    case DType::s16: return 32767.0f;
    case DType::u32: return 4294967295.0f;
    case DType::s32: return 2147483647.0f;
    case DType::f32: return 3.402823466e38f;
  }
  return 0.0f;
}

// --------------------------------------------------------------------------
// Quantization profiles (docs/DESIGN.md section 3.3). These set the radial
// weighting of the quant matrix: q(u,v,w) = q_base * (1 + a * r^b).
// --------------------------------------------------------------------------
enum class Profile : std::uint8_t {
  // Near-flat weighting. Preserves high-frequency fibre texture, which is what
  // downstream ML (segmentation, ink detection) actually consumes. Lowest ratio.
  archival = 0,
  // Default. Moderate high-frequency rolloff.
  balanced = 1,
  // Aggressive rolloff. Best ratio, intended for human inspection and previews.
  viewing = 2,
  // The quant matrix is stored explicitly in the archive header.
  custom = 3,
};

[[nodiscard]] std::string_view profile_name(Profile p) noexcept;
[[nodiscard]] bool profile_from_name(std::string_view name, Profile& out) noexcept;

struct QuantParams {
  float base = 8.0f;  // q_base; the primary quality knob
  float a = 3.0f;     // radial weighting amplitude
  float b = 2.0f;     // radial weighting exponent
  // Dead-zone offset. level = floor(|c|/q + deadzone), so the zero bin has
  // half-width (1 - deadzone) * q: 0.5 is plain round-to-nearest and *lower*
  // widens the dead zone.
  //
  // The best value depends on whether RDO is running, and strongly:
  //
  //   offset          0.20    0.34    0.45    0.55    0.70
  //   plain, BD-PSNR -3.13%  -7.16%  -6.15%  +2.75% +52.45%
  //   RDO,   BD-PSNR -4.48%  -8.73%  -9.47%  -9.49%  -9.49%
  //
  // Without RDO the dead zone is the only thing discarding noise coefficients,
  // so it has to be wide. With RDO the discarding is already done, optimally,
  // and a wide dead zone merely destroys coefficients RDO would have kept. The
  // encoder therefore picks per effort level; see deadzone_for_effort.
  float deadzone = 0.34f;

  [[nodiscard]] static QuantParams for_profile(Profile p, float quality) noexcept;
};



// --------------------------------------------------------------------------
// Error bounds (docs/DESIGN.md section 3.5). All are opt-in; a bound that is
// left unset is not enforced and costs nothing.
// --------------------------------------------------------------------------
struct ErrorBounds {
  float max_abs = 0.0f;   // hard bound on p100 absolute error, in input units
  float max_rel = 0.0f;   // hard bound relative to each chunk's measured range
  float p99 = 0.0f;       // 99th-percentile absolute error bound
  float p999 = 0.0f;      // 99.9th-percentile absolute error bound

  [[nodiscard]] bool any() const noexcept {
    return max_abs > 0.0f || max_rel > 0.0f || p99 > 0.0f || p999 > 0.0f;
  }
};

// --------------------------------------------------------------------------

struct Dims {
  std::uint32_t x = 0, y = 0, z = 0;

  [[nodiscard]] constexpr std::size_t voxels() const noexcept {
    return static_cast<std::size_t>(x) * y * z;
  }
  [[nodiscard]] constexpr bool empty() const noexcept { return x == 0 || y == 0 || z == 0; }
  [[nodiscard]] friend constexpr bool operator==(Dims, Dims) = default;
};

// Number of bricks along each axis, and in total.
[[nodiscard]] constexpr Dims brick_grid(Dims d) noexcept {
  return {(d.x + kBrickDim - 1) / kBrickDim, (d.y + kBrickDim - 1) / kBrickDim,
          (d.z + kBrickDim - 1) / kBrickDim};
}

// --------------------------------------------------------------------------
// Encoder effort. Affects speed and ratio only, never correctness or format.
// --------------------------------------------------------------------------
enum class Effort : std::uint8_t {
  fast = 0,      // quantize and go
  normal = 1,    // default
  high = 2,      // rate-distortion optimized level decisions
};

// Dead-zone offset appropriate to an effort level; see QuantParams::deadzone.
[[nodiscard]] constexpr float deadzone_for_effort(Effort e) noexcept {
  return (e == Effort::high) ? 0.60f : 0.34f;
}

struct EncodeOptions {
  Profile profile = Profile::balanced;
  float quality = 1.0f;   // scales QuantParams::base; higher = better quality, lower ratio
  Effort effort = Effort::normal;
  ErrorBounds bounds{};
  // Overrides for the profile's quantizer shape, for tuning experiments.
  // Negative means "use the profile's value".
  float deadzone_override = -1.0f;
  float qshape_override = -1.0f;
  float qamp_override = -1.0f;
  // Let each brick carry entropy tables measured on its own contents, clustered
  // to a handful (docs/DESIGN.md section 3.4). Costs a second symbol pass at
  // encode; a brick that cannot pay for its tables keeps the global ones.
  bool per_brick_tables = true;
  // P in docs/DESIGN.md section 2. Sets three things at once, which is why the
  // default is not the smallest value that works:
  //
  //   - GPU decode parallelism, which is bricks x P (+13% from 4 to 16).
  //   - The cost of a single-chunk random read, which walks one stream's prefix,
  //     so it falls roughly as 1/P (1.87 ms -> 1.29 ms per chunk from 4 to 16).
  //   - Ratio, which it costs: one rANS flush per stream per brick.
  //
  // Measured 4 -> 16 on a 512^3 scroll volume: -0.15% ratio and -8% CPU encode,
  // against +13% GPU decode and 1.45x cheaper random chunk access, with CPU
  // decode unchanged. A volume is encoded once and read many times, so 16 is the
  // better default; drop to 4 if archive size is the only thing that matters.
  //
  // 0 means the encoder picks, which is the default. It cannot be picked from
  // the settings alone: the cost of a stream is a fixed 18 bytes per brick, so
  // what P is affordable depends entirely on how large the bricks turn out --
  // the same 16 -> 64 change is +0.21% on a dense archive and +2.84% on a sparse
  // one. The encoder probes a sample of bricks and spends up to 0.35% of the
  // measured payload on extra streams, which reaches P=64 (4.65x faster GPU
  // entropy decode) wherever the rate can pay for it and stays at 16 where it
  // cannot. Pin a value to reproduce an exact archive or to override the budget.
  std::uint8_t streams_per_brick = 0;
  int threads = 0;                     // 0 = hardware concurrency
};

struct DecodeOptions {
  // Chunk-boundary deblocking.
  //
  // On by default. It was off on the argument that it changes the
  // reconstruction and a caller measuring the codec should opt in knowingly --
  // which was the right call while it was believed to be a quality trade. It is
  // not one: measured on real full-resolution scroll it costs zero bits, raises
  // PSNR at every rate, and takes seam excess from +80%/+46%/+25% down to within
  // a few percent of the volume's own interior gradient (docs/QUALITY.md 2.2).
  // A default that leaves a visible 16-voxel lattice in the output, and a false
  // edge in every downstream gradient operator, is the wrong default.
  //
  // Two paths still decline it, both by construction rather than by policy: an
  // archive carrying a correction layer (deblocking would move voxels after the
  // corrections that establish the bound), and single-brick random access
  // (no neighbours to filter against).
  bool deblock = true;
  // Scales the filter thresholds, which are otherwise derived from the
  // archive's own quantizer. 0 disables, 1 is the calibrated default.
  float deblock_strength = 1.0f;
  int threads = 0;
};

// --------------------------------------------------------------------------
// Errors. No exceptions across the API boundary; the C ABI mirrors this enum.
// --------------------------------------------------------------------------
enum class Status : std::int32_t {
  ok = 0,
  invalid_argument = 1,
  corrupt_bitstream = 2,
  unsupported_version = 3,
  unsupported_dtype = 4,
  truncated = 5,
  io_error = 6,
  out_of_memory = 7,
  not_implemented = 8,
  backend_unavailable = 9,
};

[[nodiscard]] std::string_view status_message(Status s) noexcept;

// --------------------------------------------------------------------------
// Backend selection.
// --------------------------------------------------------------------------
enum class Backend : std::uint8_t {
  automatic = 0,
  cpu_scalar = 1,  // the reference implementation; the oracle for every test
  cpu_simd = 2,
  cuda = 3,
};

[[nodiscard]] bool backend_available(Backend b) noexcept;
[[nodiscard]] std::string_view backend_name(Backend b) noexcept;

}  // namespace gpudct
