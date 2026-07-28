#include "correction.hpp"

#include <bit>
#include <cstring>

namespace gpudct::detail {
namespace {

// Exp-Golomb order 0 over bypass bits. Same construction as chunk_codec.cpp;
// kept local rather than shared because the two layers are free to diverge and
// a shared helper would quietly couple them.
inline void emit_eg0(std::uint32_t v, std::vector<Sym>& out) {
  const std::uint32_t n = v + 1;
  const int nb = 31 - std::countl_zero(n);
  for (int i = 0; i < nb; ++i) out.push_back({kBypassModel, 1});
  out.push_back({kBypassModel, 0});
  for (int i = nb - 1; i >= 0; --i)
    out.push_back({kBypassModel, static_cast<std::uint16_t>((n >> i) & 1)});
}

inline constexpr int kMaxEgPrefix = 28;

[[nodiscard]] inline bool read_eg0(RansDecoder& dec, std::uint32_t& v) {
  int nb = 0;
  for (;;) {
    std::uint32_t bit = 0;
    if (!dec.decode_bypass(bit)) return false;
    if (bit == 0) break;
    if (++nb > kMaxEgPrefix) return false;
  }
  std::uint32_t n = 1;
  for (int i = 0; i < nb; ++i) {
    std::uint32_t bit = 0;
    if (!dec.decode_bypass(bit)) return false;
    n = (n << 1) | bit;
  }
  v = n - 1;
  return true;
}

}  // namespace

void encode_correction_symbols(const std::int32_t* deltas, std::vector<Sym>& out) {
  int count = 0;
  for (int i = 0; i < kChunkVox; ++i)
    if (deltas[i] != 0) ++count;

  out.push_back({ModelIndex::corr_flag, static_cast<std::uint16_t>(count > 0 ? 1 : 0)});
  if (count == 0) return;

  emit_eg0(static_cast<std::uint32_t>(count - 1), out);

  int prev = -1;
  for (int i = 0; i < kChunkVox; ++i) {
    if (deltas[i] == 0) continue;
    emit_eg0(static_cast<std::uint32_t>(i - prev - 1), out);
    prev = i;

    const std::int32_t d = deltas[i];
    const std::int32_t mag = (d < 0) ? -d : d;
    if (mag <= 15) {
      out.push_back({ModelIndex::corr_mag, static_cast<std::uint16_t>(mag - 1)});
    } else {
      out.push_back({ModelIndex::corr_mag, static_cast<std::uint16_t>(kLevelEscape)});
      emit_eg0(static_cast<std::uint32_t>(mag) - 16, out);
    }
    out.push_back({kBypassModel, static_cast<std::uint16_t>(d < 0 ? 1 : 0)});
  }
}

bool decode_correction_symbols(RansDecoder& dec, const ModelSet& ms, std::int32_t* deltas,
                               std::int32_t max_delta) {
  std::memset(deltas, 0, sizeof(std::int32_t) * kChunkVox);

  std::uint32_t v = 0;
  if (!dec.decode(ms[ModelIndex::corr_flag], v)) return false;
  if (v == 0) return true;

  std::uint32_t count_m1 = 0;
  if (!read_eg0(dec, count_m1)) return false;
  if (count_m1 >= static_cast<std::uint32_t>(kChunkVox)) return false;
  const std::uint32_t count = count_m1 + 1;

  std::int64_t prev = -1;
  for (std::uint32_t k = 0; k < count; ++k) {
    std::uint32_t gap = 0;
    if (!read_eg0(dec, gap)) return false;
    const std::int64_t pos = prev + 1 + static_cast<std::int64_t>(gap);
    // A corrupt gap must not walk off the end of the chunk.
    if (pos >= kChunkVox) return false;
    prev = pos;

    if (!dec.decode(ms[ModelIndex::corr_mag], v)) return false;
    std::int32_t mag;
    if (v < kLevelEscape) {
      mag = static_cast<std::int32_t>(v) + 1;
    } else {
      std::uint32_t extra = 0;
      if (!read_eg0(dec, extra)) return false;
      if (extra > (1u << 24)) return false;
      mag = static_cast<std::int32_t>(extra) + 16;
    }

    // A valid encoder never emits a correction larger than the residual the
    // lossy layer can leave behind; anything bigger means a corrupt stream.
    if (mag > max_delta) return false;

    std::uint32_t sign = 0;
    if (!dec.decode_bypass(sign)) return false;
    deltas[pos] = sign ? -mag : mag;
  }
  return true;
}

}  // namespace gpudct::detail
