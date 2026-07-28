#include "chunk_codec.hpp"

#include <bit>
#include <cstdlib>
#include <cstring>

#include "quant.hpp"

namespace gpudct::detail {
namespace {


// Codes a magnitude >= 1 as a modelled bit-length plus bypass mantissa. The
// leading 1 is implicit, so a magnitude of n bits costs H(n) + (n-1) bits
// instead of Exp-Golomb's 2n-1.
inline void emit_len_mag(std::uint32_t mag, std::uint16_t len_model, std::vector<Sym>& out) {
  const int nb = 32 - std::countl_zero(mag);  // mag >= 1
  out.push_back({len_model, static_cast<std::uint16_t>(nb)});
  for (int i = nb - 2; i >= 0; --i)
    out.push_back({kBypassModel, static_cast<std::uint16_t>((mag >> i) & 1)});
}

[[nodiscard]] inline bool read_len_mag(RansDecoder& dec, const Model& len_model,
                                       std::uint32_t& mag) {
  std::uint32_t nb = 0;
  if (!dec.decode(len_model, nb)) return false;
  if (nb == 0 || nb >= kLenSyms) return false;
  std::uint32_t v = 1;
  for (std::uint32_t i = 1; i < nb; ++i) {
    std::uint32_t bit = 0;
    if (!dec.decode_bypass(bit)) return false;
    v = (v << 1) | bit;
  }
  mag = v;
  return true;
}

// Summed magnitude of a coefficient's causal 3D neighbours inside its
// sub-block. `mag` is indexed by bit position; entries above `bit` are zero.
[[nodiscard]] inline std::int32_t neighbour_sum(const std::int32_t* mag, int bit) {
  const int bu = bit & 3, bv = (bit >> 2) & 3, bw = (bit >> 4) & 3;
  std::int32_t s = 0;
  if (bu > 0) s += mag[bit - 1];
  if (bv > 0) s += mag[bit - 4];
  if (bw > 0) s += mag[bit - 16];
  return s;
}

}  // namespace

// --------------------------------------------------------------------------
// Encode
// --------------------------------------------------------------------------

void encode_chunk_symbols(const std::int32_t* levels, std::uint8_t exponent,
                          std::vector<Sym>& out) {
  // Build the two-level significance hierarchy.
  std::uint64_t sub_mask[kSubCount] = {};
  std::uint64_t l1 = 0;
  for (int idx = 0; idx < kChunkVox; ++idx) {
    if (levels[idx] == 0) continue;
    const int sb = subblock_of_coeff(idx);
    sub_mask[sb] |= (std::uint64_t{1} << bit_of_coeff(idx));
    l1 |= (std::uint64_t{1} << sb);
  }

  // L0: an entirely zero chunk stops here. Nothing else about it can matter,
  // including its exponent.
  out.push_back({ModelIndex::chunk_nonzero, static_cast<std::uint16_t>(l1 != 0 ? 1 : 0)});
  if (l1 == 0) return;

  out.push_back({ModelIndex::exponent, exponent});

  // L1: which sub-blocks are significant, as 8 bytes.
  int mctx = 0;
  for (int j = 0; j < 8; ++j) {
    const std::uint32_t byte = static_cast<std::uint32_t>((l1 >> (8 * j)) & 0xff);
    out.push_back({ModelIndex::l1(j, mctx), static_cast<std::uint16_t>(byte)});
    mctx = mask_ctx_of_prev(byte);
  }

  for (int sb = 0; sb < kSubCount; ++sb) {
    if (!((l1 >> sb) & 1)) continue;
    const int band = band_of_subblock_index(sb);

    // L2: which coefficients within this sub-block are nonzero.
    mctx = 0;
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t byte = static_cast<std::uint32_t>((sub_mask[sb] >> (8 * j)) & 0xff);
      out.push_back({ModelIndex::l2(band, mctx), static_cast<std::uint16_t>(byte)});
      mctx = mask_ctx_of_prev(byte);
    }

    // Magnitudes and signs, in increasing bit order.
    std::int32_t nb[kSubVox] = {};
    for (int bit = 0; bit < kSubVox; ++bit) {
      if (!((sub_mask[sb] >> bit) & 1)) continue;
      const int idx = coeff_of_subblock_bit(sb, bit);
      const std::int32_t lv = levels[idx];
      const std::int32_t mag = (lv < 0) ? -lv : lv;

      if (idx == 0) {
        // DC has no useful small-value alphabet: after quantization it is large
        // on essentially every chunk, so it goes straight to bit-length coding.
        emit_len_mag(static_cast<std::uint32_t>(mag), ModelIndex::dc_len, out);
      } else {
        const std::uint16_t model =
            ModelIndex::level(band, level_ctx_of_prev(neighbour_sum(nb, bit)));
        if (mag <= 15) {
          out.push_back({model, static_cast<std::uint16_t>(mag - 1)});
        } else {
          out.push_back({model, static_cast<std::uint16_t>(kLevelEscape)});
          emit_len_mag(static_cast<std::uint32_t>(mag) - 15, ModelIndex::esc_len, out);
        }
      }
      out.push_back({kBypassModel, static_cast<std::uint16_t>(lv < 0 ? 1 : 0)});
      nb[bit] = mag;
    }
  }
}

// --------------------------------------------------------------------------
// Decode -- the exact mirror of the above.
// --------------------------------------------------------------------------

bool decode_chunk_symbols(RansDecoder& dec, const ModelSet& ms, std::int32_t* levels,
                          std::uint8_t& exponent, std::int32_t max_level, int& out_nonzero) {
  std::memset(levels, 0, sizeof(std::int32_t) * kChunkVox);
  out_nonzero = 0;

  exponent = 0;
  std::uint32_t v = 0;
  if (!dec.decode(ms[ModelIndex::chunk_nonzero], v)) return false;
  if (v == 0) return true;  // all-zero chunk; levels are already cleared

  if (!dec.decode(ms[ModelIndex::exponent], v)) return false;
  exponent = static_cast<std::uint8_t>(v);

  std::uint64_t l1 = 0;
  int mctx = 0;
  for (int j = 0; j < 8; ++j) {
    if (!dec.decode(ms[ModelIndex::l1(j, mctx)], v)) return false;
    l1 |= (static_cast<std::uint64_t>(v) << (8 * j));
    mctx = mask_ctx_of_prev(v);
  }

  for (int sb = 0; sb < kSubCount; ++sb) {
    if (!((l1 >> sb) & 1)) continue;
    const int band = band_of_subblock_index(sb);

    std::uint64_t mask = 0;
    mctx = 0;
    for (int j = 0; j < 8; ++j) {
      if (!dec.decode(ms[ModelIndex::l2(band, mctx)], v)) return false;
      mask |= (static_cast<std::uint64_t>(v) << (8 * j));
      mctx = mask_ctx_of_prev(v);
    }

    std::int32_t nb[kSubVox] = {};
    for (int bit = 0; bit < kSubVox; ++bit) {
      if (!((mask >> bit) & 1)) continue;
      const int idx = coeff_of_subblock_bit(sb, bit);

      std::int32_t mag;
      if (idx == 0) {
        std::uint32_t m = 0;
        if (!read_len_mag(dec, ms[ModelIndex::dc_len], m)) return false;
        if (m > static_cast<std::uint32_t>(max_level)) return false;
        mag = static_cast<std::int32_t>(m);
      } else {
        const std::uint16_t model =
            ModelIndex::level(band, level_ctx_of_prev(neighbour_sum(nb, bit)));
        if (!dec.decode(ms[model], v)) return false;
        if (v < kLevelEscape) {
          mag = static_cast<std::int32_t>(v) + 1;
        } else {
          std::uint32_t extra = 0;
          if (!read_len_mag(dec, ms[ModelIndex::esc_len], extra)) return false;
          // A corrupt stream must not be able to name a magnitude the encoder
          // could never have produced.
          if (extra > static_cast<std::uint32_t>(kMaxLevel)) return false;
          mag = static_cast<std::int32_t>(extra) + 15;
          if (mag > max_level) return false;
        }
      }

      std::uint32_t sign = 0;
      if (!dec.decode_bypass(sign)) return false;
      levels[idx] = sign ? -mag : mag;
      ++out_nonzero;
      nb[bit] = mag;
    }
  }
  return true;
}


bool decode_chunk_coeffs(RansDecoder& dec, const ModelSet& ms, float* coeffs,
                         const float* quant, std::uint8_t& exponent,
                         std::int32_t max_level, std::vector<int>& touched) {
  touched.clear();

  exponent = 0;
  std::uint32_t v = 0;
  if (!dec.decode(ms[ModelIndex::chunk_nonzero], v)) return false;
  if (v == 0) return true;  // all-zero chunk; levels are already cleared

  if (!dec.decode(ms[ModelIndex::exponent], v)) return false;
  exponent = static_cast<std::uint8_t>(v);

  std::uint64_t l1 = 0;
  int mctx = 0;
  for (int j = 0; j < 8; ++j) {
    if (!dec.decode(ms[ModelIndex::l1(j, mctx)], v)) return false;
    l1 |= (static_cast<std::uint64_t>(v) << (8 * j));
    mctx = mask_ctx_of_prev(v);
  }

  for (int sb = 0; sb < kSubCount; ++sb) {
    if (!((l1 >> sb) & 1)) continue;
    const int band = band_of_subblock_index(sb);

    std::uint64_t mask = 0;
    mctx = 0;
    for (int j = 0; j < 8; ++j) {
      if (!dec.decode(ms[ModelIndex::l2(band, mctx)], v)) return false;
      mask |= (static_cast<std::uint64_t>(v) << (8 * j));
      mctx = mask_ctx_of_prev(v);
    }

    std::int32_t nb[kSubVox] = {};
    for (int bit = 0; bit < kSubVox; ++bit) {
      if (!((mask >> bit) & 1)) continue;
      const int idx = coeff_of_subblock_bit(sb, bit);

      std::int32_t mag;
      if (idx == 0) {
        std::uint32_t m = 0;
        if (!read_len_mag(dec, ms[ModelIndex::dc_len], m)) return false;
        if (m > static_cast<std::uint32_t>(max_level)) return false;
        mag = static_cast<std::int32_t>(m);
      } else {
        const std::uint16_t model =
            ModelIndex::level(band, level_ctx_of_prev(neighbour_sum(nb, bit)));
        if (!dec.decode(ms[model], v)) return false;
        if (v < kLevelEscape) {
          mag = static_cast<std::int32_t>(v) + 1;
        } else {
          std::uint32_t extra = 0;
          if (!read_len_mag(dec, ms[ModelIndex::esc_len], extra)) return false;
          // A corrupt stream must not be able to name a magnitude the encoder
          // could never have produced.
          if (extra > static_cast<std::uint32_t>(kMaxLevel)) return false;
          mag = static_cast<std::int32_t>(extra) + 15;
          if (mag > max_level) return false;
        }
      }

      std::uint32_t sign = 0;
      if (!dec.decode_bypass(sign)) return false;
      coeffs[idx] = static_cast<float>(sign ? -mag : mag) * quant[idx];
      touched.push_back(idx);
      nb[bit] = mag;
    }
  }
  return true;
}

}  // namespace gpudct::detail
