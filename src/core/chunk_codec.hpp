// Symbol-level codec for one 16x16x16 chunk of quantized levels.
//
// Encoder and decoder live side by side in chunk_codec.cpp on purpose: they must
// walk byte-for-byte identical symbol sequences, and the cheapest way to keep
// two mirrored state machines in sync is to make them impossible to read apart.
#pragma once

#include <cstdint>
#include <vector>

#include "models.hpp"
#include "rans.hpp"

namespace gpudct::detail {

// Appends the symbols for one chunk. `levels` is the full 4096-coefficient array
// in (u,v,w) order; zeros are implicit in the significance masks.
void encode_chunk_symbols(const std::int32_t* levels, std::uint8_t exponent,
                          std::vector<Sym>& out);

// Reads one chunk back. `levels` is zero-filled by this function. Returns false
// on any inconsistency, which is how truncated or corrupt streams are caught
// instead of read past the end.
//
// `max_level` is the largest magnitude the archive's quantizer could have
// produced (detail::max_plausible_level). Anything above it means the stream is
// corrupt, and saying so immediately is both more correct and much cheaper than
// reconstructing from it.
// `out_nonzero` receives the number of nonzero levels, which the parse counts
// anyway. The inverse transform uses it to take a shortcut for empty and
// DC-only chunks without having to rediscover the fact by scanning.
[[nodiscard]] bool decode_chunk_symbols(RansDecoder& dec, const ModelSet& ms,
                                        std::int32_t* levels, std::uint8_t& exponent,
                                        std::int32_t max_level, int& out_nonzero);

// Decode straight into a dequantized coefficient array.
//
// The separate "decode levels, then dequantize all 4096" pair walked the whole
// coefficient array twice per chunk to touch the few dozen entries that are
// actually nonzero. This writes `coeffs[i] = level * quant[i]` as each level is
// parsed and records which entries it touched, so the caller can clear exactly
// those afterwards instead of memsetting 16 KB per chunk.
//
// `coeffs` must be all-zero on entry; on return it is all-zero except at the
// indices listed in `touched`.
[[nodiscard]] bool decode_chunk_coeffs(RansDecoder& dec, const ModelSet& ms, float* coeffs,
                                       const float* quant, std::uint8_t& exponent,
                                       std::int32_t max_level, std::vector<int>& touched);

}  // namespace gpudct::detail
