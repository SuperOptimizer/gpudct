// Bounded-error correction layer (docs/DESIGN.md section 3.5).
//
// This is what separates a scientific codec from a media codec. The lossy layer
// gives a good average; this layer gives a promise about the tail, which is the
// thing a downstream analysis actually depends on.
//
// Corrections are coded into their own rANS streams so a decoder that only wants
// the fast reconstruction can skip the bytes entirely without parsing them --
// the layer is additive, never load-bearing.
//
// Positions are gap-coded rather than carried in a 4096-bit mask: corrections
// are sparse by construction (they exist to clip a tail), and a mask would cost
// 512 bytes per chunk to say "almost nothing here".
#pragma once

#include <cstdint>
#include <vector>

#include "models.hpp"
#include "rans.hpp"

namespace gpudct::detail {

// `deltas` is a full 4096-entry array in voxel order, in units of the archive's
// correction step; zero means no correction. Appends nothing at all when the
// chunk needs no corrections beyond the one flag symbol.
void encode_correction_symbols(const std::int32_t* deltas, std::vector<Sym>& out);

// Fills `deltas` (zeroed first). `max_delta` bounds what a valid encoder could
// have written, so a corrupt stream is rejected rather than reconstructed from.
[[nodiscard]] bool decode_correction_symbols(RansDecoder& dec, const ModelSet& ms,
                                             std::int32_t* deltas, std::int32_t max_delta);

}  // namespace gpudct::detail
