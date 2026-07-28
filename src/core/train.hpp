// Offline model training (docs/ROADMAP.md M3).
//
// The shipped tables are measured, not guessed. This header exposes the one
// thing the trainer needs: the ability to run the real encoder over a corpus and
// count the symbols it actually produces. Counting through the production coding
// path rather than a reimplementation is the point -- a trainer that models the
// encoder approximately trains tables that are approximately right.
#pragma once

#include "gpudct/types.hpp"
#include "models.hpp"

namespace gpudct::detail {

// Runs the encoder over one volume and accumulates symbol counts into `out`.
// Produces no archive.
void collect_counts(const void* data, Dims dims, DType dtype, const EncodeOptions& opts,
                    CountCollector& out);

// Where the bits go, by coding stage (docs/QUALITY.md section 1.4).
//
// Tuning a codec without this is guesswork: "the masks are too expensive" and
// "the levels are too expensive" call for completely different work, and the
// answer changes with rate. Estimated from the static models' own bit costs,
// so it sums to very close to the real coded size.
struct BitBreakdown {
  double chunk_flags = 0;   // is-this-chunk-empty
  double exponents = 0;
  double l1_masks = 0;      // which sub-blocks are significant
  double l2_masks = 0;      // which coefficients within them
  double levels = 0;        // AC magnitudes
  double dc = 0;            // DC magnitudes
  double escapes = 0;       // bit-length symbols for large magnitudes
  double signs = 0;         // bypass
  double mantissas = 0;     // bypass bits below a leading 1
  std::uint64_t voxels = 0;

  [[nodiscard]] double total() const {
    return chunk_flags + exponents + l1_masks + l2_masks + levels + dc + escapes + signs +
           mantissas;
  }
};

[[nodiscard]] BitBreakdown measure_bits(const void* data, Dims dims, DType dtype,
                                        const EncodeOptions& opts);

}  // namespace gpudct::detail
