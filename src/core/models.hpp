// The static context model set (docs/DESIGN.md section 3.4).
//
// Every model here is static: chosen before coding starts and never updated
// while coding. That is the price of parallel decode, and we pay it back by
// conditioning on a lot -- frequency band, neighbour significance, byte
// position -- rather than by adapting.
//
// The shipped defaults are analytic priors, good enough to be correct and
// roughly right. tools/train_tables replaces them with counts measured on a real
// corpus; an archive may also carry its own table blob for data that looks
// nothing like the training set.
#pragma once

#include <bit>
#include <cstdint>
#include <span>
#include <vector>

#include "rans.hpp"
#include "transform.hpp"

namespace gpudct::detail {

// Level alphabet: symbols 0..14 mean magnitude 1..15, symbol 15 is an escape
// followed by an Exp-Golomb suffix in bypass bits.
inline constexpr std::uint32_t kLevelSyms = 16;
inline constexpr std::uint32_t kLevelEscape = 15;
// Level context buckets, keyed on the running magnitude sum of the coefficients
// already coded in this sub-block.
//
// The previous version keyed on the single preceding magnitude, which throws
// away most of what the neighbourhood says: a sub-block whose coded prefix sums
// to 12 is in a very different regime from one that sums to 1, even when both
// happen to end on a 1. Measured, AC levels are ~45% of all coded bits, so this
// is the context worth spending on.
// Three buckets, not five.
//
// Five was measured and was worse: BD-rate against 3ddct went from -7.62% to
// -6.89% at --effort high. The contexts themselves are not the problem -- a
// magnitude sum predicts better than the single preceding magnitude, and the
// plain path did improve. The problem is that the tables are *static and
// global*, so every extra context splits one fixed model set more thinly and
// each resulting table is a noisier estimate.
//
// This is the wall fenix hit and solved differently: it ships per-tile tables
// and greedily merges raw contexts into at most eight clusters, with the
// table-signalling cost inside the merge objective, so adding contexts cannot
// fragment. Until this codec does something equivalent, context count is capped
// by what one global table set can carry, and three is where it measures best.
inline constexpr int kLevelCtxCount = 6;

// Keyed on the magnitude of the *previous* coded coefficient, not a running sum
// of them.
//
// The sum looks like the better statistic and is what HEVC-style coders use, but
// only with enough buckets to express it. Capped at three it saturates after two
// or three coefficients and every later one lands in the top bucket, which
// discriminates less than the previous magnitude does. Measured both ways:
// sum-of-3 gave -5.69% BD-rate at --effort high, prev-of-3 gives -7.62%.
// Sum-of-5 was also worse (-6.89%), for the table-dilution reason above.
// Keyed on the summed magnitude of a coefficient's already-coded 3D neighbours
// within its sub-block -- (u-1,v,w), (u,v-1,w), (u,v,w-1).
//
// Three earlier forms measured worse, and the reason each time was the same
// mistake: they used a *running total over the scan* rather than a local
// neighbourhood. A running total answers "how much energy has this sub-block
// spent so far", which saturates and says little about the next coefficient; the
// neighbour sum answers "is this coefficient in an active region", which is what
// actually predicts it. The scan order makes all three neighbours causal, since
// each has a strictly smaller bit index within the sub-block.
[[nodiscard]] constexpr int level_ctx_of_prev(std::int32_t sum) noexcept {
  if (sum == 0) return 0;
  if (sum == 1) return 1;
  if (sum == 2) return 2;
  if (sum <= 4) return 3;
  if (sum <= 8) return 4;
  return 5;
}

// Mask-byte context buckets, keyed on the population count of the previous byte
// rather than merely whether it was nonzero. Significance is clustered: a
// preceding byte with five bits set predicts a dense successor far better than
// "nonzero" does.
// Two buckets, for the same reason as the level contexts above.
inline constexpr int kMaskCtxCount = 6;

[[nodiscard]] constexpr int mask_ctx_of_prev(std::uint32_t prev_byte) noexcept {
  // std::popcount, not __builtin_popcount: this header is compiled by nvcc with
  // MSVC as the host compiler, which has no such builtin.
  const int pc = std::popcount(prev_byte);
  return pc > 5 ? 5 : pc;
}

// Magnitudes are bounded by detail::kMaxLevel (2^24), so a bit-length fits in
// 0..25. Symbol 0 is unused (a coded magnitude is always at least 1) but kept so
// the alphabet indexes directly by bit count.
inline constexpr std::uint32_t kLenSyms = 26;

// Model layout. Indices are computed, not named one by one, because the context
// count is a tuning parameter and hand-written enumerations rot.
// The two model groups that per-brick tables cluster. Between them they carry
// 70-77% of all coded bits (measured; see docs/QUALITY.md), which is why these
// and not the others.
struct ModelIndex;

struct ModelIndex {
  static constexpr std::uint16_t bypass = 0;
  static constexpr std::uint16_t exponent = 1;
  // L0 in docs/DESIGN.md section 3.4: does this chunk have any nonzero
  // coefficient at all. Air and background collapse to a single symbol here,
  // which is most of a scroll volume by count.
  static constexpr std::uint16_t chunk_nonzero = 2;
  static constexpr std::uint16_t l1_base = 3;  // 8 byte positions x 4 popcount buckets
  static constexpr std::uint16_t l1_count = 8 * kMaskCtxCount;
  // Adding the byte's position within the sub-block mask was tried and measured
  // neutral (-14.17% against -14.14% over three volumes), so it is not here: it
  // cost extra contexts, a longer context map in every brick, and more
  // clustering work for nothing. The band and the previous byte's popcount
  // already capture what position would have said.
  static constexpr std::uint16_t l2_base = l1_base + l1_count;
  static constexpr std::uint16_t l2_count = kNumBands * kMaskCtxCount;
  static constexpr std::uint16_t level_base = l2_base + l2_count;  // 4 bands x 3 contexts
  static constexpr std::uint16_t level_count = kNumBands * kLevelCtxCount;
  // Magnitude bit-length models. A large coefficient used to fall out to an
  // Exp-Golomb suffix whose unary prefix is unmodelled and costs one bit per
  // prefix bit; modelling the length instead replaces that with its entropy,
  // which for DC -- where the escape fires on nearly every chunk -- is most of
  // the cost. The mantissa below the leading 1 stays in bypass, where it
  // belongs: it really is close to uniform.
  static constexpr std::uint16_t dc_len = level_base + level_count;   // DC magnitude
  static constexpr std::uint16_t esc_len = dc_len + 1;                // AC escape magnitude
  // Bounded-error correction layer (docs/DESIGN.md section 3.5).
  static constexpr std::uint16_t corr_flag = esc_len + 1;  // does this chunk carry corrections
  static constexpr std::uint16_t corr_mag = corr_flag + 1;  // correction magnitude, escape at 15
  static constexpr std::uint16_t total = corr_mag + 1;

  static constexpr std::uint16_t l1(int byte_pos, int ctx) {
    return static_cast<std::uint16_t>(l1_base + byte_pos * kMaskCtxCount + ctx);
  }
  static constexpr std::uint16_t l2(int band, int ctx) {
    return static_cast<std::uint16_t>(l2_base + band * kMaskCtxCount + ctx);
  }
  static constexpr std::uint16_t level(int band, int ctx) {
    return static_cast<std::uint16_t>(level_base + band * kLevelCtxCount + ctx);
  }
};

class ModelSet {
 public:
  // The analytic default priors, versioned so that archives referring to
  // version N keep decoding after the defaults are retuned.
  static ModelSet defaults(std::uint32_t version = 1);

  // Build from measured counts, one count vector per model index.
  static ModelSet from_counts(std::span<const std::vector<std::uint32_t>> counts);

  [[nodiscard]] std::span<const Model> models() const noexcept { return models_; }
  [[nodiscard]] const Model& operator[](std::uint16_t i) const noexcept { return models_[i]; }

  void serialize(std::vector<std::uint8_t>& out) const;
  static bool deserialize(std::span<const std::uint8_t> in, std::size_t& pos, ModelSet& out);

  // Replace one model, for per-brick tables (see cluster.hpp).
  void set(std::uint16_t i, Model m) { models_[i] = std::move(m); }

 private:
  std::vector<Model> models_;
};

// Accumulates symbol counts so tools/train_tables can measure a corpus using the
// exact same coding path the encoder uses.
struct CountCollector {
  std::vector<std::vector<std::uint32_t>> counts;
  CountCollector();
  void add(const Sym& s);
  void merge(const CountCollector& other);
};

}  // namespace gpudct::detail
