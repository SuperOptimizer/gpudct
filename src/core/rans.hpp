// Static-model range Asymmetric Numeral Systems (rANS).
//
// 32-bit state, 8-bit renormalization, 12-bit probability precision. N states
// are interleaved over a single byte stream: symbol i is coded by state i % N.
// That is the shape a GPU warp wants (one state per lane, one shared byte
// cursor), and it is why the models must be static -- an adaptive model would
// serialize the lanes. See docs/DESIGN.md section 3.4.
//
// rANS is LIFO: the encoder must process symbols in reverse. Callers therefore
// build a symbol vector first and hand it over whole, which also guarantees the
// encoder and decoder walk the identical symbol sequence.
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace gpudct::detail {

inline constexpr std::uint32_t kProbBits = 12;
inline constexpr std::uint32_t kProbScale = 1u << kProbBits;  // 4096
// Lower bound of the state. With 8-bit renormalization this keeps the state in
// [2^23, 2^31), which is what makes the division-free encoder exact: the 31-bit
// reciprocal in Model::EncSymbol only reproduces x/freq for x below 2^31. A
// 16-bit renormalization would let the state reach 2^32 and silently corrupt
// roughly one symbol in a million -- frequent enough to break a real archive,
// rare enough to pass a small test.
inline constexpr std::uint32_t kRansL = 1u << 23;
inline constexpr std::uint32_t kProbMask = kProbScale - 1;

// Interleaved states per stream.
//
// Originally 32, chosen as the CUDA warp width so a warp could decode one state
// per lane. That turned out to be impossible for this format -- the parse is
// data-dependent, see the correction in docs/DESIGN.md section 3.4 -- which left
// 32 states with no benefit and two real costs: 128 bytes of flush per stream,
// and enough live state to spill out of registers on both CPU and GPU.
//
// Interleaving exists to break the serial state->state dependency so the CPU can
// pipeline the slot lookups. Four lanes is enough to cover that latency; more
// buys nothing. (fenix reaches the same conclusion independently and ships K=4.)
inline constexpr std::uint32_t kRansStates = 4;

// --------------------------------------------------------------------------
// A single static model: normalized frequencies plus the slot -> symbol table
// that makes decoding a lookup rather than a search.
// --------------------------------------------------------------------------
class Model {
 public:
  Model() = default;

  // Build from raw counts. Every symbol is given at least one slot so that any
  // symbol in the alphabet remains encodable, whatever the training corpus did
  // or did not contain.
  static Model from_counts(std::span<const std::uint32_t> counts);

  // Build from frequencies already normalized to kProbScale. Trained tables
  // ship in this form so that loading them cannot re-derive anything and drift
  // from what the trainer actually measured.
  static Model from_freqs(std::span<const std::uint16_t> freqs);

  // Like from_counts, but a symbol with zero count gets zero frequency rather
  // than a floor of one. That makes the table sparse -- and therefore small to
  // serialize -- which is what per-brick tables need. The symbol becomes
  // unencodable, which is correct: it does not occur in this brick.
  static Model from_counts_sparse(std::span<const std::uint32_t> counts);


  [[nodiscard]] std::uint32_t nsym() const noexcept {
    return static_cast<std::uint32_t>(freq_.size());
  }
  [[nodiscard]] std::uint16_t freq(std::uint32_t s) const noexcept { return freq_[s]; }
  [[nodiscard]] std::uint16_t cum(std::uint32_t s) const noexcept { return cum_[s]; }
  [[nodiscard]] std::uint16_t sym_of_slot(std::uint32_t slot) const noexcept {
    return slot_[slot];
  }

  // freq and cum for one symbol, adjacent in memory. The decoder needs both on
  // every symbol, and splitting them across two arrays cost a second cache miss
  // per symbol for no reason.
  struct DecSym {
    std::uint16_t freq;
    std::uint16_t cum;
  };
  [[nodiscard]] const DecSym* dec_table() const noexcept { return dec_.data(); }
  [[nodiscard]] const std::uint8_t* slot_table() const noexcept { return slot_.data(); }

  // Division-free encoding parameters.
  //
  // The textbook encoder step needs x/freq and x%freq. An integer division is
  // tens of cycles and there is one per symbol, which made it the single
  // largest cost in the whole codec -- larger than the transform by an order of
  // magnitude. These precomputed reciprocals turn it into a multiply-high and a
  // shift.
  struct EncSymbol {
    std::uint32_t rcp_freq;
    std::uint32_t freq;
    std::uint32_t bias;
    std::uint32_t cmpl_freq;
    std::uint32_t rcp_shift;
  };
  [[nodiscard]] const EncSymbol& enc(std::uint32_t s) const noexcept { return enc_[s]; }

  // Encoder parameters for one symbol, without building a Model.
  //
  // An encoder needs only the reciprocal-multiply constants; constructing a
  // Model to get them also fills a 4096-entry slot table that encoding never
  // reads. At a dozen tables per brick across thousands of bricks that was most
  // of GPU encode's table stage.
  static EncSymbol enc_symbol_of(std::uint16_t freq, std::uint16_t cum);

  // Cost of coding `s` with this model, in bits. Used by the RDO level decision
  // and by the bit-accounting in the metrics tool.
  [[nodiscard]] float bit_cost(std::uint32_t s) const noexcept;

  void serialize(std::vector<std::uint8_t>& out) const;
  static bool deserialize(std::span<const std::uint8_t> in, std::size_t& pos, Model& out);

 private:
  std::vector<std::uint16_t> freq_;
  std::vector<std::uint16_t> cum_;   // size nsym + 1, cum_[nsym] == kProbScale
  // Symbol indices fit in a byte for every alphabet this codec uses (256 max),
  // so the slot table is u8: half the cache footprint of a u16 one, and it is
  // touched once per symbol.
  std::vector<std::uint8_t> slot_;  // size kProbScale
  std::vector<DecSym> dec_;
  std::vector<EncSymbol> enc_;

  void build_derived();
};

// --------------------------------------------------------------------------
// Symbol record. Models are indexed into a ModelSet; kBypassModel is a fixed
// 50/50 binary model used for sign bits and Exp-Golomb suffixes, which costs
// exactly one bit and needs no table.
// --------------------------------------------------------------------------
struct Sym {
  std::uint16_t model;
  std::uint16_t value;
};

inline constexpr std::uint16_t kBypassModel = 0;  // by convention, model 0 in every ModelSet

// --------------------------------------------------------------------------
// Encoder. Emits u16 words in reverse; `finish` reverses them into stream order.
// --------------------------------------------------------------------------
class RansEncoder {
 public:
  explicit RansEncoder(std::uint32_t nstates) : state_(nstates, kRansL) {}

  // Encodes the whole symbol sequence and returns the byte stream. Symbols are
  // consumed back to front, which is what rANS requires; symbol i uses state
  // i % nstates, matching the decoder exactly.
  std::vector<std::uint8_t> encode(std::span<const Sym> syms,
                                   std::span<const Model> models);

 private:
  void put(std::uint8_t b) { bytes_.push_back(b); }

  std::vector<std::uint32_t> state_;
  std::vector<std::uint8_t> bytes_;
};

// --------------------------------------------------------------------------
// Decoder. Reads forward from the stream produced above.
// --------------------------------------------------------------------------
class RansDecoder {
 public:
  // Returns false if the stream is too short to hold the interleaved states.
  bool init(std::span<const std::uint8_t> bytes, std::uint32_t nstates);

  // Decodes the next symbol from the round-robin state. Returns false on
  // underrun, which is how a truncated or corrupt stream is caught rather than
  // read past the end of the buffer.
  [[nodiscard]] bool decode(const Model& m, std::uint32_t& out);

  // Convenience for the fixed 50/50 binary model.
  [[nodiscard]] bool decode_bypass(std::uint32_t& bit);

  [[nodiscard]] bool exhausted() const noexcept { return pos_ >= bytes_.size(); }

 private:
  std::vector<std::uint8_t> bytes_;
  std::vector<std::uint32_t> state_;
  std::size_t pos_ = 0;
  std::size_t next_state_ = 0;
  bool failed_ = false;
};

// The bypass model, materialized so that Model-taking code paths can use it
// uniformly. Two symbols, 2048 slots each: exactly one bit, no rounding loss.
[[nodiscard]] const Model& bypass_model();

}  // namespace gpudct::detail
