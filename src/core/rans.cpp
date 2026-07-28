#include "rans.hpp"

#include <cmath>
#include <numeric>

namespace gpudct::detail {

// --------------------------------------------------------------------------
// Model
// --------------------------------------------------------------------------

Model Model::from_counts(std::span<const std::uint32_t> counts) {
  Model m;
  const std::size_t n = counts.size();
  assert(n >= 2 && n <= kProbScale / 2);

  m.freq_.assign(n, 0);

  // Every symbol gets at least one slot. An unseen symbol with zero slots would
  // be unencodable, and "the training corpus happened not to contain it" is not
  // a good reason for the encoder to fail on real data later.
  std::uint64_t total = 0;
  for (std::uint32_t c : counts) total += c;

  if (total == 0) {
    const std::uint16_t each = static_cast<std::uint16_t>(kProbScale / n);
    for (std::size_t i = 0; i < n; ++i) m.freq_[i] = each;
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint64_t scaled = (static_cast<std::uint64_t>(counts[i]) * kProbScale) / total;
      m.freq_[i] = static_cast<std::uint16_t>(std::max<std::uint64_t>(1, scaled));
    }
  }

  // Fix up the sum to exactly kProbScale by pushing the residual onto the
  // largest bucket, which is always big enough to absorb it.
  auto sum_of = [&] {
    std::int64_t s = 0;
    for (std::uint16_t f : m.freq_) s += f;
    return s;
  };
  std::int64_t diff = static_cast<std::int64_t>(kProbScale) - sum_of();
  while (diff != 0) {
    const auto it = std::max_element(m.freq_.begin(), m.freq_.end());
    if (diff > 0) {
      const std::int64_t add = std::min<std::int64_t>(diff, kProbScale);
      *it = static_cast<std::uint16_t>(*it + add);
      diff -= add;
    } else {
      const std::int64_t take = std::min<std::int64_t>(-diff, *it - 1);
      if (take <= 0) break;  // cannot happen: sum > kProbScale implies some f > 1
      *it = static_cast<std::uint16_t>(*it - take);
      diff += take;
    }
  }

  m.cum_.assign(n + 1, 0);
  for (std::size_t i = 0; i < n; ++i)
    m.cum_[i + 1] = static_cast<std::uint16_t>(m.cum_[i] + m.freq_[i]);
  assert(m.cum_[n] == kProbScale);

  m.build_derived();
  return m;
}

Model::EncSymbol Model::enc_symbol_of(std::uint16_t freq, std::uint16_t cum) {
  EncSymbol e{};
  e.freq = freq;
  e.cmpl_freq = kProbScale - freq;
  if (freq == 0) return e;
  if (freq < 2) {
    e.rcp_freq = ~0u;
    e.rcp_shift = 0;
    e.bias = cum + kProbScale - 1;
  } else {
    std::uint32_t shift = 0;
    while (freq > (1u << shift)) ++shift;
    e.rcp_freq = static_cast<std::uint32_t>(((1ull << (shift + 31)) + freq - 1) / freq);
    e.rcp_shift = shift - 1;
    e.bias = cum;
  }
  return e;
}

Model Model::from_counts_sparse(std::span<const std::uint32_t> counts) {
  const std::size_t n = counts.size();
  std::uint64_t total = 0;
  for (std::uint32_t c : counts) total += c;
  if (total == 0) return Model::from_counts(counts);

  std::vector<std::uint16_t> freq(n, 0);
  std::int64_t assigned = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (counts[i] == 0) continue;
    const std::uint64_t scaled = (static_cast<std::uint64_t>(counts[i]) * kProbScale) / total;
    freq[i] = static_cast<std::uint16_t>(std::max<std::uint64_t>(1, scaled));
    assigned += freq[i];
  }
  // Push the residual onto the largest bucket, which can always absorb it.
  std::int64_t diff = static_cast<std::int64_t>(kProbScale) - assigned;
  while (diff != 0) {
    const auto it = std::max_element(freq.begin(), freq.end());
    if (diff > 0) {
      *it = static_cast<std::uint16_t>(*it + diff);
      diff = 0;
    } else {
      const std::int64_t take = std::min<std::int64_t>(-diff, *it - 1);
      if (take <= 0) break;
      *it = static_cast<std::uint16_t>(*it - take);
      diff += take;
    }
  }
  return Model::from_freqs(freq);
}

Model Model::from_freqs(std::span<const std::uint16_t> freqs) {
  Model m;
  const std::size_t n = freqs.size();
  assert(n >= 2);
  // A zero frequency is allowed here, unlike in from_counts: a per-brick table
  // omits symbols the brick never uses, and a symbol with no slots simply cannot
  // be decoded. That is the correct outcome -- the encoder never emits one, and
  // a stream that appears to contain one is corrupt.
  m.freq_.assign(freqs.begin(), freqs.end());
  m.cum_.assign(n + 1, 0);
  std::uint32_t sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += m.freq_[i];
    m.cum_[i + 1] = static_cast<std::uint16_t>(sum);
  }
  assert(sum == kProbScale);
  m.build_derived();
  return m;
}

// Slot lookup table plus the division-free encoder parameters. Derived from
// freq_/cum_, so every construction path shares one definition of them.
void Model::build_derived() {
  const std::size_t n = freq_.size();
  assert(n <= 256);  // slot_ is u8; every alphabet here is within that
  slot_.assign(kProbScale, 0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::uint32_t s = cum_[i]; s < cum_[i + 1]; ++s)
      slot_[s] = static_cast<std::uint8_t>(i);

  dec_.assign(n, DecSym{});
  for (std::size_t i = 0; i < n; ++i) dec_[i] = {freq_[i], cum_[i]};

  enc_.assign(n, EncSymbol{});
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t f = freq_[i];
    EncSymbol& e = enc_[i];
    e.freq = f;
    if (f == 0) continue;  // never encoded; leaving the entry zeroed is fine
    e.cmpl_freq = kProbScale - f;
    if (f < 2) {
      // A one-slot symbol cannot use the reciprocal path: the shift would be
      // negative. Multiplying by ~0u and not shifting divides by 1 exactly.
      e.rcp_freq = ~0u;
      e.rcp_shift = 0;
      e.bias = cum_[i] + kProbScale - 1;
    } else {
      std::uint32_t shift = 0;
      while (f > (1u << shift)) ++shift;
      e.rcp_freq = static_cast<std::uint32_t>(
          ((1ull << (shift + 31)) + f - 1) / f);
      e.rcp_shift = shift - 1;
      e.bias = cum_[i];
    }
  }
}

float Model::bit_cost(std::uint32_t s) const noexcept {
  if (s >= freq_.size()) return 1e9f;
  return -std::log2(static_cast<float>(freq_[s]) / static_cast<float>(kProbScale));
}

void Model::serialize(std::vector<std::uint8_t>& out) const {
  const std::uint32_t n = nsym();
  out.push_back(static_cast<std::uint8_t>(n & 0xff));
  out.push_back(static_cast<std::uint8_t>((n >> 8) & 0xff));
  for (std::uint16_t f : freq_) {
    out.push_back(static_cast<std::uint8_t>(f & 0xff));
    out.push_back(static_cast<std::uint8_t>((f >> 8) & 0xff));
  }
}

bool Model::deserialize(std::span<const std::uint8_t> in, std::size_t& pos, Model& out) {
  if (pos + 2 > in.size()) return false;
  const std::uint32_t n =
      static_cast<std::uint32_t>(in[pos]) | (static_cast<std::uint32_t>(in[pos + 1]) << 8);
  pos += 2;
  if (n < 2 || n > kProbScale / 2) return false;
  if (pos + 2 * n > in.size()) return false;

  std::vector<std::uint16_t> freq(n);
  std::uint32_t sum = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    freq[i] = static_cast<std::uint16_t>(static_cast<std::uint32_t>(in[pos]) |
                                         (static_cast<std::uint32_t>(in[pos + 1]) << 8));
    pos += 2;
    if (freq[i] == 0) return false;  // a zero-slot symbol would be undecodable
    sum += freq[i];
  }
  if (sum != kProbScale) return false;

  out.freq_ = std::move(freq);
  out.cum_.assign(n + 1, 0);
  for (std::uint32_t i = 0; i < n; ++i)
    out.cum_[i + 1] = static_cast<std::uint16_t>(out.cum_[i] + out.freq_[i]);
  out.build_derived();
  return true;
}

const Model& bypass_model() {
  static const Model m = [] {
    const std::uint32_t counts[2] = {1, 1};
    return Model::from_counts(counts);
  }();
  return m;
}

// --------------------------------------------------------------------------
// Encoder
// --------------------------------------------------------------------------

std::vector<std::uint8_t> RansEncoder::encode(std::span<const Sym> syms,
                                              std::span<const Model> models) {
  const std::uint32_t n = static_cast<std::uint32_t>(state_.size());
  std::fill(state_.begin(), state_.end(), kRansL);
  bytes_.clear();
  bytes_.reserve(syms.size() + 4 * n);

  // Reverse order: rANS is a stack. Symbol i uses state i % n, which is what the
  // decoder will assume when it walks forward.
  //
  // The state index is walked backwards rather than recomputed as `k % n`: a
  // modulo by a runtime value is an integer division, and there is one per
  // symbol on the hottest loop in the encoder.
  std::size_t si = syms.empty() ? 0 : (syms.size() - 1) % n;
  for (std::size_t k = syms.size(); k-- > 0;) {
    const Sym& s = syms[k];
    const Model::EncSymbol& e = models[s.model].enc(s.value);

    std::uint32_t x = state_[si];
    const std::uint32_t x_max = ((kRansL >> kProbBits) << 8) * e.freq;
    while (x >= x_max) {
      put(static_cast<std::uint8_t>(x & 0xff));
      x >>= 8;
    }
    // Equivalent to ((x / freq) << kProbBits) + (x % freq) + start, without the
    // division: q = x / freq by multiply-high, then x + bias + q * (M - freq).
    const std::uint32_t q =
        static_cast<std::uint32_t>((static_cast<std::uint64_t>(x) * e.rcp_freq) >> 32) >>
        e.rcp_shift;
    state_[si] = x + e.bias + q * e.cmpl_freq;
    si = (si == 0) ? (n - 1) : (si - 1);
  }

  // Flush the states. Bytes are emitted newest-first and the list is reversed at
  // the end, so this order puts state 0 at the lowest address with its bytes
  // little-endian -- exactly what RansDecoder::init reads.
  for (std::uint32_t i = n; i-- > 0;) {
    put(static_cast<std::uint8_t>(state_[i] >> 24));
    put(static_cast<std::uint8_t>(state_[i] >> 16));
    put(static_cast<std::uint8_t>(state_[i] >> 8));
    put(static_cast<std::uint8_t>(state_[i]));
  }

  std::reverse(bytes_.begin(), bytes_.end());
  return std::move(bytes_);
}

// --------------------------------------------------------------------------
// Decoder
// --------------------------------------------------------------------------

bool RansDecoder::init(std::span<const std::uint8_t> bytes, std::uint32_t nstates) {
  failed_ = false;
  pos_ = 0;
  next_state_ = 0;
  if (nstates == 0) return false;
  if (bytes.size() < static_cast<std::size_t>(nstates) * 4) return false;

  bytes_.assign(bytes.begin(), bytes.end());
  state_.resize(nstates);
  for (std::uint32_t i = 0; i < nstates; ++i) {
    state_[i] = static_cast<std::uint32_t>(bytes_[pos_]) |
                (static_cast<std::uint32_t>(bytes_[pos_ + 1]) << 8) |
                (static_cast<std::uint32_t>(bytes_[pos_ + 2]) << 16) |
                (static_cast<std::uint32_t>(bytes_[pos_ + 3]) << 24);
    pos_ += 4;
  }
  return true;
}

bool RansDecoder::decode(const Model& m, std::uint32_t& out) {
  if (failed_) return false;
  // Wrap by compare, not by modulo: a runtime `% n` here is a division on the
  // hot path, once per symbol.
  std::uint32_t& x = state_[next_state_];
  if (++next_state_ == state_.size()) next_state_ = 0;

  const std::uint32_t slot = x & kProbMask;
  const std::uint32_t s = m.slot_table()[slot];
  const Model::DecSym d = m.dec_table()[s];
  x = static_cast<std::uint32_t>(d.freq) * (x >> kProbBits) + slot - d.cum;

  while (x < kRansL) {
    if (pos_ >= bytes_.size()) {
      failed_ = true;
      return false;
    }
    x = (x << 8) | bytes_[pos_++];
  }
  out = s;
  return true;
}

// A 50/50 binary symbol needs no tables at all: the slot's top bit *is* the
// value, and both halves have the same width. Sign bits and mantissa bits are a
// large share of all symbols coded, so skipping two table lookups on each of
// them is worth the special case.
bool RansDecoder::decode_bypass(std::uint32_t& bit) {
  if (failed_) return false;
  std::uint32_t& x = state_[next_state_];
  if (++next_state_ == state_.size()) next_state_ = 0;

  const std::uint32_t slot = x & kProbMask;
  const std::uint32_t b = slot >> (kProbBits - 1);           // 0 or 1
  constexpr std::uint32_t kHalf = kProbScale / 2;
  x = kHalf * (x >> kProbBits) + (slot & (kHalf - 1));

  while (x < kRansL) {
    if (pos_ >= bytes_.size()) {
      failed_ = true;
      return false;
    }
    x = (x << 8) | bytes_[pos_++];
  }
  bit = b;
  return true;
}

}  // namespace gpudct::detail
