// Context-map clustering for per-brick entropy tables (fenix ADR-style).
//
// The static global tables cap how many contexts the coder can afford: every
// extra context splits one fixed model set more thinly, so three separate
// attempts to enrich the contexts measured *worse* (docs/QUALITY.md). The fix is
// not fewer contexts, it is tables that belong to the data.
//
// A brick carries its own tables. Raw contexts are greedily merged into a small
// number of clusters, and -- this is the part that makes it safe -- the cost of
// *signalling* the resulting tables is inside the merge objective. A context
// with too little data to pay for its own table therefore collapses into a
// neighbour automatically, so adding contexts can never fragment the model: the
// clusterer simply declines to keep them apart.
//
// The whole thing is also compared against just using the global tables, per
// brick, and the cheaper option wins. That makes this strictly non-regressive:
// the worst case is one flag byte.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "rans.hpp"

namespace gpudct::detail {

// Bytes to serialize a table with `used` distinct symbols: a symbol/frequency
// pair each, plus the count. Must match write_compact_table below.
[[nodiscard]] inline double table_cost_bits(std::size_t used) {
  return 8.0 * (2.0 + 3.0 * static_cast<double>(used));
}

// Shannon cost of coding `counts` under the distribution of `model_counts`.
[[nodiscard]] inline double cross_entropy_bits(const std::vector<std::uint64_t>& counts,
                                               const std::vector<std::uint64_t>& model_counts) {
  std::uint64_t total = 0;
  for (std::uint64_t c : model_counts) total += c;
  if (total == 0) return 0.0;
  double bits = 0.0;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] == 0) continue;
    // A symbol the model never saw still has to be codeable; the normalizer
    // gives it one slot, so charge it accordingly rather than infinity.
    const double p = model_counts[i] > 0
                         ? static_cast<double>(model_counts[i]) / static_cast<double>(total)
                         : 1.0 / static_cast<double>(kProbScale);
    // std::log2, not an approximation. A fast bit-trick log2 was tried here and
    // cost 1.8% BD-rate (-14.14% to -12.34%): the merge decisions are sensitive
    // enough that a ~1e-4 error changes which contexts get clustered together.
    // It saved about 50 ms per gigabyte, which is not worth 1.8% of the ratio.
    bits += static_cast<double>(counts[i]) * -std::log2(p);
  }
  return bits;
}

template <typename T>
[[nodiscard]] inline std::size_t distinct(const std::vector<T>& c) {
  std::size_t n = 0;
  for (T v : c)
    if (v > 0) ++n;
  return n;
}

struct ClusterResult {
  std::vector<std::uint8_t> map;                  // raw context -> cluster index
  std::vector<std::vector<std::uint64_t>> counts; // merged counts per cluster
  double bits = 0.0;                              // coding + table cost
};

// Greedy agglomerative clustering of raw contexts.
//
// Starts with every populated context in its own cluster and repeatedly performs
// the merge with the lowest cost increase, stopping when no merge helps or
// `max_clusters` is reached. Merging two clusters costs the extra coding bits
// from modelling both with one distribution, and saves one table's signalling
// cost -- so a sparse context, whose table costs more than it saves, is merged
// away immediately.
[[nodiscard]] inline ClusterResult cluster_contexts(
    const std::vector<std::vector<std::uint64_t>>& raw, std::size_t max_clusters) {
  const std::size_t n = raw.size();
  ClusterResult r;
  r.map.assign(n, 0);
  if (n == 0) return r;

  std::vector<std::vector<std::uint64_t>> cl;
  std::vector<std::vector<std::size_t>> members;
  std::vector<std::size_t> owner(n, SIZE_MAX);

  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t tot = 0;
    for (std::uint64_t v : raw[i]) tot += v;
    if (tot == 0) continue;  // unused contexts join cluster 0 at the end
    owner[i] = cl.size();
    cl.push_back(raw[i]);
    members.push_back({i});
  }
  if (cl.empty()) {
    r.counts.push_back(std::vector<std::uint64_t>(raw[0].size(), 0));
    return r;
  }

  auto cost_of = [](const std::vector<std::uint64_t>& c) {
    return cross_entropy_bits(c, c) + table_cost_bits(distinct(c));
  };

  // Self-costs and the pairwise merge deltas are cached across iterations.
  //
  // Recomputing every pair after every merge is O(n^3) cross-entropy
  // evaluations, each over a 256-symbol alphabet -- about 100M operations per
  // brick, which made per-brick tables cost more time than the transform and the
  // entropy coding put together. A merge only invalidates the row and column of
  // the cluster it produced, so everything else is reused.
  std::vector<double> self(cl.size());
  for (std::size_t i = 0; i < cl.size(); ++i) self[i] = cost_of(cl[i]);

  auto merge_delta = [&](std::size_t i, std::size_t j) {
    std::vector<std::uint64_t> merged(cl[i].size());
    for (std::size_t k = 0; k < merged.size(); ++k) merged[k] = cl[i][k] + cl[j][k];
    const double after = cross_entropy_bits(cl[i], merged) + cross_entropy_bits(cl[j], merged) +
                         table_cost_bits(distinct(merged));
    return after - (self[i] + self[j]);
  };

  const std::size_t n0 = cl.size();
  std::vector<double> delta(n0 * n0, std::numeric_limits<double>::max());
  for (std::size_t i = 0; i < cl.size(); ++i)
    for (std::size_t j = i + 1; j < cl.size(); ++j) delta[i * n0 + j] = merge_delta(i, j);

  for (;;) {
    if (cl.size() <= 1) break;
    double best = (cl.size() > max_clusters) ? std::numeric_limits<double>::max() : 0.0;
    std::size_t bi = SIZE_MAX, bj = SIZE_MAX;
    for (std::size_t i = 0; i < cl.size(); ++i)
      for (std::size_t j = i + 1; j < cl.size(); ++j)
        if (delta[i * n0 + j] < best) {
          best = delta[i * n0 + j];
          bi = i;
          bj = j;
        }

    if (bi == SIZE_MAX) break;
    if (cl.size() <= max_clusters && best >= 0.0) break;  // no merge pays

    for (std::size_t k = 0; k < cl[bi].size(); ++k) cl[bi][k] += cl[bj][k];
    members[bi].insert(members[bi].end(), members[bj].begin(), members[bj].end());
    cl.erase(cl.begin() + static_cast<std::ptrdiff_t>(bj));
    members.erase(members.begin() + static_cast<std::ptrdiff_t>(bj));
    self.erase(self.begin() + static_cast<std::ptrdiff_t>(bj));
    self[bi > bj ? bi - 1 : bi] = cost_of(cl[bi > bj ? bi - 1 : bi]);

    // Compact the cached deltas around the removed cluster, then refresh only
    // the row and column belonging to the merged one.
    const std::size_t nb = bi > bj ? bi - 1 : bi;
    for (std::size_t i = 0; i < cl.size(); ++i)
      for (std::size_t j = i + 1; j < cl.size(); ++j) {
        const std::size_t oi = i >= bj ? i + 1 : i;
        const std::size_t oj = j >= bj ? j + 1 : j;
        if (i == nb || j == nb) continue;
        delta[i * n0 + j] = delta[oi * n0 + oj];
      }
    for (std::size_t i = 0; i < cl.size(); ++i) {
      if (i == nb) continue;
      const std::size_t a = std::min(i, nb), b = std::max(i, nb);
      delta[a * n0 + b] = merge_delta(a, b);
    }
  }

  for (std::size_t c = 0; c < cl.size(); ++c)
    for (std::size_t m : members[c]) r.map[m] = static_cast<std::uint8_t>(c);
  for (std::size_t i = 0; i < n; ++i)
    if (owner[i] == SIZE_MAX) r.map[i] = 0;  // never used; any cluster will do

  r.counts = std::move(cl);
  r.bits = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    r.bits += cross_entropy_bits(raw[i], r.counts[r.map[i]]);
  for (const auto& c : r.counts) r.bits += table_cost_bits(distinct(c));
  return r;
}

// --------------------------------------------------------------------------
// Compact table serialization: only the symbols that occur.
//
// A symbol that never occurs in this brick gets no slots, so it cannot be
// decoded -- which is correct, because the encoder never emits one. The decoder
// rejects any stream that appears to contain it.
// --------------------------------------------------------------------------
inline void write_compact_table(const Model& m, std::vector<std::uint8_t>& out) {
  std::vector<std::pair<std::uint8_t, std::uint16_t>> used;
  for (std::uint32_t s = 0; s < m.nsym(); ++s)
    if (m.freq(s) > 0) used.push_back({static_cast<std::uint8_t>(s), m.freq(s)});
  out.push_back(static_cast<std::uint8_t>(used.size() & 0xff));
  out.push_back(static_cast<std::uint8_t>(used.size() >> 8));
  for (auto [sym, f] : used) {
    out.push_back(sym);
    out.push_back(static_cast<std::uint8_t>(f & 0xff));
    out.push_back(static_cast<std::uint8_t>(f >> 8));
  }
}

[[nodiscard]] inline bool read_compact_table(std::span<const std::uint8_t> in, std::size_t& pos,
                                             std::uint32_t nsym, Model& out) {
  if (pos + 2 > in.size()) return false;
  const std::size_t used = static_cast<std::size_t>(in[pos]) |
                           (static_cast<std::size_t>(in[pos + 1]) << 8);
  pos += 2;
  if (used == 0 || used > nsym) return false;
  if (pos + 3 * used > in.size()) return false;

  std::vector<std::uint16_t> freq(nsym, 0);
  std::uint32_t sum = 0;
  for (std::size_t i = 0; i < used; ++i) {
    const std::uint32_t sym = in[pos];
    const std::uint32_t f = static_cast<std::uint32_t>(in[pos + 1]) |
                            (static_cast<std::uint32_t>(in[pos + 2]) << 8);
    pos += 3;
    if (sym >= nsym || f == 0 || freq[sym] != 0) return false;
    freq[sym] = static_cast<std::uint16_t>(f);
    sum += f;
  }
  if (sum != kProbScale) return false;
  out = Model::from_freqs(freq);
  return true;
}

}  // namespace gpudct::detail
