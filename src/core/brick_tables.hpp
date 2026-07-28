// Per-brick entropy tables (docs/DESIGN.md section 3.4).
//
// A brick measures its own symbol statistics, clusters its raw contexts into a
// few tables (cluster.hpp), and ships them in its payload. The alternative --
// one static table set for every archive -- is what capped context richness:
// three attempts to enrich the contexts all measured worse because each extra
// context split one fixed model set more thinly.
//
// Only two model groups are clustered, because between them they carry 70-77% of
// all coded bits: the mask bytes (which sub-blocks and which coefficients are
// significant) and the AC level magnitudes. DC and escape lengths get one
// per-brick table each. Everything else -- the bypass model, the chunk flag, the
// exponent -- is tiny and stays global.
//
// Every brick independently chooses between its own tables and the global ones,
// counting the table bytes on its own side of the comparison. A brick too small
// or too uniform to pay for its tables simply does not use them, so this cannot
// regress: the worst case is one flag byte.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "cluster.hpp"
#include "models.hpp"

namespace gpudct::detail {

// Raw contexts that share the 256-symbol mask alphabet: the 8x2 L1 models
// followed by the 4x2 L2 models.
inline constexpr int kMaskCtxTotal = 8 * kMaskCtxCount + kNumBands * kMaskCtxCount;
inline constexpr int kLevelCtxTotal = kNumBands * kLevelCtxCount;

// Cap on tables per group. Beyond this the signalling cost outweighs the
// modelling gain at any brick size we use.
inline constexpr std::size_t kMaxMaskClusters = 8;
inline constexpr std::size_t kMaxLevelClusters = 6;

[[nodiscard]] inline std::uint16_t mask_model_index(int raw) {
  const int nl1 = 8 * kMaskCtxCount;
  return raw < nl1 ? static_cast<std::uint16_t>(ModelIndex::l1_base + raw)
                   : static_cast<std::uint16_t>(ModelIndex::l2_base + (raw - nl1));
}

struct BrickTables {
  ModelSet models;                 // global set with the clustered groups replaced
  std::vector<std::uint8_t> blob;  // serialized tables + context maps
  bool used = false;               // false => the brick uses the global tables
};

// Builds per-brick tables from measured symbol counts, and decides whether they
// are worth their own size.
//
// `counts[model][symbol]` must be the symbol histogram for this brick.
// `want_models` builds the ready-to-use ModelSet as well as the blob. Decoders
// and the CPU encoder need it; the GPU encoder does not -- it re-parses the blob
// into upload form -- and materializing it costs a copy of all 75 global models,
// each carrying a 4096-entry slot table, for every brick.
[[nodiscard]] inline BrickTables build_brick_tables(
    const ModelSet& global, const std::vector<std::vector<std::uint64_t>>& counts,
    bool want_models = true) {
  BrickTables out;
  if (want_models) out.models = global;

  auto gather = [&](auto index_of, int n) {
    std::vector<std::vector<std::uint64_t>> raw(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) raw[static_cast<std::size_t>(i)] = counts[index_of(i)];
    return raw;
  };

  const auto mask_raw = gather([](int i) { return mask_model_index(i); }, kMaskCtxTotal);
  const auto level_raw = gather(
      [](int i) { return static_cast<std::uint16_t>(ModelIndex::level_base + i); },
      kLevelCtxTotal);

  const ClusterResult mc = cluster_contexts(mask_raw, kMaxMaskClusters);
  const ClusterResult lc = cluster_contexts(level_raw, kMaxLevelClusters);

  // Cost of the global tables on this brick's statistics, against the cost of
  // the clustered ones including their own bytes.
  double global_bits = 0.0;
  for (int i = 0; i < kMaskCtxTotal; ++i) {
    const Model& g = global[mask_model_index(i)];
    for (std::size_t sym = 0; sym < mask_raw[static_cast<std::size_t>(i)].size(); ++sym)
      if (mask_raw[static_cast<std::size_t>(i)][sym])
        global_bits += static_cast<double>(mask_raw[static_cast<std::size_t>(i)][sym]) *
                       g.bit_cost(static_cast<std::uint32_t>(sym));
  }
  for (int i = 0; i < kLevelCtxTotal; ++i) {
    const Model& g = global[static_cast<std::uint16_t>(ModelIndex::level_base + i)];
    for (std::size_t sym = 0; sym < level_raw[static_cast<std::size_t>(i)].size(); ++sym)
      if (level_raw[static_cast<std::size_t>(i)][sym])
        global_bits += static_cast<double>(level_raw[static_cast<std::size_t>(i)][sym]) *
                       g.bit_cost(static_cast<std::uint32_t>(sym));
  }

  // Two more single tables for DC and escape lengths, if they carry anything.
  auto single_gain = [&](std::uint16_t idx, std::uint32_t nsym, Model& built) -> double {
    std::uint64_t tot = 0;
    for (std::uint64_t c : counts[idx]) tot += c;
    if (tot == 0) return 0.0;
    std::vector<std::uint32_t> cc(nsym, 0);
    for (std::uint32_t i = 0; i < nsym && i < counts[idx].size(); ++i)
      cc[i] = static_cast<std::uint32_t>(std::min<std::uint64_t>(counts[idx][i], 0xffffffffu));
    built = Model::from_counts(cc);
    double g = 0.0, b = 0.0;
    for (std::uint32_t i = 0; i < nsym; ++i) {
      if (!counts[idx][i]) continue;
      g += static_cast<double>(counts[idx][i]) * global[idx].bit_cost(i);
      b += static_cast<double>(counts[idx][i]) * built.bit_cost(i);
    }
    return g - (b + table_cost_bits(distinct(counts[idx])));
  };

  Model dc_model, esc_model;
  const double dc_gain = single_gain(ModelIndex::dc_len, kLenSyms, dc_model);
  const double esc_gain = single_gain(ModelIndex::esc_len, kLenSyms, esc_model);

  const double clustered_bits = mc.bits + lc.bits;
  const double total_gain = (global_bits - clustered_bits) + std::max(0.0, dc_gain) +
                            std::max(0.0, esc_gain);

  // A margin so a brick does not pay table bytes for a fraction of a percent.
  if (total_gain < 256.0) return out;

  // --- adopt ---
  std::vector<Model> mask_models, level_models;
  for (const auto& c : mc.counts) {
    std::vector<std::uint32_t> cc(c.size());
    for (std::size_t i = 0; i < c.size(); ++i)
      cc[i] = static_cast<std::uint32_t>(std::min<std::uint64_t>(c[i], 0xffffffffu));
    mask_models.push_back(Model::from_counts_sparse(cc));
  }
  for (const auto& c : lc.counts) {
    std::vector<std::uint32_t> cc(c.size());
    for (std::size_t i = 0; i < c.size(); ++i)
      cc[i] = static_cast<std::uint32_t>(std::min<std::uint64_t>(c[i], 0xffffffffu));
    level_models.push_back(Model::from_counts_sparse(cc));
  }

  const bool use_dc = dc_gain > 0.0;
  const bool use_esc = esc_gain > 0.0;
  if (want_models) {
    for (int i = 0; i < kMaskCtxTotal; ++i)
      out.models.set(mask_model_index(i), mask_models[mc.map[static_cast<std::size_t>(i)]]);
    for (int i = 0; i < kLevelCtxTotal; ++i)
      out.models.set(static_cast<std::uint16_t>(ModelIndex::level_base + i),
                     level_models[lc.map[static_cast<std::size_t>(i)]]);
    if (use_dc) out.models.set(ModelIndex::dc_len, dc_model);
    if (use_esc) out.models.set(ModelIndex::esc_len, esc_model);
  }

  out.blob.push_back(static_cast<std::uint8_t>(mask_models.size()));
  out.blob.push_back(static_cast<std::uint8_t>(level_models.size()));
  out.blob.push_back(static_cast<std::uint8_t>((use_dc ? 1 : 0) | (use_esc ? 2 : 0)));
  for (int i = 0; i < kMaskCtxTotal; ++i) out.blob.push_back(mc.map[static_cast<std::size_t>(i)]);
  for (int i = 0; i < kLevelCtxTotal; ++i) out.blob.push_back(lc.map[static_cast<std::size_t>(i)]);
  for (const Model& m : mask_models) write_compact_table(m, out.blob);
  for (const Model& m : level_models) write_compact_table(m, out.blob);
  if (use_dc) write_compact_table(dc_model, out.blob);
  if (use_esc) write_compact_table(esc_model, out.blob);

  out.used = true;
  return out;
}

// Reconstructs the per-brick model set from its blob.
[[nodiscard]] inline bool read_brick_tables(std::span<const std::uint8_t> blob,
                                            const ModelSet& global, ModelSet& out) {
  out = global;
  std::size_t pos = 0;
  if (blob.size() < 3) return false;
  const std::size_t nmask = blob[pos++];
  const std::size_t nlevel = blob[pos++];
  const std::uint8_t extra = blob[pos++];
  if (nmask == 0 || nmask > kMaxMaskClusters) return false;
  if (nlevel == 0 || nlevel > kMaxLevelClusters) return false;
  if (pos + kMaskCtxTotal + kLevelCtxTotal > blob.size()) return false;

  std::vector<std::uint8_t> mmap(blob.begin() + static_cast<std::ptrdiff_t>(pos),
                                 blob.begin() + static_cast<std::ptrdiff_t>(pos + kMaskCtxTotal));
  pos += kMaskCtxTotal;
  std::vector<std::uint8_t> lmap(
      blob.begin() + static_cast<std::ptrdiff_t>(pos),
      blob.begin() + static_cast<std::ptrdiff_t>(pos + kLevelCtxTotal));
  pos += kLevelCtxTotal;
  for (std::uint8_t v : mmap)
    if (v >= nmask) return false;
  for (std::uint8_t v : lmap)
    if (v >= nlevel) return false;

  std::vector<Model> mask_models(nmask), level_models(nlevel);
  for (std::size_t i = 0; i < nmask; ++i)
    if (!read_compact_table(blob, pos, 256, mask_models[i])) return false;
  for (std::size_t i = 0; i < nlevel; ++i)
    if (!read_compact_table(blob, pos, kLevelSyms, level_models[i])) return false;

  for (int i = 0; i < kMaskCtxTotal; ++i)
    out.set(mask_model_index(i), mask_models[mmap[static_cast<std::size_t>(i)]]);
  for (int i = 0; i < kLevelCtxTotal; ++i)
    out.set(static_cast<std::uint16_t>(ModelIndex::level_base + i),
            level_models[lmap[static_cast<std::size_t>(i)]]);

  if (extra & 1) {
    Model m;
    if (!read_compact_table(blob, pos, kLenSyms, m)) return false;
    out.set(ModelIndex::dc_len, m);
  }
  if (extra & 2) {
    Model m;
    if (!read_compact_table(blob, pos, kLenSyms, m)) return false;
    out.set(ModelIndex::esc_len, m);
  }
  return pos == blob.size();
}

// Structured form of a brick's tables, for backends that cannot hold a whole
// ModelSet per brick.
//
// A brick has at most kMaxMaskClusters + kMaxLevelClusters + 2 distinct tables,
// so uploading those plus a model->table map costs a few kilobytes rather than
// the ~300 KB a full per-brick ModelSet would. Models not covered by the map
// (bypass, chunk flag, exponent, corrections) use the global tables.
inline constexpr std::size_t kMaxBrickTables = kMaxMaskClusters + kMaxLevelClusters + 2;
inline constexpr std::uint8_t kUseGlobalTable = 0xff;

struct BrickTableView {
  std::uint32_t ntables = 0;
  // ntables * 256 frequencies, zero-padded past each table's alphabet.
  std::vector<std::uint16_t> freqs;
  // Per model index: which table, or kUseGlobalTable.
  std::vector<std::uint8_t> model_map;
};

// Parses a table blob into upload form without materializing a ModelSet.
[[nodiscard]] inline bool parse_brick_tables(std::span<const std::uint8_t> blob,
                                             BrickTableView& out) {
  out.ntables = 0;
  out.freqs.clear();
  out.model_map.assign(ModelIndex::total, kUseGlobalTable);

  std::size_t pos = 0;
  if (blob.size() < 3) return false;
  const std::size_t nmask = blob[pos++];
  const std::size_t nlevel = blob[pos++];
  const std::uint8_t extra = blob[pos++];
  if (nmask == 0 || nmask > kMaxMaskClusters) return false;
  if (nlevel == 0 || nlevel > kMaxLevelClusters) return false;
  if (pos + kMaskCtxTotal + kLevelCtxTotal > blob.size()) return false;

  const std::uint8_t* mmap = blob.data() + pos;
  pos += kMaskCtxTotal;
  const std::uint8_t* lmap = blob.data() + pos;
  pos += kLevelCtxTotal;
  for (int i = 0; i < kMaskCtxTotal; ++i)
    if (mmap[i] >= nmask) return false;
  for (int i = 0; i < kLevelCtxTotal; ++i)
    if (lmap[i] >= nlevel) return false;

  const std::size_t ndc = (extra & 1) ? 1 : 0;
  const std::size_t nesc = (extra & 2) ? 1 : 0;
  const std::size_t total = nmask + nlevel + ndc + nesc;
  if (total > kMaxBrickTables) return false;
  out.freqs.assign(total * 256, 0);

  auto read_one = [&](std::size_t table, std::uint32_t nsym) -> bool {
    if (pos + 2 > blob.size()) return false;
    const std::size_t used =
        static_cast<std::size_t>(blob[pos]) | (static_cast<std::size_t>(blob[pos + 1]) << 8);
    pos += 2;
    if (used == 0 || used > nsym || pos + 3 * used > blob.size()) return false;
    std::uint32_t sum = 0;
    for (std::size_t i = 0; i < used; ++i) {
      const std::uint32_t sym = blob[pos];
      const std::uint32_t f = static_cast<std::uint32_t>(blob[pos + 1]) |
                              (static_cast<std::uint32_t>(blob[pos + 2]) << 8);
      pos += 3;
      if (sym >= nsym || f == 0 || out.freqs[table * 256 + sym] != 0) return false;
      out.freqs[table * 256 + sym] = static_cast<std::uint16_t>(f);
      sum += f;
    }
    return sum == kProbScale;
  };

  for (std::size_t i = 0; i < nmask; ++i)
    if (!read_one(i, 256)) return false;
  for (std::size_t i = 0; i < nlevel; ++i)
    if (!read_one(nmask + i, kLevelSyms)) return false;
  std::size_t next = nmask + nlevel;
  std::size_t dc_table = 0, esc_table = 0;
  if (ndc) {
    dc_table = next;
    if (!read_one(next++, kLenSyms)) return false;
  }
  if (nesc) {
    esc_table = next;
    if (!read_one(next++, kLenSyms)) return false;
  }
  if (pos != blob.size()) return false;

  for (int i = 0; i < kMaskCtxTotal; ++i)
    out.model_map[mask_model_index(i)] = mmap[i];
  for (int i = 0; i < kLevelCtxTotal; ++i)
    out.model_map[static_cast<std::uint16_t>(ModelIndex::level_base + i)] =
        static_cast<std::uint8_t>(nmask + lmap[i]);
  if (ndc) out.model_map[ModelIndex::dc_len] = static_cast<std::uint8_t>(dc_table);
  if (nesc) out.model_map[ModelIndex::esc_len] = static_cast<std::uint8_t>(esc_table);

  out.ntables = static_cast<std::uint32_t>(total);
  return true;
}

}  // namespace gpudct::detail
