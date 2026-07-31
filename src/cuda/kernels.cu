// CUDA decode backend (docs/ROADMAP.md M7).
//
// Two kernels, deliberately not fused:
//
//   K1  one thread per rANS stream. Entropy-decodes its stream serially and
//       writes sparse (index, level) pairs plus a per-chunk count.
//   K2  one block per chunk, 256 threads. Scatters the sparse levels into 16 KB
//       of shared memory, dequantizes, runs three separable IDCT passes with one
//       pencil per thread, and writes the reconstruction out.
//
// K1 is one thread per stream rather than one warp, because this format's parse
// is data-dependent and cannot be decoded in lane-lockstep -- see the correction
// in docs/DESIGN.md section 3.4. Parallelism comes from bricks and streams: a
// large volume has hundreds of thousands of bricks, so even P=4 saturates the
// device. Archives meant for low-latency single-brick fetches should raise P.
//
// The split costs one global round-trip of the sparse coefficients. At the rates
// this codec targets a chunk holds a few hundred nonzeros, so that is roughly
// 1 KB per chunk against 8 KB of output -- cheap enough that fusing is a tuning
// question rather than a correctness one.

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include "gpudct/types.hpp"

namespace gpudct::cuda {

// Format constants are *aliased* from the host headers, never restated.
//
// They were duplicated as literals here originally, on the theory that it kept
// this file compilable in isolation. Then kRansStates changed from 32 to 4 on
// the host and this copy did not, and the GPU decoded garbage until the
// cross-backend test caught it. A constant that defines the bitstream gets
// exactly one definition; the static_asserts below turn a future divergence into
// a compile error instead of a wrong answer.
using gpudct::kBrickChunks;
using gpudct::kBrickDim;
using gpudct::kChunkDim;
using gpudct::kChunksPerBrick;
using gpudct::kChunkVox;
using gpudct::kSubCount;
using gpudct::kSubVox;

using gpudct::detail::kProbBits;
using gpudct::detail::kProbMask;
using gpudct::detail::kProbScale;
using gpudct::detail::kRansL;
using gpudct::detail::kRansStates;

static_assert(kChunkVox == 4096, "device kernels assume a 16^3 chunk");
static_assert(kChunksPerBrick == 512, "device kernels assume 8^3 chunks per brick");
static_assert(kProbScale == 4096, "slot tables are indexed by a 12-bit probability");

// Context counts, aliased like everything else that defines the bitstream.
static constexpr int kNumBandsDev = gpudct::detail::kNumBands;
static constexpr int kMaskCtxDev = gpudct::detail::kMaskCtxCount;
static constexpr int kLevelCtxDev = gpudct::detail::kLevelCtxCount;

// Threads per K1 block, and the stride between one thread's consecutive states
// in the shared pool. Laying the pool out state-major with a thread-count stride
// means lane L always touches bank (L mod 32), so a warp's accesses to the same
// state index are conflict-free.
static constexpr std::uint32_t kK1Threads = 128;

// Batch size at which the flat K1 launch overtakes the per-brick one.
//
// Below it there are too few threads to hide the slot lookup's global latency
// and staging the tables in shared memory wins; above it there is enough
// occupancy to hide it, and the flat launch's better SM utilization wins.
// Measured on K1 time alone, P=64, three reps each:
//
//   bricks    shared    flat
//        8    5.52 ms   8.89 ms
//       27    6.02 ms   9.87 ms
//       64   11.96 ms   9.90 ms
//
// The crossover is between 27 and 64, so 32 is the threshold. Note how flat's
// time barely moves across an 8x change in batch size -- it is bound by the
// per-thread serial chain, not by throughput.
static constexpr std::uint32_t kSharedLaunchMaxBricks = 32;
static constexpr std::uint32_t kStateStride = kK1Threads;

static constexpr int kLevelEscape = static_cast<int>(gpudct::detail::kLevelEscape);
static constexpr int kLenSyms = static_cast<int>(gpudct::detail::kLenSyms);

// Flattened model set, uploaded once per archive. `slot` is the 4096-entry
// symbol lookup that makes decoding a table read rather than a search.
// Per-brick tables, when a brick carries its own.
//
// Only the distinct tables are uploaded -- at most a dozen or so per brick after
// clustering -- plus a map from model index to table. A model the map leaves as
// kUseGlobal falls back to the archive-wide table set, which is why the small,
// stable models (bypass, chunk flag, exponent) cost nothing per brick.
struct DeviceBrickTables {
  // freq and cum packed into one word, exactly as DeviceModels does and for the
  // same reason: the decoder needs both for every symbol, and two u16 arrays
  // meant two dependent global loads on the hot path. Per-brick tables are the
  // default, so this path is the hot one.
  const std::uint32_t* fc;     // [brick_table_base + table][256], freq<<16 | cum
  const std::uint8_t* slot;    // [.. ][4096]
  const std::uint8_t* map;     // [brick][model] -> table index, or 0xff
  const std::uint32_t* base;   // [brick] -> first table index for that brick
};

static constexpr std::uint8_t kUseGlobalTableDev = 0xff;

struct DeviceModels {
  // freq and cum packed into one word (freq << 16 | cum). The decoder needs both
  // for every symbol, and two separate arrays meant two dependent global loads
  // where one does.
  const std::uint32_t* fc;
  // Symbol per probability slot. u8, not u16: every alphabet here is at most 256
  // symbols, and halving the table halves the L2 footprint of the one lookup
  // that happens on literally every symbol.
  const std::uint8_t* slot;
  const std::uint32_t* offset_freq;  // per-model base into fc
  std::uint32_t model_count;
};

struct BrickDesc {
  std::uint8_t has_tables;
  const std::uint8_t* payload;
  std::uint32_t payload_size;
  std::uint32_t stream_offset[64];
  std::uint32_t stream_size[64];
  std::uint32_t streams;
  std::uint8_t mode;
};

static constexpr int kModelTotalDev = gpudct::detail::ModelIndex::total;

// --------------------------------------------------------------------------
// Device-side rANS decoder. One instance per thread; the 32 interleaved states
// live in local memory, where they still provide the instruction-level
// parallelism that the interleaving was for.
// --------------------------------------------------------------------------
struct Decoder {
  // Per-brick tables for the brick this decoder is walking; null when the brick
  // uses the archive-wide set.
  const DeviceBrickTables* bt;
  std::uint32_t bt_base;
  const std::uint8_t* bt_map;
  // Block-local copy of this brick's slot tables, or null to read them from
  // global memory.
  //
  // The slot lookup is the symbol decode's first dependent load and the one that
  // matters: 4096 bytes per table, indexed by the rANS state's low bits, so the
  // access is random and there is nothing to prefetch. At the batch sizes a
  // VRAM-resident viewer decodes -- a few bricks, a few hundred threads -- there
  // is no occupancy to hide that latency behind, and K1 becomes a chain of
  // unhidden global round trips.
  const std::uint8_t* sh_slot;

  const std::uint8_t* bytes;
  std::uint32_t size;
  std::uint32_t pos;
  std::uint32_t next;
  // Points into the block's shared-memory state pool, strided so that lanes of a
  // warp hit different banks. Keeping these in a thread-local array instead cost
  // a local-memory read and write on every symbol, because an array indexed by a
  // runtime value cannot stay in registers.
  std::uint32_t* state;
  std::uint32_t stride;
  bool failed;
  std::uint32_t why;

  __device__ bool init(const std::uint8_t* b, std::uint32_t n, std::uint32_t* state_pool,
                       std::uint32_t state_stride) {
    bt = nullptr;
    bt_base = 0;
    bt_map = nullptr;
    sh_slot = nullptr;
    bytes = b;
    size = n;
    pos = 0;
    next = 0;
    failed = false;
    why = 0;
    state = state_pool;
    stride = state_stride;
    if (n < kRansStates * 4) return false;
    for (std::uint32_t i = 0; i < kRansStates; ++i) {
      state[i * stride] = static_cast<std::uint32_t>(bytes[pos]) |
                                (static_cast<std::uint32_t>(bytes[pos + 1]) << 8) |
                                (static_cast<std::uint32_t>(bytes[pos + 2]) << 16) |
                                (static_cast<std::uint32_t>(bytes[pos + 3]) << 24);
      pos += 4;
    }
    return true;
  }

  __device__ std::uint32_t decode(const DeviceModels& m, std::uint32_t model) {
    if (failed) return 0;
    std::uint32_t x = state[next * stride];
    const std::uint32_t slot_v = x & kProbMask;

    std::uint32_t sym, freq, cum;
    const std::uint8_t t = bt_map ? bt_map[model] : kUseGlobalTableDev;
    if (t == kUseGlobalTableDev) {
      sym = m.slot[model * kProbScale + slot_v];
      const std::uint32_t fc = m.fc[m.offset_freq[model] + sym];
      freq = fc >> 16;
      cum = fc & 0xffffu;
    } else {
      const std::size_t ti = bt_base + t;
      sym = sh_slot ? sh_slot[static_cast<std::size_t>(t) * kProbScale + slot_v]
                    : bt->slot[ti * kProbScale + slot_v];
      const std::uint32_t bfc = bt->fc[ti * 256 + sym];
      freq = bfc >> 16;
      cum = bfc & 0xffffu;
    }
    x = freq * (x >> kProbBits) + slot_v - cum;
    while (x < kRansL) {
      if (pos >= size) {
        failed = true;
        why = 3;
        return 0;
      }
      x = (x << 8) | bytes[pos++];
    }
    state[next * stride] = x;
    if (++next == kRansStates) next = 0;
    return sym;
  }

  // Model 0 is always exactly 50/50, so a bypass bit needs no tables at all:
  // the slot's top bit is the value and both halves are the same width. Sign
  // bits and mantissa bits are a large share of all symbols coded, so removing
  // two dependent global loads from each of them is the single cheapest win
  // available in this kernel.
  __device__ std::uint32_t decode_bypass(const DeviceModels&) {
    if (failed) return 0;
    std::uint32_t x = state[next * stride];
    const std::uint32_t slot_v = x & kProbMask;
    const std::uint32_t bit = slot_v >> (kProbBits - 1);
    constexpr std::uint32_t kHalf = kProbScale / 2;
    x = kHalf * (x >> kProbBits) + (slot_v & (kHalf - 1));
    while (x < kRansL) {
      if (pos >= size) {
        failed = true;
        why = 4;
        return 0;
      }
      x = (x << 8) | bytes[pos++];
    }
    state[next * stride] = x;
    if (++next == kRansStates) next = 0;
    return bit;
  }
};

// --------------------------------------------------------------------------
// Index maths, mirroring src/core/transform.hpp exactly.
// --------------------------------------------------------------------------
__device__ __forceinline__ int band_of_subblock_index(int sb) {
  const int su = sb & 3, sv = (sb >> 2) & 3, sw = (sb >> 4) & 3;
  const int r = su + sv + sw;
  return (r == 0) ? 0 : (r <= 2 ? 1 : (r <= 4 ? 2 : 3));
}

__device__ __forceinline__ int coeff_of_subblock_bit(int sb, int bit) {
  const int su = sb & 3, sv = (sb >> 2) & 3, sw = (sb >> 4) & 3;
  const int bu = bit & 3, bv = (bit >> 2) & 3, bw = (bit >> 4) & 3;
  return (((sw * 4 + bw) * 16) + (sv * 4 + bv)) * 16 + (su * 4 + bu);
}

// Model indices, taken from the host layout rather than restated -- same reason
// as the constants above.
struct MI {
  using H = gpudct::detail::ModelIndex;
  static constexpr int bypass = H::bypass;
  static constexpr int exponent = H::exponent;
  static constexpr int chunk_nonzero = H::chunk_nonzero;
  static constexpr int l1_base = H::l1_base;
  static constexpr int l2_base = H::l2_base;
  static constexpr int level_base = H::level_base;
  static constexpr int dc_len = H::dc_len;
  static constexpr int esc_len = H::esc_len;
  __device__ static int level(int band, int ctx) {
    return level_base + band * gpudct::detail::kLevelCtxCount + ctx;
  }
  __device__ static int l1(int j, int ctx) { return l1_base + j * kMaskCtxDev + ctx; }
  __device__ static int l2(int band, int ctx) { return l2_base + band * kMaskCtxDev + ctx; }
};

static_assert(MI::bypass == 0, "the device bypass fast path assumes model 0");

// Context helpers, aliased from the host so they cannot drift (the same lesson
// as the constants above -- a silent mismatch here decodes garbage).
__device__ __forceinline__ int level_ctx_of_prev(int sum) {
  // Mirrors detail::level_ctx_of_prev. nvcc will not call a host constexpr from
  // device code without --expt-relaxed-constexpr, and this defines the
  // bitstream, so the equivalence is asserted below rather than assumed.
  if (sum == 0) return 0;
  if (sum == 1) return 1;
  if (sum == 2) return 2;
  if (sum <= 4) return 3;
  if (sum <= 8) return 4;
  return 5;
}
static_assert(gpudct::detail::level_ctx_of_prev(0) == 0 &&
                  gpudct::detail::level_ctx_of_prev(1) == 1 &&
                  gpudct::detail::level_ctx_of_prev(2) == 2 &&
                  gpudct::detail::level_ctx_of_prev(4) == 3 &&
                  gpudct::detail::level_ctx_of_prev(99) == 5 &&
                  gpudct::detail::kLevelCtxCount == 6,
              "device level context must mirror the host one");
static_assert(gpudct::detail::mask_ctx_of_prev(0) == 0 &&
                  gpudct::detail::mask_ctx_of_prev(1) == 1 &&
                  gpudct::detail::mask_ctx_of_prev(3) == 2 &&
                  gpudct::detail::mask_ctx_of_prev(255) == 5 &&
                  gpudct::detail::kMaskCtxCount == 6,
              "device mask context must mirror the host one");
__device__ __forceinline__ int mask_ctx_of_prev(unsigned prev_byte) {
  const int pc = __popc(prev_byte);
  return pc > 5 ? 5 : pc;
}

// Magnitude coded as a modelled bit-length plus bypass mantissa.
__device__ __forceinline__ bool read_len_mag(Decoder& d, const DeviceModels& m, int model,
                                             std::uint32_t& mag) {
  const std::uint32_t nb = d.decode(m, model);
  if (d.failed || nb == 0 || nb >= kLenSyms) return false;
  std::uint32_t v = 1;
  for (std::uint32_t i = 1; i < nb; ++i) v = (v << 1) | d.decode_bypass(m);
  mag = v;
  return !d.failed;
}

// --------------------------------------------------------------------------
// K1: entropy decode. Grid is (streams_per_brick, brick_count); each thread owns
// one stream and walks its chunks in increasing index order.
// --------------------------------------------------------------------------
// Expands uploaded frequencies into cumulative and slot tables.
//
// One block per table. Building these on the host would mean writing ~4 KB per
// table per brick across the PCIe bus and through the CPU, for data that is
// fully determined by the frequencies already being sent.
__global__ void k0_build_brick_tables(const std::uint16_t* __restrict freqs,
                                      std::uint32_t* __restrict fc,
                                      std::uint8_t* __restrict slot,
                                      std::uint32_t ntables) {
  const std::uint32_t t = blockIdx.x;
  if (t >= ntables) return;
  __shared__ std::uint16_t sh_cum[257];

  if (threadIdx.x == 0) {
    std::uint32_t running = 0;
    for (int i = 0; i < 256; ++i) {
      sh_cum[i] = static_cast<std::uint16_t>(running);
      running += freqs[static_cast<std::size_t>(t) * 256 + i];
    }
    sh_cum[256] = static_cast<std::uint16_t>(running);
  }
  __syncthreads();

  for (int i = static_cast<int>(threadIdx.x); i < 256; i += blockDim.x)
    fc[static_cast<std::size_t>(t) * 256 + i] =
        (static_cast<std::uint32_t>(freqs[static_cast<std::size_t>(t) * 256 + i]) << 16) |
        sh_cum[i];

  // Fill slot[] by binary searching the cumulative table: a linear scan per slot
  // would be 256x the work and this runs once per table, not per symbol.
  for (int s = static_cast<int>(threadIdx.x); s < static_cast<int>(kProbScale);
       s += blockDim.x) {
    int lo = 0, hi = 255;
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (sh_cum[mid] <= s) lo = mid;
      else hi = mid - 1;
    }
    slot[static_cast<std::size_t>(t) * kProbScale + s] = static_cast<std::uint8_t>(lo);
  }
}

// The per-stream decode body, shared by both K1 launches.
//
// Split out so the flat launch (one thread per brick-stream pair, best for bulk
// throughput) and the per-brick launch (one block per brick, which can stage the
// brick's slot tables in shared memory) cannot drift apart. They differ only in
// how a thread finds its stream and where the slot tables live.
__device__ void k1_decode_stream(Decoder& d, const DeviceModels& models, std::uint32_t b,
                                 std::uint32_t s, std::uint32_t streams,
                                 std::int32_t max_level, std::int16_t* __restrict out_levels,
                                 std::uint16_t* __restrict out_indices,
                                 std::uint32_t* __restrict out_counts,
                                 std::uint32_t max_nonzero_per_chunk,
                                 std::uint32_t* __restrict error_flag) {
  std::uint32_t last_ci = 0;
  for (int ci = static_cast<int>(s); ci < kChunksPerBrick;
       ci += static_cast<int>(streams)) {
    last_ci = static_cast<std::uint32_t>(ci);
    const std::size_t chunk_slot =
        (static_cast<std::size_t>(b) * kChunksPerBrick + ci) * max_nonzero_per_chunk;
    std::uint32_t n = 0;

    const std::uint32_t nonzero = d.decode(models, MI::chunk_nonzero);
    if (d.failed) break;
    if (nonzero == 0) {
      out_counts[static_cast<std::size_t>(b) * kChunksPerBrick + ci] = 0;
      continue;
    }
    (void)d.decode(models, MI::exponent);  // reserved; always 0 today

    unsigned long long l1 = 0;
    int mctx = 0;
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t byte = d.decode(models, MI::l1(j, mctx));
      l1 |= (static_cast<unsigned long long>(byte) << (8 * j));
      mctx = mask_ctx_of_prev(byte);
    }
    if (d.failed) break;

    for (int sb = 0; sb < kSubCount && !d.failed; ++sb) {
      if (!((l1 >> sb) & 1ull)) continue;
      const int band = band_of_subblock_index(sb);

      unsigned long long mask = 0;
      mctx = 0;
      for (int j = 0; j < 8; ++j) {
        const std::uint32_t byte = d.decode(models, MI::l2(band, mctx));
        mask |= (static_cast<unsigned long long>(byte) << (8 * j));
        mctx = mask_ctx_of_prev(byte);
      }
      if (d.failed) break;

      int nb[kSubVox] = {};
      for (int bit = 0; bit < kSubVox; ++bit) {
        if (!((mask >> bit) & 1ull)) continue;
        const int idx = coeff_of_subblock_bit(sb, bit);

        int mag;
        if (idx == 0) {
          std::uint32_t m32 = 0;
          if (!read_len_mag(d, models, MI::dc_len, m32)) { if (!d.why) d.why = 5; break; }
          if (static_cast<std::int32_t>(m32) > max_level) { d.failed = true; d.why = 6; break; }
          mag = static_cast<int>(m32);
        } else {
          const int bu = bit & 3, bv = (bit >> 2) & 3, bw = (bit >> 4) & 3;
          int nsum = 0;
          if (bu > 0) nsum += nb[bit - 1];
          if (bv > 0) nsum += nb[bit - 4];
          if (bw > 0) nsum += nb[bit - 16];
          const std::uint32_t v = d.decode(models, MI::level(band, level_ctx_of_prev(nsum)));
          if (d.failed) break;
          if (v < kLevelEscape) {
            mag = static_cast<int>(v) + 1;
          } else {
            std::uint32_t extra = 0;
            if (!read_len_mag(d, models, MI::esc_len, extra)) { if (!d.why) d.why = 7; break; }
            mag = static_cast<int>(extra) + 15;
            if (mag > max_level) { d.failed = true; d.why = 8; break; }
          }
        }
        const std::uint32_t sign = d.decode_bypass(models);
        if (d.failed) break;

        if (n < max_nonzero_per_chunk) {
          out_indices[chunk_slot + n] = static_cast<std::uint16_t>(idx);
          out_levels[chunk_slot + n] =
              static_cast<std::int16_t>(sign ? -mag : mag);
          ++n;
        } else {
          // The sparse buffer is sized for typical chunk occupancy, not the
          // worst case, so overflow is a legitimate outcome the host retries --
          // it must be distinguishable from a corrupt stream.
          atomicExch(error_flag, 2u);
          d.failed = true;
          d.why = 9;
          break;
        }
        nb[bit] = mag;
      }
    }
    out_counts[static_cast<std::size_t>(b) * kChunksPerBrick + ci] = n;
  }

  // Do not clobber an overflow report with a generic failure.
  if (d.failed) {
    atomicCAS(error_flag, 0u, 1u);
    atomicCAS(error_flag + 1, 0u, 16u | (d.why << 4) | (b << 8) | (last_ci << 20));
  }
}

__global__ void k1_entropy_decode(const BrickDesc* __restrict bricks,
                                  DeviceModels models, DeviceBrickTables btables,
                                  std::int32_t max_level,
                                  std::int16_t* __restrict out_levels,
                                  std::uint16_t* __restrict out_indices,
                                  std::uint32_t* __restrict out_counts,
                                  std::uint32_t max_nonzero_per_chunk,
                                  std::uint32_t streams_per_brick, std::uint32_t brick_count,
                                  std::uint32_t* __restrict error_flag) {
  // One thread per (brick, stream) pair, flattened. A 2-D grid keyed on the
  // stream index wasted most of every warp whenever P was smaller than the block
  // width, which is the common case.
  __shared__ std::uint32_t sh_state[kRansStates * kK1Threads];

  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t P = streams_per_brick;
  const std::uint32_t b = tid / P;
  const std::uint32_t s = tid % P;
  if (b >= brick_count) return;

  const BrickDesc& bd = bricks[b];
  if (s >= bd.streams) return;
  if (bd.mode != 0) return;  // raw bricks are handled on the host

  Decoder d;
  if (!d.init(bd.payload + bd.stream_offset[s], bd.stream_size[s], sh_state + threadIdx.x,
              kK1Threads)) {
    atomicExch(error_flag, 1u);
    return;
  }
  if (bd.has_tables) {
    d.bt = &btables;
    d.bt_base = btables.base[b];
    d.bt_map = btables.map + static_cast<std::size_t>(b) * kModelTotalDev;
  }

  k1_decode_stream(d, models, b, s, bd.streams, max_level, out_levels, out_indices, out_counts,
                   max_nonzero_per_chunk, error_flag);
}

// K1, one block per brick, with the brick's slot tables staged in shared memory.
//
// This is the low-latency launch. The flat launch above is right when there are
// thousands of bricks in flight and the device is saturated; it is wrong when a
// viewer asks for eight bricks, because then there are only `bricks * P` threads
// -- a few hundred -- and every symbol's slot lookup is an unhidden global round
// trip. Staging the tables costs one coalesced copy per block and turns that
// lookup into a shared-memory read for the whole brick.
//
// Dynamic shared memory holds the slot tables followed by the rANS state pool.
// A brick carries at most kMaxBrickTables tables, so the host checks the size
// fits the device before choosing this launch and falls back if it does not.
__global__ void k1_entropy_decode_shared(const BrickDesc* __restrict bricks,
                                         DeviceModels models, DeviceBrickTables btables,
                                         const std::uint32_t* __restrict tab_count,
                                         std::int32_t max_level,
                                         std::int16_t* __restrict out_levels,
                                         std::uint16_t* __restrict out_indices,
                                         std::uint32_t* __restrict out_counts,
                                         std::uint32_t max_nonzero_per_chunk,
                                         std::uint32_t brick_count,
                                         std::uint32_t* __restrict error_flag) {
  extern __shared__ std::uint8_t sh_raw[];

  const std::uint32_t b = blockIdx.x;
  if (b >= brick_count) return;
  const BrickDesc& bd = bricks[b];

  // Every thread must reach the barrier, so the staging loop runs before any
  // early return -- including for raw bricks and for threads past bd.streams.
  const std::uint32_t nt = (bd.has_tables && bd.mode == 0) ? tab_count[b] : 0;
  const std::size_t slot_bytes = static_cast<std::size_t>(nt) * kProbScale;
  if (nt > 0) {
    const std::uint8_t* src = btables.slot + static_cast<std::size_t>(btables.base[b]) * kProbScale;
    for (std::size_t i = threadIdx.x; i < slot_bytes; i += blockDim.x) sh_raw[i] = src[i];
  }
  // The state pool follows the tables, aligned to 4 bytes.
  const std::size_t state_off = (slot_bytes + 3u) & ~std::size_t{3};
  std::uint32_t* sh_state = reinterpret_cast<std::uint32_t*>(sh_raw + state_off);
  __syncthreads();

  const std::uint32_t s = threadIdx.x;
  if (bd.mode != 0) return;  // raw bricks are handled on the host
  if (s >= bd.streams) return;

  Decoder d;
  if (!d.init(bd.payload + bd.stream_offset[s], bd.stream_size[s], sh_state + threadIdx.x,
              blockDim.x)) {
    atomicExch(error_flag, 1u);
    return;
  }
  if (bd.has_tables) {
    d.bt = &btables;
    d.bt_base = btables.base[b];
    d.bt_map = btables.map + static_cast<std::size_t>(b) * kModelTotalDev;
    if (nt > 0) d.sh_slot = sh_raw;
  }

  k1_decode_stream(d, models, b, s, bd.streams, max_level, out_levels, out_indices, out_counts,
                   max_nonzero_per_chunk, error_flag);
}

// --------------------------------------------------------------------------
// K2: dequantize and inverse transform. One block per chunk, 256 threads, one
// pencil per thread per pass.
// --------------------------------------------------------------------------
__constant__ float c_alpha_inv[16];
__constant__ float c_s16[8];
__constant__ float c_s8[4];
__constant__ float c_s4[2];
__constant__ float c_s2[1];

// Inverse Lee butterfly on one pencil held in registers.
//
// Mirrors lee_inv in src/core/dct16.hpp step for step -- same recursion, same
// operation order, same expressions -- so the GPU reconstruction stays within
// the tolerance docs/QUALITY.md section 4.2 allows. The recursion is on a
// template parameter, so it is fully unrolled at compile time and costs no
// stack frames.
template <int N>
__device__ __forceinline__ const float* lee_s();
template <> __device__ __forceinline__ const float* lee_s<16>() { return c_s16; }
template <> __device__ __forceinline__ const float* lee_s<8>() { return c_s8; }
template <> __device__ __forceinline__ const float* lee_s<4>() { return c_s4; }
template <> __device__ __forceinline__ const float* lee_s<2>() { return c_s2; }

template <int N>
struct LeeInv {
  __device__ __forceinline__ static void run(float* v) {
    constexpr int H = N / 2;
    float g[H], h[H];
    for (int k = 0; k < H; ++k) g[k] = v[2 * k];
    h[H - 1] = v[N - 1];
    for (int k = H - 2; k >= 0; --k) h[k] = v[2 * k + 1] - h[k + 1];

    LeeInv<H>::run(g);
    LeeInv<H>::run(h);

    const float* s = lee_s<N>();
    for (int k = 0; k < H; ++k) {
      const float d = h[k] * (0.5f / s[k]);
      const float m = g[k] * 0.5f;
      v[k] = m + d;
      v[N - 1 - k] = m - d;
    }
  }
};

template <>
struct LeeInv<1> {
  __device__ __forceinline__ static void run(float*) {}
};

__device__ __forceinline__ void idct16_pencil(float* v) {
  for (int u = 0; u < 16; ++u) v[u] *= c_alpha_inv[u];
  LeeInv<16>::run(v);
}

template <typename T>
__global__ void k2_inverse_transform(const std::int16_t* __restrict levels,
                                     const std::uint16_t* __restrict indices,
                                     const std::uint32_t* __restrict counts,
                                     std::uint32_t max_nonzero_per_chunk,
                                     const float* __restrict quant, float scale, float offset,
                                     float lo, float hi, T* __restrict out_bricks) {
  __shared__ float sh[kChunkVox];

  const std::uint32_t chunk = blockIdx.x;
  const int tid = static_cast<int>(threadIdx.x);

  for (int i = tid; i < kChunkVox; i += blockDim.x) sh[i] = 0.0f;
  __syncthreads();

  const std::uint32_t n = counts[chunk];
  const std::size_t base = static_cast<std::size_t>(chunk) * max_nonzero_per_chunk;
  for (std::uint32_t k = tid; k < n; k += blockDim.x) {
    const std::uint32_t idx = indices[base + k];
    sh[idx] = static_cast<float>(levels[base + k]) * quant[idx];
  }
  __syncthreads();

  float v[16];

  // z pass: 256 pencils, one per (y, x).
  {
    const int yx = tid;
    for (int z = 0; z < 16; ++z) v[z] = sh[z * 256 + yx];
    idct16_pencil(v);
    __syncthreads();
    for (int z = 0; z < 16; ++z) sh[z * 256 + yx] = v[z];
  }
  __syncthreads();

  // y pass: one per (z, x).
  {
    const int z = tid / 16, x = tid % 16;
    for (int y = 0; y < 16; ++y) v[y] = sh[z * 256 + y * 16 + x];
    idct16_pencil(v);
    __syncthreads();
    for (int y = 0; y < 16; ++y) sh[z * 256 + y * 16 + x] = v[y];
  }
  __syncthreads();

  // x pass: one per (z, y), contiguous.
  {
    const int zy = tid;
    for (int x = 0; x < 16; ++x) v[x] = sh[zy * 16 + x];
    idct16_pencil(v);
    __syncthreads();
    for (int x = 0; x < 16; ++x) sh[zy * 16 + x] = v[x];
  }
  __syncthreads();

  // Convert to the output dtype here rather than shipping floats back to the
  // host: the working buffer is 4x the size of a u8 brick, and it would cross
  // PCIe for no reason.
  const std::uint32_t brick = chunk / kChunksPerBrick;
  const std::uint32_t ci = chunk % kChunksPerBrick;
  const int cx = ci % kBrickChunks;
  const int cy = (ci / kBrickChunks) % kBrickChunks;
  const int cz = ci / (kBrickChunks * kBrickChunks);
  T* dst = out_bricks + static_cast<std::size_t>(brick) * kBrickDim * kBrickDim * kBrickDim;
  const float inv_scale = 1.0f / scale;

  for (int i = tid; i < kChunkVox; i += blockDim.x) {
    const int x = i % 16, y = (i / 16) % 16, z = i / 256;
    const std::size_t o =
        (static_cast<std::size_t>(cz * 16 + z) * kBrickDim + (cy * 16 + y)) * kBrickDim +
        (cx * 16 + x);
    const float w = (sh[i] - offset) * inv_scale;
    if constexpr (sizeof(T) == 4 && !std::is_integral<T>::value) {
      dst[o] = w;
    } else {
      const float c = fminf(fmaxf(w, lo), hi);
      dst[o] = static_cast<T>(__float2int_rn(c));
    }
  }
}

// As above, but writing straight into the output volume rather than a padded
// per-brick buffer.
//
// The per-brick form left the host to crop 512 padded bricks into the volume,
// which is on the order of a million short memcpys -- expensive anywhere, and
// especially so when the CPU is busy with other work. Writing final coordinates
// here turns the readback into one contiguous transfer and removes the host
// stage entirely. Voxels outside the volume are simply not written.
template <typename T>
__global__ void k2_inverse_transform_direct(const std::int16_t* __restrict levels,
                                            const std::uint16_t* __restrict indices,
                                            const std::uint32_t* __restrict counts,
                                            std::uint32_t max_nonzero_per_chunk,
                                            const float* __restrict quant, float scale,
                                            float offset, float lo, float hi,
                                            const std::uint32_t* __restrict brick_origin,
                                            std::uint32_t dim_x, std::uint32_t dim_y,
                                            std::uint32_t dim_z, T* __restrict volume) {
  __shared__ float sh[kChunkVox];

  const std::uint32_t chunk = blockIdx.x;
  const int tid = static_cast<int>(threadIdx.x);

  for (int i = tid; i < kChunkVox; i += blockDim.x) sh[i] = 0.0f;
  __syncthreads();

  const std::uint32_t n = counts[chunk];
  const std::size_t base = static_cast<std::size_t>(chunk) * max_nonzero_per_chunk;
  for (std::uint32_t k = tid; k < n; k += blockDim.x) {
    const std::uint32_t idx = indices[base + k];
    sh[idx] = static_cast<float>(levels[base + k]) * quant[idx];
  }
  __syncthreads();

  float v[16];
  {
    const int yx = tid;
    for (int z = 0; z < 16; ++z) v[z] = sh[z * 256 + yx];
    idct16_pencil(v);
    __syncthreads();
    for (int z = 0; z < 16; ++z) sh[z * 256 + yx] = v[z];
  }
  __syncthreads();
  {
    const int z = tid / 16, x = tid % 16;
    for (int y = 0; y < 16; ++y) v[y] = sh[z * 256 + y * 16 + x];
    idct16_pencil(v);
    __syncthreads();
    for (int y = 0; y < 16; ++y) sh[z * 256 + y * 16 + x] = v[y];
  }
  __syncthreads();
  {
    const int zy = tid;
    for (int x = 0; x < 16; ++x) v[x] = sh[zy * 16 + x];
    idct16_pencil(v);
    __syncthreads();
    for (int x = 0; x < 16; ++x) sh[zy * 16 + x] = v[x];
  }
  __syncthreads();

  const std::uint32_t brick = chunk / kChunksPerBrick;
  const std::uint32_t ci = chunk % kChunksPerBrick;
  const std::uint32_t ox = brick_origin[brick * 3 + 0] +
                           static_cast<std::uint32_t>((ci % kBrickChunks) * 16);
  const std::uint32_t oy = brick_origin[brick * 3 + 1] +
                           static_cast<std::uint32_t>(((ci / kBrickChunks) % kBrickChunks) * 16);
  const std::uint32_t oz =
      brick_origin[brick * 3 + 2] +
      static_cast<std::uint32_t>((ci / (kBrickChunks * kBrickChunks)) * 16);

  const float inv_scale = 1.0f / scale;
  for (int i = tid; i < kChunkVox; i += blockDim.x) {
    const std::uint32_t x = ox + static_cast<std::uint32_t>(i % 16);
    const std::uint32_t y = oy + static_cast<std::uint32_t>((i / 16) % 16);
    const std::uint32_t z = oz + static_cast<std::uint32_t>(i / 256);
    if (x >= dim_x || y >= dim_y || z >= dim_z) continue;  // padding

    const float w = (sh[i] - offset) * inv_scale;
    const std::size_t o = (static_cast<std::size_t>(z) * dim_y + y) * dim_x + x;
    if constexpr (sizeof(T) == 4 && !std::is_integral<T>::value) {
      volume[o] = w;
    } else {
      const float c = fminf(fmaxf(w, lo), hi);
      volume[o] = static_cast<T>(__float2int_rn(c));
    }
  }
}

// --------------------------------------------------------------------------
// Chunk-boundary deblocking on the device (docs/DESIGN.md 3.8)
//
// The filter used to run host-side on the decoded volume, which on the CUDA
// path meant reading a gigabyte back and then walking it on the CPU: measured,
// that cost 2.8x the entire GPU decode. In direct mode the volume is already
// here, and every part of the filter is trivially parallel -- boundary planes
// are 16 voxels apart and the filter reads at most two voxels either side, so
// no two planes on an axis interact.
//
// The axis order is fixed and is part of the reconstruction, so the three
// passes are three launches rather than one.
// --------------------------------------------------------------------------

template <class T>
__device__ inline float db_get(const T* __restrict v, std::size_t o, float scale, float offset) {
  return static_cast<float>(v[o]) * scale + offset;
}

template <class T>
__device__ inline void db_set(T* __restrict v, std::size_t o, float w, float inv_scale,
                              float offset, float lo, float hi) {
  const float s = (w - offset) * inv_scale;
  if constexpr (sizeof(T) == 4 && !std::is_integral<T>::value) {
    v[o] = s;
  } else {
    v[o] = static_cast<T>(__float2int_rn(fminf(fmaxf(s, lo), hi)));
  }
}

__device__ inline std::size_t db_index(std::uint32_t x, std::uint32_t y, std::uint32_t z,
                                       std::uint32_t dx, std::uint32_t dy) {
  return (static_cast<std::size_t>(z) * dy + y) * dx + x;
}

// Mean |first difference| along `ax` over interior steps, on a stride-16 lattice
// of lines. Mirrors the host implementation, including the stride: the estimate
// feeds the filter thresholds, so the two paths have to agree.
template <class T>
__global__ void k3_activity(const T* __restrict volume, std::uint32_t dx, std::uint32_t dy,
                            std::uint32_t dz, int ax, float scale, float offset,
                            double* __restrict acc, unsigned long long* __restrict cnt) {
  const std::uint32_t lim[3] = {dx, dy, dz};
  const int a1 = (ax + 1) % 3, a2 = (ax + 2) % 3;
  const std::uint32_t stride = 16;
  const std::uint32_t n1 = (lim[a1] + stride - 1) / stride;
  const std::uint32_t n2 = (lim[a2] + stride - 1) / stride;
  const std::uint64_t total = static_cast<std::uint64_t>(n1) * n2;

  double local = 0.0;
  unsigned long long local_n = 0;
  for (std::uint64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
       t += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    std::uint32_t c[3] = {0, 0, 0};
    c[a1] = static_cast<std::uint32_t>(t / n2) * stride;
    c[a2] = static_cast<std::uint32_t>(t % n2) * stride;

    float prev = 0.0f;
    for (std::uint32_t k = 0; k < lim[ax]; ++k) {
      std::uint32_t q[3] = {c[0], c[1], c[2]};
      q[ax] = k;
      const float cur = db_get(volume, db_index(q[0], q[1], q[2], dx, dy), scale, offset);
      if (k > 0 && (k - 1) % 16 != 15) {
        local += fabsf(cur - prev);
        ++local_n;
      }
      prev = cur;
    }
  }
  if (local_n) {
    atomicAdd(acc, local);
    atomicAdd(cnt, local_n);
  }
}

// One axis pass. `alpha`, `beta` and `clip` are the host-derived thresholds.
template <class T>
__global__ void k4_deblock_axis(T* __restrict volume, std::uint32_t dx, std::uint32_t dy,
                                std::uint32_t dz, int ax, float scale, float offset, float lo,
                                float hi, float alpha, float beta, float clip) {
  const std::uint32_t lim[3] = {dx, dy, dz};
  if (lim[ax] < 4) return;
  const std::uint32_t planes = (lim[ax] - 1) / 16;
  if (planes == 0) return;

  const int a1 = (ax + 1) % 3, a2 = (ax + 2) % 3;
  const std::uint64_t per_plane = static_cast<std::uint64_t>(lim[a1]) * lim[a2];
  const std::uint64_t total = per_plane * planes;
  const float inv_scale = 1.0f / scale;

  for (std::uint64_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
       t += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    const std::uint32_t pi = static_cast<std::uint32_t>(t / per_plane);
    const std::uint64_t r = t % per_plane;
    const std::uint32_t b = (pi + 1) * 16;
    if (b < 2 || b + 1 >= lim[ax]) continue;

    std::uint32_t c[3] = {0, 0, 0};
    c[a1] = static_cast<std::uint32_t>(r / lim[a2]);
    c[a2] = static_cast<std::uint32_t>(r % lim[a2]);

    std::size_t off[4];
    for (int j = 0; j < 4; ++j) {
      std::uint32_t q[3] = {c[0], c[1], c[2]};
      q[ax] = b - 2 + static_cast<std::uint32_t>(j);
      off[j] = db_index(q[0], q[1], q[2], dx, dy);
    }
    const float p1 = db_get(volume, off[0], scale, offset);
    float p0 = db_get(volume, off[1], scale, offset);
    float q0 = db_get(volume, off[2], scale, offset);
    const float q1 = db_get(volume, off[3], scale, offset);

    const float d = q0 - p0;
    if (fabsf(d) >= alpha) continue;
    if (fabsf(p1 - p0) >= beta) continue;
    if (fabsf(q1 - q0) >= beta) continue;

    float delta = (4.0f * d + (p1 - q1)) * 0.125f;
    delta = fminf(fmaxf(delta, -clip), clip);
    p0 += delta;
    q0 -= delta;
    db_set(volume, off[1], p0, inv_scale, offset, lo, hi);
    db_set(volume, off[2], q0, inv_scale, offset, lo, hi);
  }
}

}  // namespace gpudct::cuda
