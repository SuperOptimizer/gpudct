// Whole-volume encode / decode over the brick + chunk structure.
//
// This is the scalar reference implementation. It is the oracle every other
// backend is validated against (docs/QUALITY.md section 4.2), so it favours
// being obviously correct over being fast; the SIMD and CUDA paths are where
// speed lives.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <type_traits>
#include <thread>

#include "brick_tables.hpp"
#include "chunk_codec.hpp"
#include "correction.hpp"
#include "train.hpp"
#include "deblock.hpp"
#include "format.hpp"
#include "gpudct/gpudct.hpp"
#include "quant.hpp"
#include "rdo.hpp"
#include "transform.hpp"
#ifdef GPUDCT_HAVE_CUDA
#include "cuda/cuda_backend.hpp"
#endif
#include "cpu/transform_simd.hpp"

namespace gpudct {
namespace {

using namespace detail;

// --------------------------------------------------------------------------
// Voxel access. f32 is the widest thing we handle (docs/DESIGN.md R7), so u32
// and s32 lose their low bits here -- deliberately, since a codec quantizing to
// a few hundred levels has no use for bit 25.
// --------------------------------------------------------------------------
[[nodiscard]] inline float load_voxel(const void* base, DType t, std::size_t i) {
  switch (t) {
    case DType::u8:  return static_cast<float>(static_cast<const std::uint8_t*>(base)[i]);
    case DType::s8:  return static_cast<float>(static_cast<const std::int8_t*>(base)[i]);
    case DType::u16: return static_cast<float>(static_cast<const std::uint16_t*>(base)[i]);
    case DType::s16: return static_cast<float>(static_cast<const std::int16_t*>(base)[i]);
    case DType::u32: return static_cast<float>(static_cast<const std::uint32_t*>(base)[i]);
    case DType::s32: return static_cast<float>(static_cast<const std::int32_t*>(base)[i]);
    case DType::f32: return static_cast<const float*>(base)[i];
  }
  return 0.0f;
}

inline void store_voxel(void* base, DType t, std::size_t i, float v) {
  const float lo = dtype_min(t), hi = dtype_max(t);
  const float c = std::clamp(v, lo, hi);
  switch (t) {
    case DType::u8:  static_cast<std::uint8_t*>(base)[i]  = static_cast<std::uint8_t>(std::lrintf(c)); break;
    case DType::s8:  static_cast<std::int8_t*>(base)[i]   = static_cast<std::int8_t>(std::lrintf(c)); break;
    case DType::u16: static_cast<std::uint16_t*>(base)[i] = static_cast<std::uint16_t>(std::lrintf(c)); break;
    case DType::s16: static_cast<std::int16_t*>(base)[i]  = static_cast<std::int16_t>(std::lrintf(c)); break;
    case DType::u32: static_cast<std::uint32_t*>(base)[i] = static_cast<std::uint32_t>(std::llrintf(c)); break;
    case DType::s32: static_cast<std::int32_t*>(base)[i]  = static_cast<std::int32_t>(std::llrintf(c)); break;
    case DType::f32: static_cast<float*>(base)[i] = v; break;
  }
}

// Centre the working domain on zero so the DC coefficient stays small, and map
// wide dtypes into a comparable numeric range so one quant matrix serves all of
// them. work = raw * scale + offset.
//
// 8- and 16-bit types are only shifted, never scaled: their values are already
// exact in f32 and rescaling them would introduce rounding for no benefit.
// 32-bit and float types are measured and mapped, because a raw s32 near 2^31
// has f32 spacing in the hundreds and its DC coefficient (64x the voxel scale,
// divided by the quantizer) would run straight past what an int level can hold.
void derive_mapping(const void* data, Dims dims, DType t, float& scale, float& offset) {
  scale = 1.0f;
  offset = 0.0f;
  switch (t) {
    case DType::u8:  offset = -128.0f; return;
    case DType::u16: offset = -32768.0f; return;
    case DType::s8:
    case DType::s16: return;
    case DType::u32:
    case DType::s32:
    case DType::f32: break;
  }

  const std::size_t n = dims.voxels();
  float lo = load_voxel(data, t, 0), hi = lo;
  for (std::size_t i = 1; i < n; ++i) {
    const float v = load_voxel(data, t, i);
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  const float range = hi - lo;
  // A constant volume needs no scaling, and dividing by its zero range would be
  // the first thing a fuzzer found.
  scale = (range > 0.0f) ? (65535.0f / range) : 1.0f;
  offset = -lo * scale - 32768.0f;
}

// Which transform implementation a backend uses. The SIMD kernels are
// bit-identical to the scalar ones (see cpu/transform_simd.hpp), so selecting
// between them changes speed and nothing else -- archives and reconstructions
// match byte for byte either way.
struct TransformOps {
  void (*fwd)(const float*, float*);
  void (*inv)(const float*, float*);
  // Inverse with a nonzero-count hint; see inverse_chunk_simd_hint.
  void (*inv_hint)(const float*, float*, int);
};

inline void inverse_chunk_hint_scalar(const float* in, float* out, int) {
  inverse_chunk(in, out);
}

[[nodiscard]] inline TransformOps ops_for(Backend b) {
  if (b == Backend::cpu_scalar)
    return {&forward_chunk, &inverse_chunk, &inverse_chunk_hint_scalar};
  return {&forward_chunk_simd, &inverse_chunk_simd, &inverse_chunk_simd_hint};
}

// --------------------------------------------------------------------------

struct Geometry {
  Dims dims;
  Dims grid;  // bricks per axis

  [[nodiscard]] std::uint64_t brick_count() const {
    return static_cast<std::uint64_t>(grid.x) * grid.y * grid.z;
  }
  void brick_coords(std::uint64_t i, std::uint32_t& bx, std::uint32_t& by,
                    std::uint32_t& bz) const {
    bx = static_cast<std::uint32_t>(i % grid.x);
    by = static_cast<std::uint32_t>((i / grid.x) % grid.y);
    bz = static_cast<std::uint32_t>(i / (static_cast<std::uint64_t>(grid.x) * grid.y));
  }
};

// Typed voxel access. The dtype switch belongs outside the loop, not inside it:
// branching per voxel cost about 5 ns per voxel, which on a 2 MB brick was more
// time than the entropy decode and the inverse transform put together. These
// templates let the caller resolve the type once per brick.
template <typename T>
[[nodiscard]] inline float load_t(const void* base, std::size_t i) {
  return static_cast<float>(static_cast<const T*>(base)[i]);
}

template <typename T>
inline void store_t(void* base, std::size_t i, float v, float lo, float hi) {
  const float c = std::clamp(v, lo, hi);
  if constexpr (std::is_same_v<T, float>) {
    static_cast<T*>(base)[i] = v;
  } else {
    static_cast<T*>(base)[i] = static_cast<T>(std::lrintf(c));
  }
}

// Runs `fn` with the concrete voxel type as a template parameter.
template <typename Fn>
decltype(auto) dispatch_dtype(DType t, Fn&& fn) {
  switch (t) {
    case DType::u8:  return fn(std::type_identity<std::uint8_t>{});
    case DType::s8:  return fn(std::type_identity<std::int8_t>{});
    case DType::u16: return fn(std::type_identity<std::uint16_t>{});
    case DType::s16: return fn(std::type_identity<std::int16_t>{});
    case DType::u32: return fn(std::type_identity<std::uint32_t>{});
    case DType::s32: return fn(std::type_identity<std::int32_t>{});
    case DType::f32: break;
  }
  return fn(std::type_identity<float>{});
}

// Gathers one chunk out of the volume, edge-replicating past the boundary.
// Replication rather than zero-fill: a zero-filled margin is a step edge, and a
// step edge at the volume boundary costs high-frequency coefficients in every
// chunk that touches it.
template <typename T>
void gather_chunk_t(const void* data, Dims dims, float scale, float offset, std::uint32_t ox,
                    std::uint32_t oy, std::uint32_t oz, float* out) {
  for (int z = 0; z < kChunkDim; ++z) {
    const std::uint32_t gz = std::min(oz + static_cast<std::uint32_t>(z), dims.z - 1);
    for (int y = 0; y < kChunkDim; ++y) {
      const std::uint32_t gy = std::min(oy + static_cast<std::uint32_t>(y), dims.y - 1);
      const std::size_t row = (static_cast<std::size_t>(gz) * dims.y + gy) * dims.x;
      float* dst = out + (z * kChunkDim + y) * kChunkDim;
      if (ox + kChunkDim <= dims.x) {
        // Interior: no clamping, so this vectorizes.
        const T* src = static_cast<const T*>(data) + row + ox;
        for (int x = 0; x < kChunkDim; ++x)
          dst[x] = static_cast<float>(src[x]) * scale + offset;
      } else {
        for (int x = 0; x < kChunkDim; ++x) {
          const std::uint32_t gx = std::min(ox + static_cast<std::uint32_t>(x), dims.x - 1);
          dst[x] = load_t<T>(data, row + gx) * scale + offset;
        }
      }
    }
  }
}

void gather_chunk(const void* data, Dims dims, DType t, float scale, float offset,
                  std::uint32_t ox, std::uint32_t oy, std::uint32_t oz, float* out) {
  dispatch_dtype(t, [&](auto tag) {
    using T = typename decltype(tag)::type;
    gather_chunk_t<T>(data, dims, scale, offset, ox, oy, oz, out);
  });
}

template <typename T>
void scatter_chunk_t(void* data, Dims dims, float scale, float offset, std::uint32_t ox,
                     std::uint32_t oy, std::uint32_t oz, const float* in, float lo, float hi) {
  const float inv_scale = 1.0f / scale;
  const int nx = static_cast<int>(std::min<std::uint32_t>(kChunkDim, dims.x - ox));
  for (int z = 0; z < kChunkDim; ++z) {
    const std::uint32_t gz = oz + static_cast<std::uint32_t>(z);
    if (gz >= dims.z) break;
    for (int y = 0; y < kChunkDim; ++y) {
      const std::uint32_t gy = oy + static_cast<std::uint32_t>(y);
      if (gy >= dims.y) break;
      const std::size_t row = (static_cast<std::size_t>(gz) * dims.y + gy) * dims.x + ox;
      const float* src = in + (z * kChunkDim + y) * kChunkDim;
      for (int x = 0; x < nx; ++x)
        store_t<T>(data, row + static_cast<std::size_t>(x), (src[x] - offset) * inv_scale, lo, hi);
    }
  }
}

void scatter_chunk(void* data, Dims dims, DType t, float scale, float offset,
                   std::uint32_t ox, std::uint32_t oy, std::uint32_t oz, const float* in) {
  if (ox >= dims.x || oy >= dims.y || oz >= dims.z) return;
  const float lo = dtype_min(t), hi = dtype_max(t);
  dispatch_dtype(t, [&](auto tag) {
    using T = typename decltype(tag)::type;
    scatter_chunk_t<T>(data, dims, scale, offset, ox, oy, oz, in, lo, hi);
  });
}

// --------------------------------------------------------------------------

void parallel_for(std::uint64_t n, int threads, const std::function<void(std::uint64_t)>& fn) {
  if (threads <= 0) threads = static_cast<int>(std::thread::hardware_concurrency());
  threads = std::max(1, threads);
  if (n <= 1 || threads == 1) {
    for (std::uint64_t i = 0; i < n; ++i) fn(i);
    return;
  }
  std::atomic<std::uint64_t> next{0};
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&] {
      for (;;) {
        const std::uint64_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= n) return;
        fn(i);
      }
    });
  for (auto& th : pool) th.join();
}

// Deblocks the reconstructed volume in place, one axis at a time.
//
// Runs on the decoded output rather than per brick, because a chunk face on a
// brick boundary needs voxels from the neighbouring brick and the whole-volume
// buffer is the one place both sides exist. A brick-at-a-time decoder can call
// this per brick once it carries a two-voxel halo; until then, random-access
// brick decode simply skips deblocking, which is legal by construction.
//
// Boundary planes are 16 voxels apart and the filter reads at most two voxels
// either side, so no two planes on the same axis interact: each axis pass is
// embarrassingly parallel and order-independent within itself. The axes are
// applied in a fixed order, and that order is part of the reconstruction.
void deblock_volume(void* data, Dims dims, DType t, float scale, float offset,
                    const DeblockThresholds& th, int threads) {
  const float inv_scale = 1.0f / scale;
  auto get = [&](std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return load_voxel(data, t, (static_cast<std::size_t>(z) * dims.y + y) * dims.x + x) * scale +
           offset;
  };
  auto set = [&](std::uint32_t x, std::uint32_t y, std::uint32_t z, float w) {
    store_voxel(data, t, (static_cast<std::size_t>(z) * dims.y + y) * dims.x + x,
                (w - offset) * inv_scale);
  };

  const std::uint32_t lim[3] = {dims.x, dims.y, dims.z};
  for (int ax = 0; ax < 3; ++ax) {
    if (lim[ax] < 4) continue;
    // Every chunk boundary along this axis.
    const std::uint32_t planes = (lim[ax] - 1) / kChunkDim;
    if (planes == 0) continue;

    parallel_for(planes, threads, [&](std::uint64_t pi) {
      const std::uint32_t b = static_cast<std::uint32_t>(pi + 1) * kChunkDim;
      if (b < 2 || b + 1 >= lim[ax]) return;

      // Iterate the two axes orthogonal to `ax`.
      const int a1 = (ax + 1) % 3, a2 = (ax + 2) % 3;
      for (std::uint32_t u = 0; u < lim[a1]; ++u)
        for (std::uint32_t v = 0; v < lim[a2]; ++v) {
          std::uint32_t c[3] = {0, 0, 0};
          c[a1] = u;
          c[a2] = v;

          auto at = [&](std::uint32_t k) {
            std::uint32_t q[3] = {c[0], c[1], c[2]};
            q[ax] = k;
            return get(q[0], q[1], q[2]);
          };
          auto put = [&](std::uint32_t k, float w) {
            std::uint32_t q[3] = {c[0], c[1], c[2]};
            q[ax] = k;
            set(q[0], q[1], q[2], w);
          };

          const float p1 = at(b - 2);
          float p0 = at(b - 1);
          float q0 = at(b);
          const float q1 = at(b + 1);
          const float op0 = p0, oq0 = q0;
          filter_face(p1, p0, q0, q1, th);
          if (p0 != op0) put(b - 1, p0);
          if (q0 != oq0) put(b, q0);
        }
    });
  }
}

// Brick payload modes. The first byte of every payload.
inline constexpr std::uint8_t kBrickCoded = 0;  // rANS streams
inline constexpr std::uint8_t kBrickRaw = 1;    // verbatim voxels, see encode_brick

// The part of a brick that actually lies inside the volume. Edge bricks are
// mostly padding, and storing padding would defeat the raw fallback entirely.
[[nodiscard]] Dims brick_valid_extent(Dims dims, std::uint32_t bx, std::uint32_t by,
                                      std::uint32_t bz) {
  return {std::min<std::uint32_t>(kBrickDim, dims.x - bx * kBrickDim),
          std::min<std::uint32_t>(kBrickDim, dims.y - by * kBrickDim),
          std::min<std::uint32_t>(kBrickDim, dims.z - bz * kBrickDim)};
}

// Gathers the in-volume part of a brick as source-dtype bytes. The padding is
// reconstructed by edge replication on the way out, exactly as the coded path
// produces it, so both modes agree on what lies outside the volume.
std::vector<std::uint8_t> gather_brick_raw(const void* data, Dims dims, DType t,
                                           std::uint32_t bx, std::uint32_t by,
                                           std::uint32_t bz) {
  const Dims v = brick_valid_extent(dims, bx, by, bz);
  const std::size_t esz = dtype_size(t);
  std::vector<std::uint8_t> out(v.voxels() * esz);
  for (std::uint32_t z = 0; z < v.z; ++z)
    for (std::uint32_t y = 0; y < v.y; ++y) {
      const std::size_t src =
          ((static_cast<std::size_t>(bz * kBrickDim + z) * dims.y + (by * kBrickDim + y)) *
               dims.x +
           bx * kBrickDim);
      const std::size_t dst = (static_cast<std::size_t>(z) * v.y + y) * v.x;
      std::memcpy(out.data() + dst * esz,
                  static_cast<const std::uint8_t*>(data) + src * esz, v.x * esz);
    }
  return out;
}

// Encodes one brick and returns its payload bytes.
// What the correction layer has to achieve for one archive.
struct BoundSpec {
  float max_abs = 0.0f;   // absolute bound, input units; 0 = not requested
  float max_rel = 0.0f;   // bound relative to each chunk's own value range
  float percentile_tau = 0.0f;  // threshold derived from a global error histogram
  float corr_step = 1.0f;

  [[nodiscard]] bool active() const {
    return max_abs > 0.0f || max_rel > 0.0f || percentile_tau > 0.0f;
  }
};

// Reconstructs a chunk the way the decoder will, so the encoder can measure the
// error it is about to ship. Everything about the bound guarantee rests on this
// matching decode exactly.
void reconstruct_chunk(const std::int32_t* levels, const QuantMatrix& qm, float* coeffs,
                       float* out_voxels, TransformOps ops) {
  for (int i = 0; i < kChunkVox; ++i)
    coeffs[i] = dequantize(levels[i], qm.q[static_cast<std::size_t>(i)]);
  ops.inv(coeffs, out_voxels);
}

// Per-voxel error of a reconstructed chunk, in input units, together with the
// corrections needed to bring every voxel inside `tau`.
//
// Only voxels inside the volume are considered: padding beyond the boundary is
// discarded on decode, and correcting it would spend bits on nothing.
int compute_chunk_corrections(const float* orig_work, const float* dec_work, DType t,
                              float scale, float offset, float tau, float corr_step,
                              Dims dims, std::uint32_t ox, std::uint32_t oy, std::uint32_t oz,
                              std::int32_t* deltas) {
  const float inv_scale = 1.0f / scale;
  const float lo = dtype_min(t), hi = dtype_max(t);
  int n = 0;
  for (int z = 0; z < kChunkDim; ++z) {
    if (oz + static_cast<std::uint32_t>(z) >= dims.z) break;
    for (int y = 0; y < kChunkDim; ++y) {
      if (oy + static_cast<std::uint32_t>(y) >= dims.y) break;
      for (int x = 0; x < kChunkDim; ++x) {
        if (ox + static_cast<std::uint32_t>(x) >= dims.x) break;
        const int i = (z * kChunkDim + y) * kChunkDim + x;

        const float orig_raw = (orig_work[i] - offset) * inv_scale;
        // Exactly what store_voxel will write.
        const float dec_clamped = std::clamp((dec_work[i] - offset) * inv_scale, lo, hi);
        const float dec_raw = (t == DType::f32)
                                  ? dec_clamped
                                  : static_cast<float>(std::lrintf(dec_clamped));

        const float err = orig_raw - dec_raw;
        if (std::fabs(err) <= tau) continue;

        const int d = static_cast<int>(std::lrintf(err / corr_step));
        if (d == 0) continue;
        deltas[i] = d;
        ++n;
      }
    }
  }
  return n;
}

std::vector<std::uint8_t> encode_brick(const void* data, Dims dims, DType t, float scale,
                                       float offset, const QuantMatrix& qm, float deadzone,
                                       const ModelSet& ms, std::uint32_t P,
                                       std::uint32_t bx, std::uint32_t by, std::uint32_t bz,
                                       const BoundSpec& bounds,
                                       std::vector<std::uint64_t>* err_histogram,
                                       TransformOps ops, bool rdo, bool per_brick_tables,
                                       CountCollector* counts = nullptr) {
  std::vector<std::vector<Sym>> streams(P);
  // Each stream carries 512/P chunks, and a dense chunk emits on the order of a
  // few thousand symbols. Reserving up front turns millions of push_backs per
  // brick into plain stores.
  for (auto& st : streams) st.reserve((kChunksPerBrick / P + 1) * 1024);
  std::vector<std::vector<Sym>> corr_streams(bounds.active() ? P : 0);
  std::vector<float> voxels(kChunkVox), coeffs(kChunkVox), recon(kChunkVox);
  std::vector<std::int32_t> levels(kChunkVox), deltas(kChunkVox);
  bool any_corrections = false;

  for (int ci = 0; ci < kChunksPerBrick; ++ci) {
    const int cx = ci % kBrickChunks;
    const int cy = (ci / kBrickChunks) % kBrickChunks;
    const int cz = ci / (kBrickChunks * kBrickChunks);
    const std::uint32_t ox = bx * kBrickDim + static_cast<std::uint32_t>(cx * kChunkDim);
    const std::uint32_t oy = by * kBrickDim + static_cast<std::uint32_t>(cy * kChunkDim);
    const std::uint32_t oz = bz * kBrickDim + static_cast<std::uint32_t>(cz * kChunkDim);

    gather_chunk(data, dims, t, scale, offset, ox, oy, oz, voxels.data());
    ops.fwd(voxels.data(), coeffs.data());
    for (int i = 0; i < kChunkVox; ++i)
      levels[static_cast<std::size_t>(i)] =
          quantize(coeffs[static_cast<std::size_t>(i)], qm.rq[static_cast<std::size_t>(i)],
                   deadzone);
    if (rdo) rdo_optimize(coeffs.data(), qm, ms, levels.data());

    encode_chunk_symbols(levels.data(), 0,
                         streams[static_cast<std::size_t>(ci) % P]);

    if (!bounds.active() && err_histogram == nullptr) continue;

    reconstruct_chunk(levels.data(), qm, coeffs.data(), recon.data(), ops);

    // Measuring pass: accumulate the error distribution so a percentile bound
    // can be turned into an absolute threshold.
    if (err_histogram != nullptr) {
      const float inv_scale = 1.0f / scale;
      const float lo = dtype_min(t), hi = dtype_max(t);
      for (int z = 0; z < kChunkDim; ++z) {
        if (oz + static_cast<std::uint32_t>(z) >= dims.z) break;
        for (int y = 0; y < kChunkDim; ++y) {
          if (oy + static_cast<std::uint32_t>(y) >= dims.y) break;
          for (int x = 0; x < kChunkDim; ++x) {
            if (ox + static_cast<std::uint32_t>(x) >= dims.x) break;
            const int i = (z * kChunkDim + y) * kChunkDim + x;
            const float orig_raw = (voxels[i] - offset) * inv_scale;
            const float dc = std::clamp((recon[i] - offset) * inv_scale, lo, hi);
            const float dec_raw =
                (t == DType::f32) ? dc : static_cast<float>(std::lrintf(dc));
            const std::size_t bin = std::min<std::size_t>(
                err_histogram->size() - 1,
                static_cast<std::size_t>(std::fabs(orig_raw - dec_raw) / bounds.corr_step));
            ++(*err_histogram)[bin];
          }
        }
      }
    }

    if (!bounds.active()) continue;

    float tau = std::numeric_limits<float>::max();
    if (bounds.max_abs > 0.0f) tau = std::min(tau, bounds.max_abs);
    if (bounds.percentile_tau > 0.0f) tau = std::min(tau, bounds.percentile_tau);
    if (bounds.max_rel > 0.0f) {
      // Relative to this chunk's own value range, as 3ddct does: a bound that
      // scales with local contrast is what "relative error" means for data
      // whose dynamic range varies enormously between air and material.
      float vlo = voxels[0], vhi = voxels[0];
      for (int i = 1; i < kChunkVox; ++i) {
        vlo = std::min(vlo, voxels[i]);
        vhi = std::max(vhi, voxels[i]);
      }
      tau = std::min(tau, bounds.max_rel * (vhi - vlo) / scale);
    }

    std::fill(deltas.begin(), deltas.end(), 0);
    const int nc = compute_chunk_corrections(voxels.data(), recon.data(), t, scale, offset, tau,
                                             bounds.corr_step, dims, ox, oy, oz, deltas.data());
    if (nc > 0) any_corrections = true;
    encode_correction_symbols(deltas.data(), corr_streams[static_cast<std::size_t>(ci) % P]);
  }

  // Training only needs the symbols, not the bytes they compress to.
  if (counts != nullptr) {
    for (const auto& st : streams)
      for (const Sym& sy : st) counts->add(sy);
    for (const auto& st : corr_streams)
      for (const Sym& sy : st) counts->add(sy);
    return {};
  }

  // Per-brick tables. The brick's own symbol histogram is clustered into a few
  // tables (brick_tables.hpp); if they do not pay for their own bytes, the brick
  // keeps the global ones.
  BrickTables bt;
  const ModelSet* use = &ms;
  if (per_brick_tables) {
    std::vector<std::vector<std::uint64_t>> hist(ModelIndex::total);
    for (std::uint16_t i = 0; i < ModelIndex::total; ++i)
      hist[i].assign(ms[i].nsym(), 0);
    for (const auto& st : streams)
      for (const Sym& sy : st)
        if (sy.value < hist[sy.model].size()) ++hist[sy.model][sy.value];
    bt = build_brick_tables(ms, hist);
    if (bt.used) use = &bt.models;
  }

  std::vector<std::vector<std::uint8_t>> blobs(P);
  for (std::uint32_t s = 0; s < P; ++s) {
    RansEncoder enc(kRansStates);
    blobs[s] = enc.encode(streams[s], use->models());
  }

  // Corrections go in their own streams so a decoder that only wants the fast
  // lossy reconstruction skips the bytes without parsing them.
  std::vector<std::vector<std::uint8_t>> corr_blobs;
  const bool emit_corr = bounds.active() && any_corrections;
  if (emit_corr) {
    corr_blobs.resize(P);
    for (std::uint32_t s = 0; s < P; ++s) {
      RansEncoder enc(kRansStates);
      corr_blobs[s] = enc.encode(corr_streams[s], ms.models());
    }
  }

  std::vector<std::uint8_t> payload;
  std::size_t total = 5 + bt.blob.size() + 4 * static_cast<std::size_t>(P);
  for (const auto& b : blobs) total += b.size();
  for (const auto& b : corr_blobs) total += b.size() + 4;
  payload.reserve(total);
  payload.push_back(kBrickCoded);
  payload.push_back(bt.used ? 1 : 0);
  if (bt.used) {
    put_u32(payload, static_cast<std::uint32_t>(bt.blob.size()));
    payload.insert(payload.end(), bt.blob.begin(), bt.blob.end());
  }
  for (const auto& b : blobs) put_u32(payload, static_cast<std::uint32_t>(b.size()));
  payload.push_back(emit_corr ? 1 : 0);
  for (const auto& b : corr_blobs) put_u32(payload, static_cast<std::uint32_t>(b.size()));
  for (const auto& b : blobs) payload.insert(payload.end(), b.begin(), b.end());
  for (const auto& b : corr_blobs) payload.insert(payload.end(), b.begin(), b.end());

  // Incompressible data must never be made bigger. Noise-dominated regions are
  // real -- air in a CT scan is mostly detector noise -- and without this a
  // "compressed" archive of one can be several times the size of the input.
  //
  // For 8- and 16-bit dtypes the working-domain mapping is an exact integer
  // shift, so a raw brick also decodes losslessly: where this triggers it is
  // strictly better on both size and error. For f32/u32/s32 the mapping is
  // itself lossy, so a raw brick is merely much more accurate, not exact.
  //
  // Suppressed under an error bound for the wide dtypes, where storing the
  // source bytes still costs the working-domain mapping's precision on the way
  // out and so cannot be relied on to satisfy a tight bound. For 8- and 16-bit
  // types the mapping is exact, so the fallback is always safe.
  const bool raw_is_exact = (t == DType::u8 || t == DType::s8 || t == DType::u16 ||
                             t == DType::s16);
  std::vector<std::uint8_t> raw =
      (bounds.active() && !raw_is_exact) ? std::vector<std::uint8_t>{}
                                         : gather_brick_raw(data, dims, t, bx, by, bz);
  if (!raw.empty() && raw.size() + 1 < payload.size()) {
    std::vector<std::uint8_t> alt;
    alt.reserve(raw.size() + 1);
    alt.push_back(kBrickRaw);
    alt.insert(alt.end(), raw.begin(), raw.end());
    return alt;
  }
  return payload;
}

// Where a decoded chunk should go. When set, chunks are written straight into
// the output volume as they are produced, which avoids staging the whole brick
// in a float working buffer and then copying it back out -- two passes over four
// times the output size in bytes, for nothing.
struct DirectScatter {
  void* out;
  Dims dims;
  DType dtype;
  float scale, offset;
  std::uint32_t bx, by, bz;
};

// Decodes one brick's payload. Writes into `out_work` (a kBrickDim^3 float
// buffer) unless `direct` is set, in which case it scatters into the volume.
[[nodiscard]] Status decode_brick_payload(std::span<const std::uint8_t> payload,
                                          const QuantMatrix& qm, const ModelSet& ms,
                                          std::uint32_t P, DType t, float scale, float offset,
                                          Dims valid, TransformOps ops, float* out_work,
                                          std::vector<std::pair<std::uint32_t, std::int32_t>>*
                                              out_corrections,
                                          const DirectScatter* direct) {
  if (payload.empty()) return Status::truncated;
  const std::uint8_t mode = payload[0];

  if (mode == kBrickRaw) {
    if (valid.empty()) return Status::corrupt_bitstream;
    if (payload.size() != 1 + valid.voxels() * dtype_size(t)) return Status::corrupt_bitstream;
    const void* src = payload.data() + 1;
    for (int z = 0; z < kBrickDim; ++z) {
      const std::uint32_t sz = std::min<std::uint32_t>(static_cast<std::uint32_t>(z), valid.z - 1);
      for (int y = 0; y < kBrickDim; ++y) {
        const std::uint32_t sy =
            std::min<std::uint32_t>(static_cast<std::uint32_t>(y), valid.y - 1);
        for (int x = 0; x < kBrickDim; ++x) {
          const std::uint32_t sx =
              std::min<std::uint32_t>(static_cast<std::uint32_t>(x), valid.x - 1);
          const std::size_t si = (static_cast<std::size_t>(sz) * valid.y + sy) * valid.x + sx;
          const std::size_t di =
              (static_cast<std::size_t>(z) * kBrickDim + static_cast<std::size_t>(y)) *
                  kBrickDim + static_cast<std::size_t>(x);
          out_work[di] = load_voxel(src, t, si) * scale + offset;
        }
      }
    }
    if (direct != nullptr) {
      for (int cz = 0; cz < kBrickChunks; ++cz)
        for (int cy = 0; cy < kBrickChunks; ++cy)
          for (int cx = 0; cx < kBrickChunks; ++cx) {
            float chunk[kChunkVox];
            for (int z = 0; z < kChunkDim; ++z)
              for (int y = 0; y < kChunkDim; ++y)
                std::memcpy(chunk + (z * kChunkDim + y) * kChunkDim,
                            out_work + ((static_cast<std::size_t>(cz * kChunkDim + z) * kBrickDim +
                                         static_cast<std::size_t>(cy * kChunkDim + y)) *
                                            kBrickDim +
                                        static_cast<std::size_t>(cx * kChunkDim)),
                            kChunkDim * sizeof(float));
            scatter_chunk(direct->out, direct->dims, direct->dtype, direct->scale,
                          direct->offset,
                          direct->bx * kBrickDim + static_cast<std::uint32_t>(cx * kChunkDim),
                          direct->by * kBrickDim + static_cast<std::uint32_t>(cy * kChunkDim),
                          direct->bz * kBrickDim + static_cast<std::uint32_t>(cz * kChunkDim),
                          chunk);
          }
    }
    return Status::ok;
  }
  if (mode != kBrickCoded) return Status::corrupt_bitstream;

  if (payload.size() < 3) return Status::truncated;
  std::size_t pos = 1;

  // Per-brick tables, if this brick chose to carry them.
  ModelSet brick_ms;
  const ModelSet* use = &ms;
  const std::uint8_t table_flag = payload[pos++];
  if (table_flag > 1) return Status::corrupt_bitstream;
  if (table_flag == 1) {
    if (pos + 4 > payload.size()) return Status::truncated;
    const std::uint32_t blob_size = get_u32(payload, pos);
    pos += 4;
    if (blob_size > payload.size() - pos) return Status::corrupt_bitstream;
    if (!read_brick_tables(payload.subspan(pos, blob_size), ms, brick_ms))
      return Status::corrupt_bitstream;
    pos += blob_size;
    use = &brick_ms;
  }

  if (payload.size() < pos + 4u * P + 1) return Status::truncated;
  std::vector<std::uint32_t> sizes(P);
  std::uint64_t sum = 0;
  for (std::uint32_t s = 0; s < P; ++s) {
    sizes[s] = get_u32(payload, pos);
    pos += 4;
    sum += sizes[s];
  }

  const std::uint8_t has_corr = payload[pos++];
  if (has_corr > 1) return Status::corrupt_bitstream;
  std::vector<std::uint32_t> corr_sizes;
  if (has_corr) {
    if (pos + 4u * P > payload.size()) return Status::truncated;
    corr_sizes.resize(P);
    for (std::uint32_t s = 0; s < P; ++s) {
      corr_sizes[s] = get_u32(payload, pos);
      pos += 4;
      sum += corr_sizes[s];
    }
  }
  if (pos + sum != payload.size()) return Status::corrupt_bitstream;

  // One decoder per stream; each walks its own chunks in increasing index order.
  std::vector<RansDecoder> decoders(P);
  for (std::uint32_t s = 0; s < P; ++s) {
    if (!decoders[s].init(payload.subspan(pos, sizes[s]), kRansStates))
      return Status::corrupt_bitstream;
    pos += sizes[s];
  }

  // The correction streams are only parsed when the caller wants them. Skipping
  // them is exactly the "additive layer" property the format promises.
  const bool want_corr = has_corr && out_corrections != nullptr;
  std::vector<RansDecoder> corr_decoders;
  if (want_corr) {
    corr_decoders.resize(P);
    for (std::uint32_t s = 0; s < P; ++s) {
      if (!corr_decoders[s].init(payload.subspan(pos, corr_sizes[s]), kRansStates))
        return Status::corrupt_bitstream;
      pos += corr_sizes[s];
    }
  }

  std::vector<std::int32_t> levels(kChunkVox), deltas(kChunkVox);
  std::vector<float> coeffs(kChunkVox, 0.0f), voxels(kChunkVox);
  std::vector<int> touched;
  touched.reserve(256);
  const std::int32_t max_level = max_plausible_level(qm);

  for (int ci = 0; ci < kChunksPerBrick; ++ci) {
    std::uint8_t exponent = 0;
    if (!decode_chunk_coeffs(decoders[static_cast<std::size_t>(ci) % P], *use, coeffs.data(),
                             qm.q.data(), exponent, max_level, touched))
      return Status::corrupt_bitstream;
    const int nonzero = static_cast<int>(touched.size());

    if (want_corr) {
      if (!decode_correction_symbols(corr_decoders[static_cast<std::size_t>(ci) % P], *use,
                                     deltas.data(), max_level))
        return Status::corrupt_bitstream;
      const int cx0 = ci % kBrickChunks;
      const int cy0 = (ci / kBrickChunks) % kBrickChunks;
      const int cz0 = ci / (kBrickChunks * kBrickChunks);
      for (int i = 0; i < kChunkVox; ++i) {
        if (deltas[i] == 0) continue;
        const int lx = i % kChunkDim, ly = (i / kChunkDim) % kChunkDim, lz = i / (kChunkDim * kChunkDim);
        const std::uint32_t bi = static_cast<std::uint32_t>(
            ((cz0 * kChunkDim + lz) * kBrickDim + (cy0 * kChunkDim + ly)) * kBrickDim +
            (cx0 * kChunkDim + lx));
        out_corrections->emplace_back(bi, deltas[i]);
      }
    }

    ops.inv_hint(coeffs.data(), voxels.data(), nonzero);
    // Restore the all-zero precondition by clearing only what was written.
    for (int i : touched) coeffs[static_cast<std::size_t>(i)] = 0.0f;

    const int cx = ci % kBrickChunks;
    const int cy = (ci / kBrickChunks) % kBrickChunks;
    const int cz = ci / (kBrickChunks * kBrickChunks);

    if (direct != nullptr) {
      scatter_chunk(direct->out, direct->dims, direct->dtype, direct->scale, direct->offset,
                    direct->bx * kBrickDim + static_cast<std::uint32_t>(cx * kChunkDim),
                    direct->by * kBrickDim + static_cast<std::uint32_t>(cy * kChunkDim),
                    direct->bz * kBrickDim + static_cast<std::uint32_t>(cz * kChunkDim),
                    voxels.data());
      continue;
    }

    for (int z = 0; z < kChunkDim; ++z)
      for (int y = 0; y < kChunkDim; ++y) {
        const std::size_t dst =
            ((static_cast<std::size_t>(cz * kChunkDim + z) * kBrickDim +
              static_cast<std::size_t>(cy * kChunkDim + y)) *
                 kBrickDim +
             static_cast<std::size_t>(cx * kChunkDim));
        std::memcpy(out_work + dst,
                    voxels.data() + (z * kChunkDim + y) * kChunkDim, kChunkDim * sizeof(float));
      }
  }
  return Status::ok;
}

}  // namespace

namespace detail {

void collect_counts(const void* data, Dims dims, DType dtype, const EncodeOptions& opts,
                    CountCollector& out) {
  if (data == nullptr || dims.empty()) return;
  const Geometry geo{dims, brick_grid(dims)};
  QuantParams qp = QuantParams::for_profile(opts.profile, opts.quality);
  qp.deadzone = deadzone_for_effort(opts.effort);
  const QuantMatrix qm = QuantMatrix::build(qp);
  const ModelSet ms = ModelSet::defaults();
  float scale = 1.0f, offset = 0.0f;
  derive_mapping(data, dims, dtype, scale, offset);

  // Each brick counts into its own collector and the results are merged, so the
  // trained tables do not depend on thread scheduling.
  std::vector<CountCollector> per_brick(static_cast<std::size_t>(geo.brick_count()));
  parallel_for(geo.brick_count(), opts.threads, [&](std::uint64_t i) {
    std::uint32_t bx, by, bz;
    geo.brick_coords(i, bx, by, bz);
    BoundSpec none;
    (void)encode_brick(data, dims, dtype, scale, offset, qm, qp.deadzone, ms,
                       opts.streams_per_brick, bx, by, bz, none, nullptr,
                       ops_for(Backend::automatic), opts.effort == Effort::high, false,
                       &per_brick[static_cast<std::size_t>(i)]);
  });
  for (const CountCollector& c : per_brick) out.merge(c);
}

BitBreakdown measure_bits(const void* data, Dims dims, DType dtype,
                          const EncodeOptions& opts) {
  BitBreakdown out;
  if (data == nullptr || dims.empty()) return out;
  const Geometry geo{dims, brick_grid(dims)};
  QuantParams qp = QuantParams::for_profile(opts.profile, opts.quality);
  qp.deadzone = deadzone_for_effort(opts.effort);
  const QuantMatrix qm = QuantMatrix::build(qp);
  const ModelSet ms = ModelSet::defaults(2);
  float scale = 1.0f, offset = 0.0f;
  derive_mapping(data, dims, dtype, scale, offset);
  out.voxels = dims.voxels();

  // Symbols are re-derived per brick and classified by model, which is why the
  // accounting cannot drift from what the encoder actually emits: it is the
  // same symbol stream, costed with the same tables.
  std::vector<BitBreakdown> per_brick(static_cast<std::size_t>(geo.brick_count()));
  parallel_for(geo.brick_count(), opts.threads, [&](std::uint64_t i) {
    std::uint32_t bx, by, bz;
    geo.brick_coords(i, bx, by, bz);
    std::vector<float> voxels(kChunkVox), coeffs(kChunkVox);
    std::vector<std::int32_t> levels(kChunkVox);
    std::vector<Sym> syms;
    BitBreakdown& b = per_brick[static_cast<std::size_t>(i)];

    for (int ci = 0; ci < kChunksPerBrick; ++ci) {
      const int cx = ci % kBrickChunks;
      const int cy = (ci / kBrickChunks) % kBrickChunks;
      const int cz = ci / (kBrickChunks * kBrickChunks);
      gather_chunk(data, dims, dtype, scale, offset,
                   bx * kBrickDim + static_cast<std::uint32_t>(cx * kChunkDim),
                   by * kBrickDim + static_cast<std::uint32_t>(cy * kChunkDim),
                   bz * kBrickDim + static_cast<std::uint32_t>(cz * kChunkDim), voxels.data());
      forward_chunk(voxels.data(), coeffs.data());
      for (int k = 0; k < kChunkVox; ++k)
        levels[static_cast<std::size_t>(k)] = quantize(
            coeffs[static_cast<std::size_t>(k)], qm.rq[static_cast<std::size_t>(k)],
            qp.deadzone);
      if (opts.effort == Effort::high) rdo_optimize(coeffs.data(), qm, ms, levels.data());

      syms.clear();
      encode_chunk_symbols(levels.data(), 0, syms);

      // A bypass bit that follows a magnitude symbol is its mantissa; one that
      // follows anything else is a sign. Tracking the previous model is enough
      // to tell them apart without changing the coder.
      bool prev_was_len = false;
      for (const Sym& sy : syms) {
        const double bits = ms[sy.model].bit_cost(sy.value);
        if (sy.model == ModelIndex::bypass) {
          if (prev_was_len) b.mantissas += 1.0;
          else b.signs += 1.0;
          continue;
        }
        prev_was_len = false;
        if (sy.model == ModelIndex::chunk_nonzero) b.chunk_flags += bits;
        else if (sy.model == ModelIndex::exponent) b.exponents += bits;
        else if (sy.model >= ModelIndex::l1_base &&
                 sy.model < ModelIndex::l1_base + ModelIndex::l1_count) b.l1_masks += bits;
        else if (sy.model >= ModelIndex::l2_base &&
                 sy.model < ModelIndex::l2_base + ModelIndex::l2_count) b.l2_masks += bits;
        else if (sy.model >= ModelIndex::level_base &&
                 sy.model < ModelIndex::level_base + ModelIndex::level_count) b.levels += bits;
        else if (sy.model == ModelIndex::dc_len) { b.dc += bits; prev_was_len = true; }
        else if (sy.model == ModelIndex::esc_len) { b.escapes += bits; prev_was_len = true; }
      }
    }
  });

  for (const BitBreakdown& b : per_brick) {
    out.chunk_flags += b.chunk_flags;
    out.exponents += b.exponents;
    out.l1_masks += b.l1_masks;
    out.l2_masks += b.l2_masks;
    out.levels += b.levels;
    out.dc += b.dc;
    out.escapes += b.escapes;
    out.signs += b.signs;
    out.mantissas += b.mantissas;
  }
  return out;
}

}  // namespace detail

// --------------------------------------------------------------------------
// Public entry points
// --------------------------------------------------------------------------

Status inspect(std::span<const std::uint8_t> archive, VolumeInfo& out) {
  detail::FileHeader h;
  const Status s = detail::read_header(archive, h);
  if (s != Status::ok) return s;
  out.dims = h.dims;
  out.dtype = h.dtype;
  out.profile = h.profile;
  out.version = h.version;
  out.brick_count = h.brick_count;
  out.streams_per_brick = h.streams_per_brick;
  out.quant = {h.q_base, h.q_a, h.q_b, h.deadzone};
  return Status::ok;
}

Status encode(const void* data, Dims dims, DType dtype, const EncodeOptions& opts,
              std::vector<std::uint8_t>& out, Backend backend) {
  if (data == nullptr || dims.empty()) return Status::invalid_argument;
  if (opts.streams_per_brick == 0 || opts.streams_per_brick > 64)
    return Status::invalid_argument;
  if (backend != Backend::automatic && backend != Backend::cpu_scalar)
    if (!backend_available(backend)) return Status::backend_unavailable;

  Geometry geo{dims, brick_grid(dims)};
  const std::uint32_t P = opts.streams_per_brick;

  detail::FileHeader h;
  h.dims = dims;
  h.dtype = dtype;
  h.profile = opts.profile;
  h.streams_per_brick = static_cast<std::uint8_t>(P);
  h.brick_count = geo.brick_count();

  QuantParams qp = QuantParams::for_profile(opts.profile, opts.quality);
  qp.deadzone = deadzone_for_effort(opts.effort);
  if (opts.deadzone_override >= 0.0f) qp.deadzone = opts.deadzone_override;
  if (opts.qshape_override > 0.0f) qp.b = opts.qshape_override;
  if (opts.qamp_override >= 0.0f) qp.a = opts.qamp_override;
  h.q_base = qp.base;
  h.q_a = qp.a;
  h.q_b = qp.b;
  h.deadzone = qp.deadzone;
  derive_mapping(data, dims, dtype, h.data_scale, h.data_offset);

  const detail::QuantMatrix qm = detail::QuantMatrix::build(qp);
  const detail::ModelSet ms = detail::ModelSet::defaults(h.table_version);
  const TransformOps ops = ops_for(backend);

  // Correction step: one input unit for integer dtypes, which makes a
  // correction exact. For f32 there is no natural unit, so the step is tied to
  // the requested bound -- half a step of residual keeps the bound satisfied.
  BoundSpec bounds;
  bounds.max_abs = opts.bounds.max_abs;
  bounds.max_rel = opts.bounds.max_rel;
  if (dtype == DType::f32) {
    float t = opts.bounds.max_abs;
    if (t <= 0.0f) t = opts.bounds.p99;
    if (t <= 0.0f) t = opts.bounds.p999;
    bounds.corr_step = (t > 0.0f) ? t : 1.0f;
  } else {
    bounds.corr_step = 1.0f;
  }
  h.corr_step = bounds.corr_step;

  // A percentile bound is a statement about the whole volume, so it cannot be
  // evaluated chunk by chunk. Measure the error distribution first, convert the
  // percentile into an absolute threshold, then encode for real. This costs a
  // second transform pass, and only when a percentile bound is actually asked
  // for.
  if (opts.bounds.p99 > 0.0f || opts.bounds.p999 > 0.0f) {
    constexpr std::size_t kBins = 1 << 16;
    std::vector<std::vector<std::uint64_t>> per_brick(
        static_cast<std::size_t>(h.brick_count));
    BoundSpec measure;  // no bound active: measuring only
    measure.corr_step = bounds.corr_step;
    parallel_for(h.brick_count, opts.threads, [&](std::uint64_t i) {
      std::uint32_t bx, by, bz;
      geo.brick_coords(i, bx, by, bz);
      auto& hist = per_brick[static_cast<std::size_t>(i)];
      hist.assign(kBins, 0);
      (void)encode_brick(data, dims, dtype, h.data_scale, h.data_offset, qm, qp.deadzone, ms, P,
                         bx, by, bz, measure, &hist, ops, opts.effort == Effort::high,
                         opts.per_brick_tables);
    });

    std::vector<std::uint64_t> hist(kBins, 0);
    std::uint64_t total = 0;
    for (const auto& b : per_brick)
      for (std::size_t k = 0; k < kBins; ++k) {
        hist[k] += b[k];
        total += b[k];
      }

    // Turns "the q-th percentile of the error must not exceed tau_bound" into
    // the absolute threshold T above which voxels get corrected.
    //
    // Correcting every voxel above T leaves the ones in (tau_bound, T] still in
    // violation, so T is chosen to make exactly that leftover fit inside the
    // (1-q) budget:
    //
    //     F(T) <= (1 - q) + F(tau_bound)
    //
    // which is what makes a percentile bound cheaper than a hard one. Requiring
    // p99 <= 4 when 95% of voxels are already within 4 corrects the worst 4%,
    // not the worst 5% -- and if 99% are already within 4 it corrects nothing at
    // all. Taking T = tau_bound instead, as a hard bound does, is always
    // sufficient but never cheaper, which is the whole point of the mode.
    // Derived straight from the constraint rather than from a quantile, because
    // the two differ by a bin and the bound is the thing being promised.
    //
    // The reported p_q is the ceil(q*n)-th smallest error, so p_q <= tau_bound
    // holds exactly when at least ceil(q*n) voxels are within tau_bound -- that
    // is, when at most `allowed = n - ceil(q*n)` voxels exceed it. Correcting
    // everything above T leaves the voxels in (tau_bound, T] in violation, so
    // pick the largest T whose leftover still fits the budget.
    auto threshold_for = [&](float q, float tau_bound) -> float {
      if (tau_bound <= 0.0f || total == 0) return std::numeric_limits<float>::max();

      const std::size_t bin_tau = std::min<std::size_t>(
          kBins - 1, static_cast<std::size_t>(tau_bound / bounds.corr_step));
      std::uint64_t c_tau = 0;
      for (std::size_t k = 0; k <= bin_tau; ++k) c_tau += hist[k];

      const std::uint64_t need = static_cast<std::uint64_t>(
          std::ceil(static_cast<double>(q) * static_cast<double>(total)));
      if (c_tau >= need) return std::numeric_limits<float>::max();  // already satisfied
      const std::uint64_t allowed = total - need;

      std::uint64_t running = c_tau;
      std::size_t best = bin_tau;
      for (std::size_t k = bin_tau + 1; k < kBins; ++k) {
        running += hist[k];
        if (running - c_tau > allowed) break;
        best = k;
      }
      return static_cast<float>(best) * bounds.corr_step;
    };

    float tau = std::numeric_limits<float>::max();
    tau = std::min(tau, threshold_for(0.99f, opts.bounds.p99));
    tau = std::min(tau, threshold_for(0.999f, opts.bounds.p999));
    if (tau < std::numeric_limits<float>::max()) bounds.percentile_tau = tau;
  }

  if (bounds.active()) h.flags |= detail::kFlagCorrections;

  // Encode every brick independently, then lay them out. Bricks are compressed
  // in parallel and assembled in order, so output bytes never depend on thread
  // scheduling.
  std::vector<std::vector<std::uint8_t>> payloads(static_cast<std::size_t>(h.brick_count));

  bool encoded = false;
#ifdef GPUDCT_HAVE_CUDA
  // The GPU encoder implements the plain coded path only. It declines anything
  // it cannot reproduce exactly -- error bounds, and chunks too dense for its
  // fixed device scratch -- and we fall through to the CPU, which is always
  // correct. Raw-brick substitution is applied below either way.
  if (backend == Backend::cuda && cuda::available() && !bounds.active()) {
    if (cuda::encode_bricks(data, dims, dtype, h.data_scale, h.data_offset, qm, qp.deadzone,
                            ms, P, opts.effort == Effort::high, opts.per_brick_tables,
                            payloads) == Status::ok)
      encoded = true;
  }
#endif

  if (!encoded) {
    parallel_for(h.brick_count, opts.threads, [&](std::uint64_t i) {
      std::uint32_t bx, by, bz;
      geo.brick_coords(i, bx, by, bz);
      payloads[static_cast<std::size_t>(i)] =
          encode_brick(data, dims, dtype, h.data_scale, h.data_offset, qm, qp.deadzone, ms, P,
                       bx, by, bz, bounds, nullptr, ops, opts.effort == Effort::high,
                       opts.per_brick_tables);
    });
  } else {
    // The device path does not evaluate the raw-brick fallback, so apply it
    // here: an archive must never be larger than its input, whichever backend
    // produced it.
    const bool raw_is_exact = (dtype == DType::u8 || dtype == DType::s8 ||
                               dtype == DType::u16 || dtype == DType::s16);
    if (raw_is_exact) {
      parallel_for(h.brick_count, opts.threads, [&](std::uint64_t i) {
        std::uint32_t bx, by, bz;
        geo.brick_coords(i, bx, by, bz);
        std::vector<std::uint8_t> raw = gather_brick_raw(data, dims, dtype, bx, by, bz);
        auto& p = payloads[static_cast<std::size_t>(i)];
        if (raw.size() + 1 < p.size()) {
          std::vector<std::uint8_t> alt;
          alt.reserve(raw.size() + 1);
          alt.push_back(kBrickRaw);
          alt.insert(alt.end(), raw.begin(), raw.end());
          p = std::move(alt);
        }
      });
    }
  }

  h.index_offset = detail::FileHeader::kSize;
  const std::uint64_t payload_start =
      h.index_offset + h.brick_count * detail::BrickEntry::kSize;

  out.clear();
  out.reserve(static_cast<std::size_t>(payload_start) + h.brick_count * 256);
  detail::write_header(h, out);

  std::uint64_t cursor = payload_start;
  for (std::uint64_t i = 0; i < h.brick_count; ++i) {
    detail::put_u64(out, cursor);
    detail::put_u32(out, static_cast<std::uint32_t>(payloads[static_cast<std::size_t>(i)].size()));
    detail::put_u32(out, 0);
    cursor += payloads[static_cast<std::size_t>(i)].size();
  }
  for (auto& p : payloads) {
    out.insert(out.end(), p.begin(), p.end());
    p.clear();
    p.shrink_to_fit();
  }
  return Status::ok;
}

Status decode(std::span<const std::uint8_t> archive, const DecodeOptions& opts,
              std::vector<std::uint8_t>& out, VolumeInfo& info, Backend backend) {
  detail::FileHeader probe;
  if (const Status s = detail::read_header(archive, probe); s != Status::ok) return s;
  out.resize(probe.dims.voxels() * dtype_size(probe.dtype));
  const Status s = decode_into(archive, opts, out, info, backend);
  if (s != Status::ok) out.clear();
  return s;
}

Status decode_into(std::span<const std::uint8_t> archive, const DecodeOptions& opts,
                   std::span<std::uint8_t> out, VolumeInfo& info, Backend backend) {
  detail::FileHeader h;
  const Status s = detail::read_header(archive, h);
  if (s != Status::ok) return s;
  if (backend != Backend::automatic && backend != Backend::cpu_scalar)
    if (!backend_available(backend)) return Status::backend_unavailable;

  if (const Status is = inspect(archive, info); is != Status::ok) return is;

  const Geometry geo{h.dims, brick_grid(h.dims)};
  const std::uint32_t P = h.streams_per_brick;
  const detail::QuantMatrix qm =
      detail::QuantMatrix::build({h.q_base, h.q_a, h.q_b, h.deadzone});
  const detail::ModelSet ms = detail::ModelSet::defaults(h.table_version);
  const TransformOps ops = ops_for(backend);

#ifdef GPUDCT_HAVE_CUDA
  // Explicit request only. The GPU path declines archives it cannot decode
  // exactly (raw bricks, correction layers) and we fall through to the CPU,
  // which is always correct, rather than returning a partial answer.
  if (backend == Backend::cuda && cuda::available()) {
    const Status cs = cuda::decode_archive(archive, h, qm, ms, out);
    if (cs == Status::ok) {
      if (opts.deblock && opts.deblock_strength > 0.0f &&
          (h.flags & detail::kFlagCorrections) == 0)
        deblock_volume(out.data(), h.dims, h.dtype, h.data_scale, h.data_offset,
                       DeblockThresholds::from(qm, opts.deblock_strength), opts.threads);
      return Status::ok;
    }
    if (cs != Status::backend_unavailable) return cs;
  }
#endif

  if (out.size() != h.dims.voxels() * dtype_size(h.dtype)) return Status::invalid_argument;

  std::atomic<int> failure{static_cast<int>(Status::ok)};
  parallel_for(h.brick_count, opts.threads, [&](std::uint64_t i) {
    if (failure.load(std::memory_order_relaxed) != static_cast<int>(Status::ok)) return;

    const std::size_t e = static_cast<std::size_t>(h.index_offset + i * detail::BrickEntry::kSize);
    const std::uint64_t off = detail::get_u64(archive, e);
    const std::uint32_t size = detail::get_u32(archive, e + 8);
    if (off > archive.size() || size > archive.size() - off) {
      failure.store(static_cast<int>(Status::corrupt_bitstream), std::memory_order_relaxed);
      return;
    }

    std::uint32_t bx, by, bz;
    geo.brick_coords(i, bx, by, bz);

    // Only the raw-brick path still needs the float staging buffer; the coded
    // path scatters each chunk into the volume as it is reconstructed.
    std::vector<float> work(static_cast<std::size_t>(kBrickDim) * kBrickDim * kBrickDim);
    std::vector<std::pair<std::uint32_t, std::int32_t>> corrections;
    const DirectScatter direct{out.data(), h.dims, h.dtype, h.data_scale,
                               h.data_offset, bx, by, bz};
    const Status ds = decode_brick_payload(
        archive.subspan(static_cast<std::size_t>(off), size), qm, ms, P, h.dtype, h.data_scale,
        h.data_offset, brick_valid_extent(h.dims, bx, by, bz), ops, work.data(), &corrections,
        &direct);
    if (ds != Status::ok) {
      failure.store(static_cast<int>(ds), std::memory_order_relaxed);
      return;
    }

    // Corrections are applied to the rounded output value, not to the working
    // float, because the bound was measured against exactly what store_voxel
    // writes.
    for (const auto& [bi, delta] : corrections) {
      const std::uint32_t lx = bi % kBrickDim;
      const std::uint32_t ly = (bi / kBrickDim) % kBrickDim;
      const std::uint32_t lz = bi / (kBrickDim * kBrickDim);
      const std::uint32_t gx = bx * kBrickDim + lx, gy = by * kBrickDim + ly,
                          gz = bz * kBrickDim + lz;
      if (gx >= h.dims.x || gy >= h.dims.y || gz >= h.dims.z) continue;
      const std::size_t gi = (static_cast<std::size_t>(gz) * h.dims.y + gy) * h.dims.x + gx;
      store_voxel(out.data(), h.dtype, gi,
                  load_voxel(out.data(), h.dtype, gi) +
                      static_cast<float>(delta) * h.corr_step);
    }
  });

  const Status f = static_cast<Status>(failure.load());
  if (f != Status::ok) return f;

  // Deblocking would move voxels after the corrections that establish the
  // bound, so an archive carrying a correction layer declines it. The guarantee
  // outranks the cosmetics.
  const bool has_corrections = (h.flags & detail::kFlagCorrections) != 0;
  if (opts.deblock && opts.deblock_strength > 0.0f && !has_corrections)
    deblock_volume(out.data(), h.dims, h.dtype, h.data_scale, h.data_offset,
                   DeblockThresholds::from(qm, opts.deblock_strength), opts.threads);
  return Status::ok;
}

Status decode_brick(std::span<const std::uint8_t> archive, std::uint32_t bx, std::uint32_t by,
                    std::uint32_t bz, std::vector<std::uint8_t>& out, Backend backend) {
  detail::FileHeader h;
  const Status s = detail::read_header(archive, h);
  if (s != Status::ok) return s;

  const Dims grid = brick_grid(h.dims);
  if (bx >= grid.x || by >= grid.y || bz >= grid.z) return Status::invalid_argument;
  const std::uint64_t i =
      (static_cast<std::uint64_t>(bz) * grid.y + by) * grid.x + bx;

  const std::size_t e = static_cast<std::size_t>(h.index_offset + i * detail::BrickEntry::kSize);
  const std::uint64_t off = detail::get_u64(archive, e);
  const std::uint32_t size = detail::get_u32(archive, e + 8);
  if (off > archive.size() || size > archive.size() - off) return Status::corrupt_bitstream;

  const detail::QuantMatrix qm =
      detail::QuantMatrix::build({h.q_base, h.q_a, h.q_b, h.deadzone});
  const detail::ModelSet ms = detail::ModelSet::defaults(h.table_version);

  std::vector<float> work(static_cast<std::size_t>(kBrickDim) * kBrickDim * kBrickDim);
  const Status ds = decode_brick_payload(archive.subspan(static_cast<std::size_t>(off), size),
                                         qm, ms, h.streams_per_brick, h.dtype, h.data_scale,
                                         h.data_offset, brick_valid_extent(h.dims, bx, by, bz),
                                         ops_for(backend), work.data(), nullptr, nullptr);
  if (ds != Status::ok) return ds;

  const std::size_t n = work.size();
  out.assign(n * dtype_size(h.dtype), 0);
  const float inv = 1.0f / h.data_scale;
  for (std::size_t k = 0; k < n; ++k)
    store_voxel(out.data(), h.dtype, k, (work[k] - h.data_offset) * inv);
  return Status::ok;
}

}  // namespace gpudct
