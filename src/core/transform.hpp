// Separable 3D DCT-II / DCT-III over a 16x16x16 chunk.
//
// Voxel layout is z-major, x contiguous: idx(x,y,z) = (z*16 + y)*16 + x.
// Coefficient layout uses the same indexing with (u,v,w) frequency triples.
//
// The three passes are deliberately written as "gather a pencil, transform,
// scatter it back" rather than fused: it is the shape the CUDA kernel uses (one
// thread per pencil) and the shape the SIMD kernels vectorize across, so the
// reference and the optimized paths stay structurally comparable.
#pragma once

#include "dct16.hpp"
#include "gpudct/types.hpp"

namespace gpudct::detail {

inline constexpr int kD = kChunkDim;   // 16
inline constexpr int kD2 = kD * kD;    // 256

// Transform along the contiguous (x) axis.
template <bool Forward>
inline void pass_x(const float* __restrict src, float* __restrict dst) {
  for (int zy = 0; zy < kD2; ++zy) {
    const float* in = src + zy * kD;
    float* out = dst + zy * kD;
    if constexpr (Forward) dct16(in, out);
    else idct16(in, out);
  }
}

// Transform along y (stride 16 within a z-slice).
template <bool Forward>
inline void pass_y(const float* __restrict src, float* __restrict dst) {
  float pencil[kD], result[kD];
  for (int z = 0; z < kD; ++z) {
    for (int x = 0; x < kD; ++x) {
      const int base = z * kD2 + x;
      for (int y = 0; y < kD; ++y) pencil[y] = src[base + y * kD];
      if constexpr (Forward) dct16(pencil, result);
      else idct16(pencil, result);
      for (int y = 0; y < kD; ++y) dst[base + y * kD] = result[y];
    }
  }
}

// Transform along z (stride 256).
template <bool Forward>
inline void pass_z(const float* __restrict src, float* __restrict dst) {
  float pencil[kD], result[kD];
  for (int yx = 0; yx < kD2; ++yx) {
    for (int z = 0; z < kD; ++z) pencil[z] = src[yx + z * kD2];
    if constexpr (Forward) dct16(pencil, result);
    else idct16(pencil, result);
    for (int z = 0; z < kD; ++z) dst[yx + z * kD2] = result[z];
  }
}

// Forward 3D DCT-II. in and out may alias.
inline void forward_chunk(const float* __restrict in, float* __restrict out) {
  float a[kChunkVox], b[kChunkVox];
  pass_x<true>(in, a);
  pass_y<true>(a, b);
  pass_z<true>(b, out);
}

// Inverse 3D DCT-III. in and out may alias.
inline void inverse_chunk(const float* __restrict in, float* __restrict out) {
  float a[kChunkVox], b[kChunkVox];
  pass_z<false>(in, a);
  pass_y<false>(a, b);
  pass_x<false>(b, out);
}

// Reference variants using the naive matrix product, for validation only.
inline void forward_chunk_ref(const float* in, float* out) {
  float a[kChunkVox], b[kChunkVox];
  for (int zy = 0; zy < kD2; ++zy) dct16_ref(in + zy * kD, a + zy * kD);
  {
    float p[kD], r[kD];
    for (int z = 0; z < kD; ++z)
      for (int x = 0; x < kD; ++x) {
        const int base = z * kD2 + x;
        for (int y = 0; y < kD; ++y) p[y] = a[base + y * kD];
        dct16_ref(p, r);
        for (int y = 0; y < kD; ++y) b[base + y * kD] = r[y];
      }
  }
  {
    float p[kD], r[kD];
    for (int yx = 0; yx < kD2; ++yx) {
      for (int z = 0; z < kD; ++z) p[z] = b[yx + z * kD2];
      dct16_ref(p, r);
      for (int z = 0; z < kD; ++z) out[yx + z * kD2] = r[z];
    }
  }
}

// --------------------------------------------------------------------------
// Frequency-band classification, used for entropy-coding contexts and for the
// progressive profile. Bands are radial shells over sub-block coordinates, so a
// band is a whole number of sub-blocks and the mask hierarchy can be ordered by
// band without splitting any 64-bit mask.
// --------------------------------------------------------------------------
inline constexpr int kNumBands = 4;

// su, sv, sw are sub-block coordinates in [0,4).
[[nodiscard]] constexpr int band_of_subblock(int su, int sv, int sw) noexcept {
  const int r = su + sv + sw;  // 0..9
  if (r == 0) return 0;
  if (r <= 2) return 1;
  if (r <= 4) return 2;
  return 3;
}

// Linear sub-block index (0..63) -> band. Sub-block index is (sw*4 + sv)*4 + su.
[[nodiscard]] constexpr int band_of_subblock_index(int sb) noexcept {
  const int su = sb & 3;
  const int sv = (sb >> 2) & 3;
  const int sw = (sb >> 4) & 3;
  return band_of_subblock(su, sv, sw);
}

// Coefficient index (0..4095) in (u,v,w) space -> its sub-block index (0..63).
[[nodiscard]] constexpr int subblock_of_coeff(int idx) noexcept {
  const int u = idx & 15;
  const int v = (idx >> 4) & 15;
  const int w = (idx >> 8) & 15;
  return ((w >> 2) * 4 + (v >> 2)) * 4 + (u >> 2);
}

// Coefficient index -> its bit position (0..63) within its sub-block mask.
[[nodiscard]] constexpr int bit_of_coeff(int idx) noexcept {
  const int u = idx & 15;
  const int v = (idx >> 4) & 15;
  const int w = (idx >> 8) & 15;
  return ((w & 3) * 4 + (v & 3)) * 4 + (u & 3);
}

// Inverse of the two above: (sub-block, bit) -> coefficient index.
[[nodiscard]] constexpr int coeff_of_subblock_bit(int sb, int bit) noexcept {
  const int su = sb & 3, sv = (sb >> 2) & 3, sw = (sb >> 4) & 3;
  const int bu = bit & 3, bv = (bit >> 2) & 3, bw = (bit >> 4) & 3;
  const int u = su * 4 + bu, v = sv * 4 + bv, w = sw * 4 + bw;
  return (w * 16 + v) * 16 + u;
}

}  // namespace gpudct::detail
