// Lane-parallel 16-point DCT (docs/ROADMAP.md M6).
//
// The scalar reference transforms one pencil at a time. This version transforms
// sixteen at once by putting each pencil in its own lane, which is what makes
// the inner loops vectorize: 16 contiguous f32 is exactly one AVX-512 register,
// two AVX2 registers, or four NEON registers, and the chunk edge is 16 by
// construction (docs/DESIGN.md section 2).
//
// The important property is that this is *bit-identical* to the scalar path,
// not merely close. Each lane executes exactly the same sequence of scalar
// float operations, in the same order, as the reference does for its pencil --
// vectorizing across independent lanes never reassociates anything. Combined
// with -ffp-contract=off that makes the SIMD backend produce byte-identical
// archives and byte-identical reconstructions, so it can be tested by equality
// rather than by tolerance. That is a much stronger guarantee than
// docs/QUALITY.md section 4.2 asks for, and it is free.
#pragma once

#include <cstring>

#include "core/dct16.hpp"
#include "gpudct/types.hpp"

namespace gpudct::detail {

// Sixteen pencils, interleaved: v[k * 16 + l] is sample k of lane l.
inline constexpr int kLanes = 16;

template <int N>
inline void lee_fwd_lanes(float* __restrict v) {
  constexpr int H = N / 2;
  const auto& c = dct_constants();
  const float* s = nullptr;
  if constexpr (N == 16) s = c.s16.data();
  else if constexpr (N == 8) s = c.s8.data();
  else if constexpr (N == 4) s = c.s4.data();
  else s = c.s2.data();

  float g[H * kLanes], h[H * kLanes];
  for (int k = 0; k < H; ++k) {
    const float sk = s[k];
    const float* a = v + k * kLanes;
    const float* b = v + (N - 1 - k) * kLanes;
    float* gk = g + k * kLanes;
    float* hk = h + k * kLanes;
    for (int l = 0; l < kLanes; ++l) {
      const float av = a[l], bv = b[l];
      gk[l] = av + bv;
      hk[l] = (av - bv) * sk;
    }
  }

  lee_fwd_lanes<H>(g);
  lee_fwd_lanes<H>(h);

  for (int k = 0; k < H; ++k)
    for (int l = 0; l < kLanes; ++l) v[2 * k * kLanes + l] = g[k * kLanes + l];
  for (int k = 0; k < H - 1; ++k)
    for (int l = 0; l < kLanes; ++l)
      v[(2 * k + 1) * kLanes + l] = h[k * kLanes + l] + h[(k + 1) * kLanes + l];
  for (int l = 0; l < kLanes; ++l) v[(N - 1) * kLanes + l] = h[(H - 1) * kLanes + l];
}

template <>
inline void lee_fwd_lanes<1>(float* __restrict) {}

template <int N>
inline void lee_inv_lanes(float* __restrict v) {
  constexpr int H = N / 2;
  const auto& c = dct_constants();
  const float* s = nullptr;
  if constexpr (N == 16) s = c.s16.data();
  else if constexpr (N == 8) s = c.s8.data();
  else if constexpr (N == 4) s = c.s4.data();
  else s = c.s2.data();

  float g[H * kLanes], h[H * kLanes];
  for (int k = 0; k < H; ++k)
    for (int l = 0; l < kLanes; ++l) g[k * kLanes + l] = v[2 * k * kLanes + l];
  for (int l = 0; l < kLanes; ++l) h[(H - 1) * kLanes + l] = v[(N - 1) * kLanes + l];
  for (int k = H - 2; k >= 0; --k)
    for (int l = 0; l < kLanes; ++l)
      h[k * kLanes + l] = v[(2 * k + 1) * kLanes + l] - h[(k + 1) * kLanes + l];

  lee_inv_lanes<H>(g);
  lee_inv_lanes<H>(h);

  for (int k = 0; k < H; ++k) {
    const float half_over_s = 0.5f / s[k];
    for (int l = 0; l < kLanes; ++l) {
      const float d = h[k * kLanes + l] * half_over_s;
      const float m = g[k * kLanes + l] * 0.5f;
      v[k * kLanes + l] = m + d;
      v[(N - 1 - k) * kLanes + l] = m - d;
    }
  }
}

template <>
inline void lee_inv_lanes<1>(float* __restrict) {}

inline void dct16_lanes(float* __restrict v) {
  const auto& c = dct_constants();
  lee_fwd_lanes<16>(v);
  for (int u = 0; u < 16; ++u) {
    const float a = c.alpha[static_cast<std::size_t>(u)];
    for (int l = 0; l < kLanes; ++l) v[u * kLanes + l] *= a;
  }
}

inline void idct16_lanes(float* __restrict v) {
  const auto& c = dct_constants();
  for (int u = 0; u < 16; ++u) {
    const float a = c.inv_alpha[static_cast<std::size_t>(u)];
    for (int l = 0; l < kLanes; ++l) v[u * kLanes + l] *= a;
  }
  lee_inv_lanes<16>(v);
}

// 16x16 transpose of one z-slice, so the x-axis pass can reuse the lane-parallel
// kernel instead of needing a separate horizontal implementation.
inline void transpose16(const float* __restrict src, float* __restrict dst) {
  for (int a = 0; a < 16; ++a)
    for (int b = 0; b < 16; ++b) dst[a * 16 + b] = src[b * 16 + a];
}

// --------------------------------------------------------------------------
// 3D chunk transforms. Pass order matches the scalar reference exactly (x, then
// y, then z forward; the reverse on the way back), because differing pass order
// would change per-voxel rounding even though the maths is equivalent.
// --------------------------------------------------------------------------

inline void forward_chunk_simd(const float* __restrict in, float* __restrict out) {
  alignas(64) float buf[kChunkVox];
  alignas(64) float tmp[256];
  std::memcpy(buf, in, sizeof(buf));

  // x: samples along x, lanes along y, one z-slice at a time.
  for (int z = 0; z < 16; ++z) {
    float* slice = buf + z * 256;
    transpose16(slice, tmp);
    dct16_lanes(tmp);
    transpose16(tmp, slice);
  }
  // y: samples along y, lanes along x -- already the right layout.
  for (int z = 0; z < 16; ++z) dct16_lanes(buf + z * 256);
  // z: samples along z (stride 256), lanes along x, one y-row at a time.
  for (int y = 0; y < 16; ++y) {
    for (int z = 0; z < 16; ++z)
      std::memcpy(tmp + z * 16, buf + z * 256 + y * 16, 16 * sizeof(float));
    dct16_lanes(tmp);
    for (int z = 0; z < 16; ++z)
      std::memcpy(buf + z * 256 + y * 16, tmp + z * 16, 16 * sizeof(float));
  }
  std::memcpy(out, buf, sizeof(buf));
}

// Inverse transform with a sparsity hint.
//
// `nonzero` is the number of nonzero coefficients, which the entropy decoder
// already counted while parsing the significance masks -- so the two shortcuts
// below are free to test. Deriving the same facts by scanning the coefficient
// array costs more than it saves: discovering that a chunk is *not* DC-only
// means reading all 4095 AC coefficients first, and at any useful rate most
// chunks are not.
inline void inverse_chunk_simd_hint(const float* __restrict in, float* __restrict out,
                                    int nonzero) {
  if (nonzero == 0) {
    for (int i = 0; i < kChunkVox; ++i) out[i] = 0.0f;
    return;
  }
  // A DC-only chunk reconstructs to a constant, and those dominate air and
  // background -- most of a scroll volume by count. The forward transform maps a
  // constant c to a DC of 64c (16 * alpha(0) per axis, cubed).
  if (nonzero == 1 && in[0] != 0.0f) {
    const float c = in[0] * (1.0f / 64.0f);
    for (int i = 0; i < kChunkVox; ++i) out[i] = c;
    return;
  }

  alignas(64) float buf[kChunkVox];
  alignas(64) float tmp[256];
  std::memcpy(buf, in, sizeof(buf));

  for (int y = 0; y < 16; ++y) {
    for (int z = 0; z < 16; ++z)
      std::memcpy(tmp + z * 16, buf + z * 256 + y * 16, 16 * sizeof(float));
    idct16_lanes(tmp);
    for (int z = 0; z < 16; ++z)
      std::memcpy(buf + z * 256 + y * 16, tmp + z * 16, 16 * sizeof(float));
  }
  for (int z = 0; z < 16; ++z) idct16_lanes(buf + z * 256);
  for (int z = 0; z < 16; ++z) {
    float* slice = buf + z * 256;
    transpose16(slice, tmp);
    idct16_lanes(tmp);
    transpose16(tmp, slice);
  }
  std::memcpy(out, buf, sizeof(buf));
}

inline void inverse_chunk_simd(const float* __restrict in, float* __restrict out) {
  inverse_chunk_simd_hint(in, out, 2);  // 2 = "no hint": take the full path
}

}  // namespace gpudct::detail
