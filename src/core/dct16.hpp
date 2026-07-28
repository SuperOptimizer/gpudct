// 16-point DCT-II / DCT-III in f32.
//
// Two implementations, both orthonormal and mutually validated by
// tests/test_transform.cpp:
//
//   dct16_ref / idct16_ref    naive matrix product, 256 multiplies. The oracle.
//   dct16 / idct16            Lee's recursive factorization, ~31 multiplies.
//
// Orthonormal scaling: X[u] = a(u) * sum_n x[n] cos((2n+1)u*pi/32),
// with a(0) = sqrt(1/16) and a(u>0) = sqrt(2/16). The inverse is the transpose,
// so a round trip is the identity and Parseval holds -- which matters, because
// the quantizer's error model assumes coefficient-domain error maps 1:1 to
// voxel-domain error.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace gpudct::detail {

// --------------------------------------------------------------------------
// Constant tables, built once at static-init. Not constexpr because <cmath> is
// only constexpr from C++26 and we still want to build under C++23 toolchains.
// --------------------------------------------------------------------------
struct DctConstants {
  // Orthonormal matrix M[u][n] = a(u) * cos((2n+1)*u*pi/32), row-major.
  std::array<float, 256> m{};
  // Per-frequency orthonormal scale a(u), and its reciprocal.
  std::array<float, 16> alpha{};
  std::array<float, 16> inv_alpha{};
  // Lee butterfly constants s_N[k] = 1 / (2*cos((2k+1)*pi/(2N))) for N = 16, 8, 4, 2.
  std::array<float, 8> s16{};
  std::array<float, 4> s8{};
  std::array<float, 2> s4{};
  std::array<float, 1> s2{};

  DctConstants() {
    constexpr double pi = std::numbers::pi_v<double>;
    for (int u = 0; u < 16; ++u) {
      const double a = (u == 0) ? std::sqrt(1.0 / 16.0) : std::sqrt(2.0 / 16.0);
      alpha[static_cast<std::size_t>(u)] = static_cast<float>(a);
      inv_alpha[static_cast<std::size_t>(u)] = static_cast<float>(1.0 / a);
      for (int n = 0; n < 16; ++n) {
        const double c = std::cos((2 * n + 1) * u * pi / 32.0);
        m[static_cast<std::size_t>(u * 16 + n)] = static_cast<float>(a * c);
      }
    }
    auto fill = [&](auto& tab, int n) {
      for (std::size_t k = 0; k < tab.size(); ++k) {
        const double c = std::cos((2.0 * static_cast<double>(k) + 1.0) * pi / (2.0 * n));
        tab[k] = static_cast<float>(1.0 / (2.0 * c));
      }
    };
    fill(s16, 16);
    fill(s8, 8);
    fill(s4, 4);
    fill(s2, 2);
  }
};

// A namespace-scope inline variable rather than a function-local static: a
// function-local static costs a thread-safe-initialization guard check on every
// call, and the recursive butterflies below call this several hundred thousand
// times per brick. That guard was measurably the largest single cost in the
// transform before this was changed.
inline const DctConstants kDctConstants{};

inline const DctConstants& dct_constants() { return kDctConstants; }

// --------------------------------------------------------------------------
// Reference: naive matrix product. Slow, obviously correct, never deleted.
// --------------------------------------------------------------------------
inline void dct16_ref(const float* __restrict in, float* __restrict out) {
  const auto& c = dct_constants();
  for (int u = 0; u < 16; ++u) {
    float acc = 0.0f;
    for (int n = 0; n < 16; ++n) acc += c.m[static_cast<std::size_t>(u * 16 + n)] * in[n];
    out[u] = acc;
  }
}

inline void idct16_ref(const float* __restrict in, float* __restrict out) {
  const auto& c = dct_constants();
  for (int n = 0; n < 16; ++n) {
    float acc = 0.0f;
    for (int u = 0; u < 16; ++u) acc += c.m[static_cast<std::size_t>(u * 16 + n)] * in[u];
    out[n] = acc;
  }
}

// --------------------------------------------------------------------------
// Lee's recursive factorization of the unnormalized DCT-II
//   C_N(x)[u] = sum_n x[n] * cos((2n+1)*u*pi/(2N))
//
//   g[k] = x[k] + x[N-1-k]
//   h[k] = (x[k] - x[N-1-k]) * s_N[k]
//   C_N(x)[2k]   = C_{N/2}(g)[k]
//   C_N(x)[2k+1] = C_{N/2}(h)[k] + C_{N/2}(h)[k+1]      (last term: H[N/2-1])
//
// Orthonormal scaling is applied by the callers below, so this stays a pure
// linear map that the inverse can undo exactly.
// --------------------------------------------------------------------------
template <int N>
inline void lee_fwd(const float* __restrict x, float* __restrict X) {
  constexpr int H = N / 2;
  float g[H], h[H], G[H], Hh[H];
  const auto& c = dct_constants();
  const float* s = nullptr;
  if constexpr (N == 16) s = c.s16.data();
  else if constexpr (N == 8) s = c.s8.data();
  else if constexpr (N == 4) s = c.s4.data();
  else s = c.s2.data();

  for (int k = 0; k < H; ++k) {
    g[k] = x[k] + x[N - 1 - k];
    h[k] = (x[k] - x[N - 1 - k]) * s[k];
  }
  lee_fwd<H>(g, G);
  lee_fwd<H>(h, Hh);
  for (int k = 0; k < H; ++k) X[2 * k] = G[k];
  for (int k = 0; k < H - 1; ++k) X[2 * k + 1] = Hh[k] + Hh[k + 1];
  X[N - 1] = Hh[H - 1];
}

template <>
inline void lee_fwd<1>(const float* __restrict x, float* __restrict X) {
  X[0] = x[0];
}

template <int N>
inline void lee_inv(const float* __restrict X, float* __restrict x) {
  constexpr int H = N / 2;
  float g[H], h[H], G[H], Hh[H];
  const auto& c = dct_constants();
  const float* s = nullptr;
  if constexpr (N == 16) s = c.s16.data();
  else if constexpr (N == 8) s = c.s8.data();
  else if constexpr (N == 4) s = c.s4.data();
  else s = c.s2.data();

  for (int k = 0; k < H; ++k) G[k] = X[2 * k];
  Hh[H - 1] = X[N - 1];
  for (int k = H - 2; k >= 0; --k) Hh[k] = X[2 * k + 1] - Hh[k + 1];

  lee_inv<H>(G, g);
  lee_inv<H>(Hh, h);

  for (int k = 0; k < H; ++k) {
    const float d = h[k] * (0.5f / s[k]);  // == (x[k] - x[N-1-k]) / 2
    const float m = g[k] * 0.5f;           // == (x[k] + x[N-1-k]) / 2
    x[k] = m + d;
    x[N - 1 - k] = m - d;
  }
}

template <>
inline void lee_inv<1>(const float* __restrict X, float* __restrict x) {
  x[0] = X[0];
}

// --------------------------------------------------------------------------
// Fast orthonormal 16-point transforms.
// --------------------------------------------------------------------------
inline void dct16(const float* __restrict in, float* __restrict out) {
  const auto& c = dct_constants();
  float t[16];
  lee_fwd<16>(in, t);
  for (int u = 0; u < 16; ++u) out[u] = t[u] * c.alpha[static_cast<std::size_t>(u)];
}

inline void idct16(const float* __restrict in, float* __restrict out) {
  const auto& c = dct_constants();
  float t[16];
  for (int u = 0; u < 16; ++u) t[u] = in[u] * c.inv_alpha[static_cast<std::size_t>(u)];
  lee_inv<16>(t, out);
}

}  // namespace gpudct::detail
