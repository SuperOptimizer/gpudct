// Frequency-weighted dead-zone quantization (docs/DESIGN.md section 3.3).
//
// The weighting is radial and isotropic. Image codecs weight per-axis to match
// the human visual system; volumetric CT has no such asymmetry to exploit and is
// usually consumed by a model rather than an eye, so a radial shell is the
// honest choice.
#pragma once

#include <array>
#include <cmath>

#include "gpudct/types.hpp"

namespace gpudct::detail {

struct QuantMatrix {
  std::array<float, kChunkVox> q{};   // step size per coefficient
  std::array<float, kChunkVox> rq{};  // 1/q, so the hot loop has no division

  static QuantMatrix build(const QuantParams& p) {
    QuantMatrix m;
    // Normalizer for the radial coordinate: the far corner (15,15,15) maps to 1.
    const float rmax = std::sqrt(3.0f * 15.0f * 15.0f);
    for (int w = 0; w < kChunkDim; ++w)
      for (int v = 0; v < kChunkDim; ++v)
        for (int u = 0; u < kChunkDim; ++u) {
          const float r =
              std::sqrt(static_cast<float>(u * u + v * v + w * w)) / rmax;
          const float q = p.base * (1.0f + p.a * std::pow(r, p.b));
          const std::size_t idx = static_cast<std::size_t>((w * kChunkDim + v) * kChunkDim + u);
          m.q[idx] = q;
          m.rq[idx] = 1.0f / q;
        }
    return m;
  }
};

// Largest magnitude a level may take. 2^24 is the last integer f32 represents
// exactly, so dequantization stays lossless in the level itself, and the bound
// keeps a pathological input from overflowing the float-to-int conversion --
// which is undefined behaviour, not a large number.
inline constexpr int kMaxLevel = 1 << 24;

// Every dtype is mapped into a working domain of +/-32768 before the transform
// (see derive_mapping), so a chunk's voxel vector has L2 norm at most
// sqrt(4096) * 32768. The transform is orthonormal, so no coefficient can
// exceed that either.
inline constexpr float kMaxCoeff = 64.0f * 32768.0f;

// The largest level a valid encoder could have produced for a given quantizer.
// A decoder that knows this rejects impossible values immediately instead of
// reconstructing nonsense from them -- which matters for robustness, and
// happens to make fuzzing an order of magnitude cheaper because a corrupt
// stream is abandoned early rather than expanded into millions of symbols.
[[nodiscard]] inline int max_plausible_level(const QuantMatrix& qm) noexcept {
  float qmin = qm.q[0];
  for (float q : qm.q) qmin = std::fmin(qmin, q);
  if (!(qmin > 0.0f)) return kMaxLevel;
  const float bound = kMaxCoeff / qmin + 2.0f;
  return (bound >= static_cast<float>(kMaxLevel)) ? kMaxLevel : static_cast<int>(bound);
}

// Dead-zone quantizer. `deadzone` below 0.5 widens the zero bin, which buys rate
// cheaply because the coefficients it zeroes are mostly noise.
[[nodiscard]] inline int quantize(float coeff, float rq, float deadzone) noexcept {
  const float a = std::fabs(coeff) * rq + deadzone;
  if (!(a >= 1.0f)) return 0;  // also catches NaN
  const int mag = (a >= static_cast<float>(kMaxLevel)) ? kMaxLevel : static_cast<int>(a);
  return (coeff < 0.0f) ? -mag : mag;
}

[[nodiscard]] inline float dequantize(int level, float q) noexcept {
  return static_cast<float>(level) * q;
}

}  // namespace gpudct::detail
