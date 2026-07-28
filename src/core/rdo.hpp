// Rate-distortion optimized quantization (docs/DESIGN.md section 3.3).
//
// The plain dead-zone quantizer picks each level by rounding, which minimizes
// distortion and ignores what the level costs to code. RDO picks the level that
// minimizes J = D + lambda*R instead -- so a coefficient whose rounded level is
// expensive but only marginally better than a cheaper one gets the cheaper one.
//
// Encoder-only. The bitstream it produces is ordinary and decodes with no
// knowledge that this ran, so it can be enabled or disabled per archive and
// costs decoders nothing.
//
// Two stages, both parallel over independent units, which is what keeps this
// portable to the GPU:
//
//   1. Per coefficient: choose between the rounded level, one step toward zero,
//      and zero.
//   2. Per sub-block: zero the whole sub-block if that is cheaper overall.
//      This is the stage that matters most, because zeroing every coefficient in
//      a sub-block also removes its 64-bit significance mask and its bit in the
//      level above -- a saving the per-coefficient stage cannot see.
//
// The rate estimates use the *actual* coding context, tracked as the pass walks
// a sub-block in coding order. An earlier version used a fixed mid context to
// keep every coefficient independent; that was measurably worse once the context
// set grew, because a single representative context is a poor stand-in for five.
// Sub-blocks remain independent of each other, which is the parallelism that
// matters -- the serialization is only within one sub-block's 64 coefficients.
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>

#include "models.hpp"
#include "quant.hpp"
#include "transform.hpp"

namespace gpudct::detail {

// Cost in bits of coding a magnitude in a given band, plus its sign.
[[nodiscard]] inline float magnitude_bits(const ModelSet& ms, int band, int ctx,
                                          std::int32_t mag, bool is_dc) {
  if (mag == 0) return 0.0f;
  float bits = 1.0f;  // sign, always a bypass bit

  if (is_dc) {
    const int nb = 32 - std::countl_zero(static_cast<std::uint32_t>(mag));
    bits += ms[ModelIndex::dc_len].bit_cost(static_cast<std::uint32_t>(nb));
    bits += static_cast<float>(nb - 1);  // mantissa, bypass
    return bits;
  }

  const Model& lm = ms[ModelIndex::level(band, ctx)];
  if (mag <= 15) {
    bits += lm.bit_cost(static_cast<std::uint32_t>(mag - 1));
  } else {
    bits += lm.bit_cost(kLevelEscape);
    const std::uint32_t rest = static_cast<std::uint32_t>(mag) - 15;
    const int nb = 32 - std::countl_zero(rest);
    bits += ms[ModelIndex::esc_len].bit_cost(static_cast<std::uint32_t>(nb));
    bits += static_cast<float>(nb - 1);
  }
  return bits;
}

// Bits to signal that a sub-block is significant: its eight L2 mask bytes. An
// all-zero sub-block still costs a bit in the L1 mask, so only the difference
// matters here.
[[nodiscard]] inline float subblock_overhead_bits(const ModelSet& ms, int band) {
  // A significant sub-block writes eight mask bytes; an empty one writes none.
  // Estimated at the cost of a typical sparse byte rather than a specific value.
  return 8.0f * ms[ModelIndex::l2(band, 0)].bit_cost(1);
}

// Lagrange multiplier scale for a quantizer step.
//
// At the rate-distortion optimum the slope dD/dR is -lambda, and for a uniform
// quantizer of step q the distortion scale is q^2, so lambda = scale * q^2.
//
// The scale is empirical. Measured by BD-rate against plain quantization on a
// real scroll brick:
//
//   scale   0.04    0.08    0.12    0.20    0.35    0.60
//   BD-PSNR -2.06%  -2.70%  -2.81%  -1.85%  +1.89%  +8.94%
//
// so 0.12 it is -- shallow either side of 0.08-0.12, and sharply worse past 0.2
// where the coder starts discarding coefficients it should have kept.
// GPUDCT_RDO_LAMBDA overrides it so the constant stays re-derivable.
[[nodiscard]] inline float rdo_lambda_scale() {
  static const float v = [] {
    if (const char* e = std::getenv("GPUDCT_RDO_LAMBDA")) {
      const float f = std::strtof(e, nullptr);
      if (f > 0.0f) return f;
    }
    return 0.12f;
  }();
  return v;
}

// Rewrites `levels` in place with rate-distortion optimized choices.
//
// `coeffs` are the untransformed-domain coefficients, `levels` the plain
// quantizer's output for them.
inline void rdo_optimize(const float* __restrict coeffs, const QuantMatrix& qm,
                         const ModelSet& ms, std::int32_t* __restrict levels) {
  for (int sb = 0; sb < kSubCount; ++sb) {
    const int band = band_of_subblock_index(sb);
    const float lambda_band = rdo_lambda_scale();

    float keep_cost = 0.0f;   // J of the sub-block as chosen
    float zero_cost = 0.0f;   // J of forcing the whole sub-block to zero
    bool any_nonzero = false;
    std::int32_t nb[kSubVox] = {};  // neighbour magnitudes, as the coder tracks them

    for (int bit = 0; bit < kSubVox; ++bit) {
      const int idx = coeff_of_subblock_bit(sb, bit);
      const float c = coeffs[idx];
      const float q = qm.q[static_cast<std::size_t>(idx)];
      const float lambda = lambda_band * q * q;
      const std::int32_t base = levels[idx];
      const std::int32_t mag = base < 0 ? -base : base;
      const float ac = std::fabs(c);

      // Distortion of coding zero, needed by both the per-coefficient choice and
      // the sub-block decision.
      const float d_zero = ac * ac;
      zero_cost += d_zero;

      if (mag == 0) {
        keep_cost += d_zero;
        continue;
      }

      const int bu = bit & 3, bv = (bit >> 2) & 3, bw = (bit >> 4) & 3;
      std::int32_t nsum = 0;
      if (bu > 0) nsum += nb[bit - 1];
      if (bv > 0) nsum += nb[bit - 4];
      if (bw > 0) nsum += nb[bit - 16];
      const int ctx = level_ctx_of_prev(nsum);

      // Candidates: the rounded level, one step toward zero, and zero.
      std::int32_t best_mag = 0;
      float best_j = d_zero;  // zero costs no bits
      for (std::int32_t m = mag; m >= mag - 1 && m >= 1; --m) {
        const float err = ac - static_cast<float>(m) * q;
        const float j = err * err +
                        lambda * magnitude_bits(ms, band, ctx, m, idx == 0);
        if (j < best_j) {
          best_j = j;
          best_mag = m;
        }
      }

      levels[idx] = (c < 0.0f) ? -best_mag : best_mag;
      keep_cost += best_j;
      nb[bit] = best_mag;
      if (best_mag != 0) any_nonzero = true;
    }

    if (!any_nonzero) continue;

    // Sub-block decision. Zeroing removes the eight mask bytes as well as every
    // level, which is the part the per-coefficient pass cannot account for.
    const float q0 = qm.q[static_cast<std::size_t>(coeff_of_subblock_bit(sb, 0))];
    const float overhead = lambda_band * q0 * q0 * subblock_overhead_bits(ms, band);
    if (zero_cost < keep_cost + overhead) {
      for (int bit = 0; bit < kSubVox; ++bit) levels[coeff_of_subblock_bit(sb, bit)] = 0;
    }
  }
}

}  // namespace gpudct::detail
