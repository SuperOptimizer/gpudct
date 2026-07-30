// Chunk-boundary deblocking (docs/DESIGN.md section 3.8).
//
// Independent block transforms disagree at their shared faces, and at low rates
// that disagreement is visible as a 16-voxel lattice. It costs SSIM, and more
// importantly it puts a false edge into every downstream gradient and
// segmentation operator.
//
// This filter is decoder-side and entirely optional: skipping it still produces
// the format's defined reconstruction. It is deliberately conditional, in the
// H.264 sense -- a face is only smoothed when the jump across it is small enough
// to be quantization noise and the neighbourhood either side is flat enough that
// there was no real edge there to begin with. A filter that smoothed
// unconditionally would erase genuine structure that happens to land on a
// multiple of 16, which is a far worse failure than a visible seam.
#pragma once

#include <algorithm>
#include <cmath>

#include "gpudct/types.hpp"
#include "quant.hpp"

namespace gpudct::detail {

// Expected voxel-domain RMS reconstruction error for a quant matrix.
//
// The transform is orthonormal, so error energy in the coefficient domain equals
// error energy in the voxel domain. A dead-zone quantizer with step q leaves
// error variance q^2/12 per coefficient, so the per-voxel RMS is the root mean
// of that over the matrix. Deriving the filter thresholds from this rather than
// from a magic constant is what lets one strength setting work across profiles
// and quality levels.
[[nodiscard]] inline float expected_voxel_rms(const QuantMatrix& qm) noexcept {
  double acc = 0.0;
  for (float q : qm.q) acc += static_cast<double>(q) * q;
  return static_cast<float>(std::sqrt(acc / static_cast<double>(kChunkVox) / 12.0));
}

struct DeblockThresholds {
  float alpha;  // largest cross-face jump still considered an artifact
  float beta;   // largest neighbourhood variation still considered flat
  float clip;   // largest correction applied to any one voxel

  static DeblockThresholds from_rms(float rms) noexcept {
    return {3.0f * rms, 2.0f * rms, rms};
  }
};

// Reconciles the quantizer's predicted error with what the data can actually
// support.
//
// expected_voxel_rms is scale-free: it grows linearly in the quantizer step
// forever. Real reconstruction error does not, because it cannot exceed the
// signal's own local variation -- once the step is coarse enough to zero a
// coefficient, the error is that coefficient's magnitude, not the step. So the
// model overpredicts at coarse quantizers, exactly where the filter is already
// most aggressive, and the result is a filter that smooths chunk faces flatter
// than the volume's interior. Measured, the shipped strength overshot at every
// rate and cost PSNR outright at quality 1.0 (docs/QUALITY.md 2.2).
//
// `activity` is the mean absolute first difference along the filtered axis,
// taken over interior steps of the decoded volume -- the same quantity the
// blockiness index divides by. It is per axis because scroll data is not
// isotropic; on real full-resolution PHerc it differs by 1.8x between axes.
//
// The two limits are combined as a reciprocal sum rather than a minimum. They
// are independent bounds on the same error, so the effective one is softly the
// smaller: a hard min switches abruptly at the crossover and, measured, cut the
// filter to almost nothing across the whole useful rate range. The reciprocal
// form reduces to model_rms when the quantizer is fine and to kActivityGain *
// activity when it is coarse, with no discontinuity between.
//
// kActivityGain was fitted by back-solving the measured seam-optimal thresholds
// at quality 0.25, 0.5 and 1.0, where model_rms spans 19.5 to 4.9. It comes out
// at 1.74 / 1.64 / 1.78 -- constant to within the measurement, which is the
// evidence that this is the right functional form and not a curve fit.
inline constexpr float kActivityGain = 1.7f;

[[nodiscard]] inline DeblockThresholds deblock_thresholds(float model_rms, float activity,
                                                          float strength) noexcept {
  const float a = kActivityGain * std::max(activity, 1e-6f);
  const float rms = (model_rms * a / (model_rms + a)) * std::max(0.0f, strength);
  return DeblockThresholds::from_rms(rms);
}

// Filters one face. p1,p0 are the two voxels before the boundary and q0,q1 the
// two after; p0/q0 are adjusted in place.
//
// The correction is the H.264 luma form: it moves p0 and q0 toward each other by
// an amount driven mostly by the jump itself and partly by the slope of the
// surrounding voxels, so a face sitting on a genuine ramp is straightened rather
// than flattened.
inline void filter_face(float p1, float& p0, float& q0, float q1,
                        const DeblockThresholds& t) noexcept {
  const float d = q0 - p0;
  if (std::fabs(d) >= t.alpha) return;                 // too big to be quantization noise
  if (std::fabs(p1 - p0) >= t.beta) return;            // real structure on the near side
  if (std::fabs(q1 - q0) >= t.beta) return;            // real structure on the far side

  float delta = (4.0f * d + (p1 - q1)) * 0.125f;
  delta = std::clamp(delta, -t.clip, t.clip);
  p0 += delta;
  q0 -= delta;
}

}  // namespace gpudct::detail
