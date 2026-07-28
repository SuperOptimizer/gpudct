// Quality measurement (docs/QUALITY.md section 1).
//
// Two principles this header exists to enforce:
//
//   1. Percentiles are exact, never approximated. A t-digest is fine at p50 and
//      worthless at p99.99, and p99.99 is the number that decides whether a
//      codec is usable for science.
//   2. Aggregate metrics over a large volume are dominated by air. Everything
//      here therefore comes with a conditioned or worst-case companion --
//      per-axis, per-intensity-band, per-brick -- because the whole-volume mean
//      is the least informative number in the set.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "gpudct/types.hpp"

namespace gpudct {

struct Percentiles {
  double p50 = 0, p90 = 0, p95 = 0, p99 = 0, p999 = 0, p9999 = 0, max = 0;
};

struct Metrics {
  std::size_t voxels = 0;
  double data_min = 0, data_max = 0;

  // --- error distribution ---
  double mae = 0, mse = 0, rmse = 0;
  // A nonzero bias means the quantizer offset is wrong, and it compounds under
  // any downstream averaging.
  double bias = 0;
  Percentiles abs_err{};

  // --- fidelity ---
  // psnr_dtype uses the dtype's full range as the peak; psnr_range uses the
  // data's actual range. Scroll volumes rarely span their dtype, so psnr_range
  // is the honest number and psnr_dtype is the one everyone else quotes.
  double psnr_dtype = 0, psnr_range = 0;
  double ssim = 0;      // mean of the 3D SSIM map, 11^3 Gaussian window
  double ssim_p01 = 0;  // 1st percentile of the map: the worst regions

  // --- structure preservation ---
  double gradient_mae = 0, gradient_p99 = 0;
  // Mean |delta| across chunk faces divided by mean |delta| in the interior.
  // 1.0 means no visible seams.
  double blockiness = 0;
  // Ratio of high-pass energy retained. A single scalar for "how much detail
  // did we destroy".
  double laplacian_ratio = 0;
  // Retained energy per radial frequency band, low to high. A codec that kills
  // fibre texture shows a clean rolloff here while PSNR still looks fine.
  std::array<double, 4> band_energy_ratio{};
  // Error in per-8^3-block standard deviation. Catches the failure where flat
  // regions are perfect and textured regions are mush.
  double local_std_err_p50 = 0, local_std_err_p99 = 0;
  // The error field should look like white noise. Structured error means we are
  // removing signal, not noise -- visible here long before it reaches PSNR.
  std::array<double, 3> error_autocorr{};
  double hist_emd = 0;

  // --- anisotropy: a separable 3D transform can fail on one axis only, and
  // slice-wise viewing hides it completely ---
  std::array<double, 3> worst_slice_mae{};  // normal to x, y, z
  std::array<double, 3> axis_gradient_mae{};

  // --- conditioned: the numbers that predict downstream utility ---
  std::array<double, 4> mae_by_intensity{};  // quartiles of the data range
  double mae_high_gradient = 0;              // top decile of |grad| in the original

  // --- rate, filled in by the caller that knows the byte counts ---
  double bits_per_voxel = 0;
  double ratio = 0;
};

// orig and dec are the two volumes as f32, both dims.voxels() long.
[[nodiscard]] Metrics compute_metrics(const float* orig, const float* dec, Dims dims,
                                      DType dtype);

// Exact absolute-error percentiles. Two passes and a histogram, no sampling and
// no sketch: the reported value is a value that actually occurs in the data.
[[nodiscard]] Percentiles exact_abs_percentiles(const float* a, const float* b,
                                                std::size_t n);

struct BrickScore {
  std::uint32_t bx = 0, by = 0, bz = 0;
  double rmse = 0;
  double p99 = 0;
  double max_abs = 0;
};

// The worst bricks by RMSE, so bad regions can be inspected directly rather than
// inferred from an aggregate.
[[nodiscard]] std::vector<BrickScore> worst_bricks(const float* orig, const float* dec,
                                                   Dims dims, std::size_t count);

// Human-readable report. `verbose` adds the anisotropy and conditioned sections.
[[nodiscard]] std::string format_report(const Metrics& m, bool verbose);

}  // namespace gpudct
