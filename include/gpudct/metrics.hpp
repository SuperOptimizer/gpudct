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
#include <limits>
#include <string>
#include <vector>

#include "gpudct/types.hpp"

namespace gpudct {

struct Percentiles {
  double p50 = 0, p90 = 0, p95 = 0, p99 = 0, p999 = 0, p9999 = 0, max = 0;
};

// --------------------------------------------------------------------------
// Isosurface displacement (docs/QUALITY.md section 1.3).
//
// The downstream task on scroll CT is tracing a papyrus sheet as an isosurface
// and flattening it. What ruins that is not noise, it is the surface *moving*:
// a codec can hold a fine PSNR while shifting the surface a fraction of a voxel,
// and nothing else in this file would notice. Distances are in voxels.
// --------------------------------------------------------------------------
struct IsoDispStats {
  double mean_abs = 0, p99 = 0, max_abs = 0, signed_mean = 0;
};

struct IsoDisplacement {
  double isovalue = 0;              // the threshold actually used
  std::uint64_t edges = 0;          // grid edges examined
  std::uint64_t crossings = 0;      // edges crossed by both surfaces: the sample count
  double crossing_frac = 0;         // original's crossings / edges -- surface density
  // Edges where exactly one of the two volumes crosses, over edges where either
  // does. The surface appeared or vanished rather than moved, which is worse
  // than any displacement: it breaks the trace instead of bending it.
  double topology_frac = 0;

  // Shift of the crossing along the edge. This is the literal quantity, and an
  // upper bound on how far the surface actually went: a surface oblique to the
  // grid moves d/|n.a| along axis a for a normal displacement d. A translation
  // of the whole volume shows up in `signed_mean`.
  IsoDispStats axis{};
  // The same shift projected onto the original's intensity gradient, i.e. the
  // distance the surface moved along its own normal. Positive `signed_mean`
  // means the surface moved towards higher intensity -- the bright side eroded.
  // The two biases catch different things and neither subsumes the other: a
  // translation cancels in `normal.signed_mean` over a closed surface, and a
  // uniform erosion cancels in `axis.signed_mean`.
  IsoDispStats normal{};
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
  // How far the isosurface moved. The metric that corresponds to "does the
  // segmentation still land in the same place".
  IsoDisplacement iso{};

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
//
// `isovalue` picks the surface for the displacement metric. The default, NaN,
// takes Otsu's threshold on the *original* -- never on the decoded volume, or
// the metric would move its own goalposts with the rate.
[[nodiscard]] Metrics compute_metrics(
    const float* orig, const float* dec, Dims dims, DType dtype,
    double isovalue = std::numeric_limits<double>::quiet_NaN());

// Otsu's threshold over a 256-bin histogram of the data range. Scroll CT is
// bimodal -- air and papyrus -- and this lands in the valley between the two
// modes, which is where a human tracing a sheet puts it.
[[nodiscard]] double otsu_threshold(const float* v, std::size_t n);

// Displacement of the `isovalue` isosurface between orig and dec, in voxels.
// NaN takes Otsu's threshold on `orig`. Allocates two histograms and nothing
// else: the volumes are gigabytes and this streams over them twice.
[[nodiscard]] IsoDisplacement isosurface_displacement(const float* orig, const float* dec,
                                                      Dims dims, double isovalue);

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
