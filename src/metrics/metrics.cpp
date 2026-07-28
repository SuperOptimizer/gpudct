#include "gpudct/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>

#include "core/transform.hpp"

namespace gpudct {
namespace {

using detail::kD;

[[nodiscard]] inline std::size_t idx_of(Dims d, std::uint32_t x, std::uint32_t y,
                                        std::uint32_t z) {
  return (static_cast<std::size_t>(z) * d.y + y) * d.x + x;
}

// Clamped sampling, so every neighbourhood operator below has a defined answer
// at the boundary without a special case at each call site.
[[nodiscard]] inline float sample(const float* v, Dims d, std::int64_t x, std::int64_t y,
                                  std::int64_t z) {
  x = std::clamp<std::int64_t>(x, 0, d.x - 1);
  y = std::clamp<std::int64_t>(y, 0, d.y - 1);
  z = std::clamp<std::int64_t>(z, 0, d.z - 1);
  return v[idx_of(d, static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y),
                  static_cast<std::uint32_t>(z))];
}

// --------------------------------------------------------------------------
// Exact percentiles over |a - b|.
//
// Pass 1 histograms the errors; pass 2 revisits only the bins that contain a
// requested rank and sorts their exact contents. The result is a value that
// really occurs in the data, at the right rank -- not an interpolation and not
// a sketch. Memory is the histogram plus one bin's worth of values.
// --------------------------------------------------------------------------
constexpr std::size_t kBins = 1u << 20;

[[nodiscard]] inline std::size_t bin_of(double err, double maxerr) {
  if (!(maxerr > 0.0)) return 0;
  const double t = err / maxerr;
  const std::size_t b = static_cast<std::size_t>(t * static_cast<double>(kBins - 1) + 0.5);
  return std::min(b, kBins - 1);
}

}  // namespace

Percentiles exact_abs_percentiles(const float* a, const float* b, std::size_t n) {
  Percentiles p;
  if (n == 0) return p;

  double maxerr = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    maxerr = std::max(maxerr, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  p.max = maxerr;
  if (maxerr == 0.0) return p;

  std::vector<std::uint64_t> hist(kBins, 0);
  for (std::size_t i = 0; i < n; ++i)
    ++hist[bin_of(std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])), maxerr)];

  struct Target {
    double q;
    double* out;
  };
  Target targets[] = {{0.50, &p.p50},   {0.90, &p.p90},     {0.95, &p.p95},
                      {0.99, &p.p99},   {0.999, &p.p999},   {0.9999, &p.p9999}};

  // Locate each target's bin and its rank within that bin.
  struct Resolved {
    std::size_t bin;
    std::uint64_t rank_in_bin;
    double* out;
  };
  std::vector<Resolved> resolved;
  for (const Target& t : targets) {
    // Rank convention: the smallest value v such that at least q*n samples are
    // <= v. For n samples this is the (ceil(q*n))-th smallest, 1-based.
    std::uint64_t want = static_cast<std::uint64_t>(
        std::ceil(t.q * static_cast<double>(n)));
    if (want == 0) want = 1;
    if (want > n) want = n;

    std::uint64_t acc = 0;
    for (std::size_t bi = 0; bi < kBins; ++bi) {
      if (acc + hist[bi] >= want) {
        resolved.push_back({bi, want - acc, t.out});
        break;
      }
      acc += hist[bi];
    }
  }
  hist.clear();
  hist.shrink_to_fit();

  // Pass 2: exact values, only for the bins that matter.
  std::vector<std::size_t> bins;
  for (const Resolved& r : resolved) bins.push_back(r.bin);
  std::sort(bins.begin(), bins.end());
  bins.erase(std::unique(bins.begin(), bins.end()), bins.end());

  for (std::size_t bi : bins) {
    std::vector<double> vals;
    for (std::size_t i = 0; i < n; ++i) {
      const double e = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
      if (bin_of(e, maxerr) == bi) vals.push_back(e);
    }
    std::sort(vals.begin(), vals.end());
    for (const Resolved& r : resolved)
      if (r.bin == bi) {
        const std::size_t k = static_cast<std::size_t>(r.rank_in_bin) - 1;
        *r.out = vals.empty() ? 0.0 : vals[std::min(k, vals.size() - 1)];
      }
  }
  return p;
}

namespace {

// --------------------------------------------------------------------------
// Separable 3D Gaussian, radius 5 / sigma 1.5 -- the 11^3 window SSIM is
// conventionally defined with, done as three 1D passes rather than a 1331-tap
// convolution.
// --------------------------------------------------------------------------
struct Gauss {
  static constexpr int R = 5;
  float k[2 * R + 1];
  Gauss() {
    float sum = 0.0f;
    for (int i = -R; i <= R; ++i) {
      const float x = static_cast<float>(i);
      k[i + R] = std::exp(-x * x / (2.0f * 1.5f * 1.5f));
      sum += k[i + R];
    }
    for (float& v : k) v /= sum;
  }
};

void blur(const std::vector<float>& in, std::vector<float>& out, Dims d) {
  static const Gauss g;
  std::vector<float> tmp(in.size());
  // x
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x) {
        float acc = 0.0f;
        for (int i = -Gauss::R; i <= Gauss::R; ++i)
          acc += g.k[i + Gauss::R] * sample(in.data(), d, static_cast<std::int64_t>(x) + i, y, z);
        tmp[idx_of(d, x, y, z)] = acc;
      }
  // y
  std::vector<float> tmp2(in.size());
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x) {
        float acc = 0.0f;
        for (int i = -Gauss::R; i <= Gauss::R; ++i)
          acc += g.k[i + Gauss::R] * sample(tmp.data(), d, x, static_cast<std::int64_t>(y) + i, z);
        tmp2[idx_of(d, x, y, z)] = acc;
      }
  // z
  out.assign(in.size(), 0.0f);
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x) {
        float acc = 0.0f;
        for (int i = -Gauss::R; i <= Gauss::R; ++i)
          acc += g.k[i + Gauss::R] * sample(tmp2.data(), d, x, y, static_cast<std::int64_t>(z) + i);
        out[idx_of(d, x, y, z)] = acc;
      }
}

// True volumetric SSIM. Slice-averaged 2D SSIM is blind to z-axis artifacts,
// which is exactly the failure mode a separable 3D transform can have.
void compute_ssim(const float* a, const float* b, Dims d, double dyn_range, double& mean,
                  double& p01) {
  const std::size_t n = d.voxels();
  std::vector<float> x(a, a + n), y(b, b + n), xx(n), yy(n), xy(n);
  for (std::size_t i = 0; i < n; ++i) {
    xx[i] = x[i] * x[i];
    yy[i] = y[i] * y[i];
    xy[i] = x[i] * y[i];
  }
  std::vector<float> mx, my, mxx, myy, mxy;
  blur(x, mx, d);
  blur(y, my, d);
  blur(xx, mxx, d);
  blur(yy, myy, d);
  blur(xy, mxy, d);

  const double c1 = std::pow(0.01 * dyn_range, 2.0);
  const double c2 = std::pow(0.03 * dyn_range, 2.0);

  std::vector<float> map(n);
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double ux = mx[i], uy = my[i];
    const double vx = std::max(0.0, static_cast<double>(mxx[i]) - ux * ux);
    const double vy = std::max(0.0, static_cast<double>(myy[i]) - uy * uy);
    const double vxy = static_cast<double>(mxy[i]) - ux * uy;
    const double s = ((2 * ux * uy + c1) * (2 * vxy + c2)) /
                     ((ux * ux + uy * uy + c1) * (vx + vy + c2));
    map[i] = static_cast<float>(s);
    sum += s;
  }
  mean = sum / static_cast<double>(n);

  // The 1st percentile of the map says more than its mean: it is the regions
  // that were actually damaged.
  std::vector<float> sorted = map;
  const std::size_t k = static_cast<std::size_t>(0.01 * static_cast<double>(n));
  std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(k), sorted.end());
  p01 = sorted[k];
}

// 3D Sobel gradient magnitude: derivative along one axis, [1 2 1] smoothing on
// the other two.
void gradient_magnitude(const float* v, Dims d, std::vector<float>& out) {
  out.assign(d.voxels(), 0.0f);
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x) {
        double g[3] = {0, 0, 0};
        for (int dz = -1; dz <= 1; ++dz)
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              const double s = sample(v, d, static_cast<std::int64_t>(x) + dx,
                                      static_cast<std::int64_t>(y) + dy,
                                      static_cast<std::int64_t>(z) + dz);
              const double wx = (dy == 0 ? 2.0 : 1.0) * (dz == 0 ? 2.0 : 1.0);
              const double wy = (dx == 0 ? 2.0 : 1.0) * (dz == 0 ? 2.0 : 1.0);
              const double wz = (dx == 0 ? 2.0 : 1.0) * (dy == 0 ? 2.0 : 1.0);
              g[0] += dx * wx * s;
              g[1] += dy * wy * s;
              g[2] += dz * wz * s;
            }
        out[idx_of(d, x, y, z)] =
            static_cast<float>(std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]));
      }
}

// Mean |delta| across chunk faces divided by mean |delta| in the interior.
//
// Taken alone this is not a blockiness measure: a volume whose own texture has
// structure at a period commensurate with the chunk size scores far from 1.0
// before any codec touches it. The caller divides the decoded figure by the
// original's, so what gets reported is the blockiness the codec *added*.
double face_to_interior_ratio(const float* v, Dims d) {
  double face_sum = 0, face_n = 0, inner_sum = 0, inner_n = 0;
  auto axis = [&](int ax) {
    const std::uint32_t lim[3] = {d.x, d.y, d.z};
    if (lim[ax] < 2) return;
    for (std::uint32_t z = 0; z < d.z; ++z)
      for (std::uint32_t y = 0; y < d.y; ++y)
        for (std::uint32_t x = 0; x < d.x; ++x) {
          std::uint32_t c[3] = {x, y, z};
          if (c[ax] == 0) continue;
          std::uint32_t p[3] = {x, y, z};
          --p[ax];
          const double jump = std::fabs(static_cast<double>(v[idx_of(d, c[0], c[1], c[2])]) -
                                        v[idx_of(d, p[0], p[1], p[2])]);
          if (c[ax] % static_cast<std::uint32_t>(kD) == 0) {
            face_sum += jump;
            ++face_n;
          } else {
            inner_sum += jump;
            ++inner_n;
          }
        }
  };
  axis(0);
  axis(1);
  axis(2);
  if (face_n == 0 || inner_n == 0 || inner_sum == 0) return 1.0;
  return (face_sum / face_n) / (inner_sum / inner_n);
}

double laplacian_energy(const float* v, Dims d) {
  double e = 0;
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x) {
        const double c = v[idx_of(d, x, y, z)];
        const double l =
            sample(v, d, static_cast<std::int64_t>(x) - 1, y, z) +
            sample(v, d, static_cast<std::int64_t>(x) + 1, y, z) +
            sample(v, d, x, static_cast<std::int64_t>(y) - 1, z) +
            sample(v, d, x, static_cast<std::int64_t>(y) + 1, z) +
            sample(v, d, x, y, static_cast<std::int64_t>(z) - 1) +
            sample(v, d, x, y, static_cast<std::int64_t>(z) + 1) - 6.0 * c;
        e += l * l;
      }
  return e;
}

// Energy retained per radial frequency band, measured with the codec's own
// transform on aligned chunks. This is the metric that catches a quant matrix
// erasing fibre texture while PSNR still looks respectable.
void band_energy(const float* a, const float* b, Dims d, std::array<double, 4>& ratio) {
  std::array<double, 4> ea{}, eb{};
  std::vector<float> ca(kChunkVox), cb(kChunkVox), fa(kChunkVox), fb(kChunkVox);

  for (std::uint32_t z = 0; z + kD <= d.z; z += kD)
    for (std::uint32_t y = 0; y + kD <= d.y; y += kD)
      for (std::uint32_t x = 0; x + kD <= d.x; x += kD) {
        for (int k = 0; k < kD; ++k)
          for (int j = 0; j < kD; ++j)
            for (int i = 0; i < kD; ++i) {
              const std::size_t s = idx_of(d, x + static_cast<std::uint32_t>(i),
                                           y + static_cast<std::uint32_t>(j),
                                           z + static_cast<std::uint32_t>(k));
              ca[static_cast<std::size_t>((k * kD + j) * kD + i)] = a[s];
              cb[static_cast<std::size_t>((k * kD + j) * kD + i)] = b[s];
            }
        detail::forward_chunk(ca.data(), fa.data());
        detail::forward_chunk(cb.data(), fb.data());
        for (int i = 1; i < kChunkVox; ++i) {  // skip DC: it is not a "band"
          const int band = detail::band_of_subblock_index(detail::subblock_of_coeff(i));
          ea[static_cast<std::size_t>(band)] +=
              static_cast<double>(fa[static_cast<std::size_t>(i)]) * fa[static_cast<std::size_t>(i)];
          eb[static_cast<std::size_t>(band)] +=
              static_cast<double>(fb[static_cast<std::size_t>(i)]) * fb[static_cast<std::size_t>(i)];
        }
      }
  for (std::size_t i = 0; i < 4; ++i) ratio[i] = (ea[i] > 0) ? eb[i] / ea[i] : 1.0;
}

// Per-8^3-block standard deviation error. Flat regions being perfect while
// textured regions turn to mush is invisible in a whole-volume average.
void local_std_error(const float* a, const float* b, Dims d, double& p50, double& p99) {
  constexpr std::uint32_t B = 8;
  std::vector<double> errs;
  for (std::uint32_t z = 0; z + B <= d.z; z += B)
    for (std::uint32_t y = 0; y + B <= d.y; y += B)
      for (std::uint32_t x = 0; x + B <= d.x; x += B) {
        double sa = 0, sb = 0, qa = 0, qb = 0;
        for (std::uint32_t k = 0; k < B; ++k)
          for (std::uint32_t j = 0; j < B; ++j)
            for (std::uint32_t i = 0; i < B; ++i) {
              const std::size_t s = idx_of(d, x + i, y + j, z + k);
              sa += a[s];
              sb += b[s];
              qa += static_cast<double>(a[s]) * a[s];
              qb += static_cast<double>(b[s]) * b[s];
            }
        const double n = B * B * B;
        const double va = std::max(0.0, qa / n - (sa / n) * (sa / n));
        const double vb = std::max(0.0, qb / n - (sb / n) * (sb / n));
        errs.push_back(std::fabs(std::sqrt(va) - std::sqrt(vb)));
      }
  if (errs.empty()) return;
  std::sort(errs.begin(), errs.end());
  auto pick = [&](double q) {
    const std::size_t k = std::min(errs.size() - 1,
                                   static_cast<std::size_t>(q * static_cast<double>(errs.size())));
    return errs[k];
  };
  p50 = pick(0.50);
  p99 = pick(0.99);
}

}  // namespace

Metrics compute_metrics(const float* orig, const float* dec, Dims dims, DType dtype) {
  Metrics m;
  const std::size_t n = dims.voxels();
  m.voxels = n;
  if (n == 0) return m;

  double lo = orig[0], hi = orig[0], sum_abs = 0, sum_sq = 0, sum_signed = 0;
  for (std::size_t i = 0; i < n; ++i) {
    lo = std::min<double>(lo, orig[i]);
    hi = std::max<double>(hi, orig[i]);
    const double e = static_cast<double>(dec[i]) - static_cast<double>(orig[i]);
    sum_signed += e;
    sum_abs += std::fabs(e);
    sum_sq += e * e;
  }
  m.data_min = lo;
  m.data_max = hi;
  m.mae = sum_abs / static_cast<double>(n);
  m.mse = sum_sq / static_cast<double>(n);
  m.rmse = std::sqrt(m.mse);
  m.bias = sum_signed / static_cast<double>(n);
  m.abs_err = exact_abs_percentiles(orig, dec, n);

  const double dtype_peak = static_cast<double>(dtype_max(dtype)) - dtype_min(dtype);
  const double range_peak = hi - lo;
  m.psnr_dtype = (m.rmse > 0 && std::isfinite(dtype_peak))
                     ? 20.0 * std::log10(dtype_peak / m.rmse)
                     : std::numeric_limits<double>::infinity();
  m.psnr_range = (m.rmse > 0 && range_peak > 0) ? 20.0 * std::log10(range_peak / m.rmse)
                                                : std::numeric_limits<double>::infinity();

  const double dyn = (range_peak > 0) ? range_peak : 1.0;
  compute_ssim(orig, dec, dims, dyn, m.ssim, m.ssim_p01);

  std::vector<float> ga, gb;
  gradient_magnitude(orig, dims, ga);
  gradient_magnitude(dec, dims, gb);
  {
    double s = 0;
    for (std::size_t i = 0; i < n; ++i) s += std::fabs(static_cast<double>(ga[i]) - gb[i]);
    m.gradient_mae = s / static_cast<double>(n);
    const Percentiles gp = exact_abs_percentiles(ga.data(), gb.data(), n);
    m.gradient_p99 = gp.p99;
  }

  {
    const double fo = face_to_interior_ratio(orig, dims);
    const double fd = face_to_interior_ratio(dec, dims);
    m.blockiness = (fo > 0) ? fd / fo : 1.0;
  }
  {
    const double la = laplacian_energy(orig, dims);
    const double lb = laplacian_energy(dec, dims);
    m.laplacian_ratio = (la > 0) ? lb / la : 1.0;
  }
  band_energy(orig, dec, dims, m.band_energy_ratio);
  local_std_error(orig, dec, dims, m.local_std_err_p50, m.local_std_err_p99);

  // Error autocorrelation at lag 1 along each axis. White noise is what a good
  // codec leaves behind; structure here means we removed signal.
  {
    const std::uint32_t lim[3] = {dims.x, dims.y, dims.z};
    for (int ax = 0; ax < 3; ++ax) {
      double s0 = 0, s1 = 0, s00 = 0, s11 = 0, s01 = 0, cnt = 0;
      if (lim[ax] < 2) continue;
      for (std::uint32_t z = 0; z < dims.z; ++z)
        for (std::uint32_t y = 0; y < dims.y; ++y)
          for (std::uint32_t x = 0; x < dims.x; ++x) {
            std::uint32_t c[3] = {x, y, z};
            if (c[ax] + 1 >= lim[ax]) continue;
            std::uint32_t q[3] = {x, y, z};
            ++q[ax];
            const double e0 = static_cast<double>(dec[idx_of(dims, c[0], c[1], c[2])]) -
                              orig[idx_of(dims, c[0], c[1], c[2])];
            const double e1 = static_cast<double>(dec[idx_of(dims, q[0], q[1], q[2])]) -
                              orig[idx_of(dims, q[0], q[1], q[2])];
            s0 += e0; s1 += e1; s00 += e0 * e0; s11 += e1 * e1; s01 += e0 * e1; ++cnt;
          }
      if (cnt < 2) continue;
      const double cov = s01 / cnt - (s0 / cnt) * (s1 / cnt);
      const double v0 = s00 / cnt - (s0 / cnt) * (s0 / cnt);
      const double v1 = s11 / cnt - (s1 / cnt) * (s1 / cnt);
      m.error_autocorr[static_cast<std::size_t>(ax)] =
          (v0 > 0 && v1 > 0) ? cov / std::sqrt(v0 * v1) : 0.0;
    }
  }

  // Histogram earth-mover distance over 256 bins of the data range.
  {
    constexpr int NB = 256;
    std::vector<double> ha(NB, 0), hb(NB, 0);
    const double span = (range_peak > 0) ? range_peak : 1.0;
    for (std::size_t i = 0; i < n; ++i) {
      const int bi = std::clamp(static_cast<int>((orig[i] - lo) / span * (NB - 1)), 0, NB - 1);
      const int bj = std::clamp(static_cast<int>((dec[i] - lo) / span * (NB - 1)), 0, NB - 1);
      ha[static_cast<std::size_t>(bi)] += 1.0;
      hb[static_cast<std::size_t>(bj)] += 1.0;
    }
    double ca = 0, cb = 0, emd = 0;
    for (int i = 0; i < NB; ++i) {
      ca += ha[static_cast<std::size_t>(i)] / static_cast<double>(n);
      cb += hb[static_cast<std::size_t>(i)] / static_cast<double>(n);
      emd += std::fabs(ca - cb);
    }
    m.hist_emd = emd / NB;
  }

  // Anisotropy: worst slice normal to each axis, and per-axis gradient error.
  {
    const std::uint32_t lim[3] = {dims.x, dims.y, dims.z};
    for (int ax = 0; ax < 3; ++ax) {
      std::vector<double> slice_sum(lim[static_cast<std::size_t>(ax)], 0.0);
      std::vector<double> slice_n(lim[static_cast<std::size_t>(ax)], 0.0);
      for (std::uint32_t z = 0; z < dims.z; ++z)
        for (std::uint32_t y = 0; y < dims.y; ++y)
          for (std::uint32_t x = 0; x < dims.x; ++x) {
            const std::uint32_t c[3] = {x, y, z};
            const std::size_t s = idx_of(dims, x, y, z);
            slice_sum[c[static_cast<std::size_t>(ax)]] +=
                std::fabs(static_cast<double>(dec[s]) - orig[s]);
            slice_n[c[static_cast<std::size_t>(ax)]] += 1.0;
          }
      double worst = 0;
      for (std::size_t i = 0; i < slice_sum.size(); ++i)
        if (slice_n[i] > 0) worst = std::max(worst, slice_sum[i] / slice_n[i]);
      m.worst_slice_mae[static_cast<std::size_t>(ax)] = worst;
    }
  }

  // Conditioned metrics: error by intensity quartile, and error where the
  // original has the steepest gradients. The latter runs 10-50x the whole-volume
  // average on real data, and it is the number that predicts whether a
  // segmentation still lands in the same place.
  {
    std::array<double, 4> sum{}, cnt{};
    const double span = (range_peak > 0) ? range_peak : 1.0;
    for (std::size_t i = 0; i < n; ++i) {
      const int q = std::clamp(static_cast<int>((orig[i] - lo) / span * 4.0), 0, 3);
      sum[static_cast<std::size_t>(q)] +=
          std::fabs(static_cast<double>(dec[i]) - orig[i]);
      cnt[static_cast<std::size_t>(q)] += 1.0;
    }
    for (std::size_t q = 0; q < 4; ++q)
      m.mae_by_intensity[q] = (cnt[q] > 0) ? sum[q] / cnt[q] : 0.0;

    std::vector<float> gsorted = ga;
    std::sort(gsorted.begin(), gsorted.end());
    const float thresh = gsorted[static_cast<std::size_t>(0.9 * static_cast<double>(n))];
    double s = 0, c = 0;
    for (std::size_t i = 0; i < n; ++i)
      if (ga[i] >= thresh) {
        s += std::fabs(static_cast<double>(dec[i]) - orig[i]);
        c += 1.0;
      }
    m.mae_high_gradient = (c > 0) ? s / c : 0.0;
  }

  // Per-axis gradient error, from central differences. A separable 3D transform
  // can degrade one axis and not the others; the isotropic Sobel figure above
  // averages that away.
  {
    const std::uint32_t lim[3] = {dims.x, dims.y, dims.z};
    for (int ax = 0; ax < 3; ++ax) {
      if (lim[static_cast<std::size_t>(ax)] < 3) continue;
      double s = 0, c = 0;
      for (std::uint32_t z = 0; z < dims.z; ++z)
        for (std::uint32_t y = 0; y < dims.y; ++y)
          for (std::uint32_t x = 0; x < dims.x; ++x) {
            std::int64_t p[3] = {x, y, z}, q[3] = {x, y, z};
            --p[ax];
            ++q[ax];
            const double da = sample(orig, dims, q[0], q[1], q[2]) -
                              sample(orig, dims, p[0], p[1], p[2]);
            const double db = sample(dec, dims, q[0], q[1], q[2]) -
                              sample(dec, dims, p[0], p[1], p[2]);
            s += std::fabs(da - db);
            c += 1.0;
          }
      m.axis_gradient_mae[static_cast<std::size_t>(ax)] = (c > 0) ? s / (2.0 * c) : 0.0;
    }
  }

  return m;
}

std::vector<BrickScore> worst_bricks(const float* orig, const float* dec, Dims dims,
                                     std::size_t count) {
  const Dims g = brick_grid(dims);
  std::vector<BrickScore> scores;
  std::vector<float> a, b;
  a.reserve(static_cast<std::size_t>(kBrickDim) * kBrickDim * kBrickDim);
  b.reserve(a.capacity());

  for (std::uint32_t bz = 0; bz < g.z; ++bz)
    for (std::uint32_t by = 0; by < g.y; ++by)
      for (std::uint32_t bx = 0; bx < g.x; ++bx) {
        a.clear();
        b.clear();
        for (std::uint32_t z = bz * kBrickDim; z < std::min(dims.z, (bz + 1) * kBrickDim); ++z)
          for (std::uint32_t y = by * kBrickDim; y < std::min(dims.y, (by + 1) * kBrickDim); ++y)
            for (std::uint32_t x = bx * kBrickDim; x < std::min(dims.x, (bx + 1) * kBrickDim);
                 ++x) {
              const std::size_t i = idx_of(dims, x, y, z);
              a.push_back(orig[i]);
              b.push_back(dec[i]);
            }
        if (a.empty()) continue;
        double sq = 0, mx = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
          const double e = std::fabs(static_cast<double>(a[i]) - b[i]);
          sq += e * e;
          mx = std::max(mx, e);
        }
        BrickScore s;
        s.bx = bx;
        s.by = by;
        s.bz = bz;
        s.rmse = std::sqrt(sq / static_cast<double>(a.size()));
        s.max_abs = mx;
        s.p99 = exact_abs_percentiles(a.data(), b.data(), a.size()).p99;
        scores.push_back(s);
      }

  std::sort(scores.begin(), scores.end(),
            [](const BrickScore& x, const BrickScore& y) { return x.rmse > y.rmse; });
  if (scores.size() > count) scores.resize(count);
  return scores;
}

std::string format_report(const Metrics& m, bool verbose) {
  std::string s;
  char buf[512];
  auto line = [&](const char* fmt, auto... args) {
    std::snprintf(buf, sizeof(buf), fmt, args...);
    s += buf;
  };

  line("error\n");
  line("  mae                %12.6f\n", m.mae);
  line("  rmse               %12.6f\n", m.rmse);
  line("  bias               %12.6f\n", m.bias);
  line("  p50 / p90 / p95    %12.4f %12.4f %12.4f\n", m.abs_err.p50, m.abs_err.p90,
       m.abs_err.p95);
  line("  p99 / p99.9        %12.4f %12.4f\n", m.abs_err.p99, m.abs_err.p999);
  line("  p99.99 / max       %12.4f %12.4f\n", m.abs_err.p9999, m.abs_err.max);
  line("fidelity\n");
  line("  psnr (dtype range) %12.3f dB\n", m.psnr_dtype);
  line("  psnr (data range)  %12.3f dB\n", m.psnr_range);
  line("  ssim 3d            %12.6f  (p01 %.6f)\n", m.ssim, m.ssim_p01);
  line("structure\n");
  line("  gradient mae/p99   %12.6f %12.6f\n", m.gradient_mae, m.gradient_p99);
  line("  blockiness         %12.4f  (1.0 = no seams)\n", m.blockiness);
  line("  laplacian ratio    %12.4f  (1.0 = detail preserved)\n", m.laplacian_ratio);
  line("  band energy ratio  %8.4f %8.4f %8.4f %8.4f  (low -> high)\n",
       m.band_energy_ratio[0], m.band_energy_ratio[1], m.band_energy_ratio[2],
       m.band_energy_ratio[3]);
  line("  local std err      p50 %.5f  p99 %.5f\n", m.local_std_err_p50, m.local_std_err_p99);
  line("  hist emd           %12.6f\n", m.hist_emd);

  if (verbose) {
    line("anisotropy\n");
    line("  worst slice mae    x %.5f  y %.5f  z %.5f\n", m.worst_slice_mae[0],
         m.worst_slice_mae[1], m.worst_slice_mae[2]);
    line("  axis gradient mae  x %.5f  y %.5f  z %.5f\n", m.axis_gradient_mae[0],
         m.axis_gradient_mae[1], m.axis_gradient_mae[2]);
    line("  error autocorr     x %+.4f  y %+.4f  z %+.4f  (0 = white)\n",
         m.error_autocorr[0], m.error_autocorr[1], m.error_autocorr[2]);
    line("conditioned\n");
    line("  mae by intensity   %.5f %.5f %.5f %.5f  (low -> high)\n", m.mae_by_intensity[0],
         m.mae_by_intensity[1], m.mae_by_intensity[2], m.mae_by_intensity[3]);
    line("  mae, top decile of gradient %.5f\n", m.mae_high_gradient);
  }
  if (m.ratio > 0) {
    line("rate\n");
    line("  ratio              %12.2fx\n", m.ratio);
    line("  bits per voxel     %12.4f\n", m.bits_per_voxel);
  }
  return s;
}

}  // namespace gpudct
