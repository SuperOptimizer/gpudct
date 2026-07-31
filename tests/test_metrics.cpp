// Isosurface displacement (docs/QUALITY.md section 1.3).
//
// The metric claims to measure how far the surface moved, in voxels. These
// tests hold it to that claim on volumes whose displacement is known in closed
// form. A displacement metric that is merely "small when the codec is good" is
// indistinguishable from every other error metric and worth nothing; what has
// to be demonstrated is that it separates a systematic shift from jitter of the
// same magnitude, because those two are worlds apart downstream.

#include "gpudct/metrics.hpp"

#include <cmath>
#include <vector>

#include "test.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct_test;

namespace {

// A linear ramp along x. The isosurface is a plane normal to x, so the crossing
// on every x-edge is exact under linear interpolation and shifting the volume
// contents by s voxels displaces it by exactly s.
//
// The isovalue is picked so the crossing sits mid-edge: a shift large enough to
// push it onto the neighbouring edge is a topology change by construction, and
// that is a different test.
constexpr double kRampIso = 25.0;
constexpr float kRampSlope = 2.0f;

Volume x_ramp(Dims d) {
  Volume v = make(d);
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x) v.at(x, y, z) = kRampSlope * static_cast<float>(x);
  return v;
}

// A plane wave with all three frequencies nonzero. Level sets are parallel
// planes with a normal of (0.768, 0.512, 0.384), so no edge is ever tangent to
// the surface -- tangency is what makes a noise test report topology changes
// instead of displacement, and it is a property of the test volume, not of the
// metric.
Volume plane_wave(Dims d) {
  return sinusoid_volume(d, 1.0f / 32.0f, 1.0f / 48.0f, 1.0f / 64.0f, 40.0f, 128.0f);
}

}  // namespace

// An identical pair has nothing to report: no displacement, no bias, and above
// all no topology change. Anything else here is the metric inventing motion.
TEST(iso_identical_is_zero) {
  const Dims d{24, 12, 12};
  const Volume v = x_ramp(d);
  const IsoDisplacement r = isosurface_displacement(v.data.data(), v.data.data(), d, kRampIso);
  REQUIRE(r.crossings > 0);
  CHECK_EQ(r.axis.mean_abs, 0.0);
  CHECK_EQ(r.axis.max_abs, 0.0);
  CHECK_EQ(r.axis.p99, 0.0);
  CHECK_EQ(r.normal.mean_abs, 0.0);
  CHECK_EQ(r.normal.signed_mean, 0.0);
  CHECK_EQ(r.topology_frac, 0.0);
}

// The load-bearing test: a known sub-voxel shift must come back as that shift,
// with the right sign, on both the edge-aligned and the normal-projected
// figure. Shifting the contents by +s means dec(x) = orig(x - s), which on a
// ramp of slope k is a subtraction of k*s.
TEST(iso_known_subvoxel_shift) {
  const Dims d{24, 12, 12};
  for (double s : {0.25, -0.25, 0.125, -0.4}) {
    const Volume a = x_ramp(d);
    Volume b = a;
    for (float& f : b.data) f -= static_cast<float>(s) * kRampSlope;

    const IsoDisplacement r = isosurface_displacement(a.data.data(), b.data.data(), d, kRampIso);
    REQUIRE(r.crossings > 0);
    CHECK_EQ(r.topology_frac, 0.0);
    CHECK_NEAR(r.axis.signed_mean, s, 1e-5);
    CHECK_NEAR(r.axis.mean_abs, std::fabs(s), 1e-5);
    CHECK_NEAR(r.axis.max_abs, std::fabs(s), 1e-5);
    CHECK_NEAR(r.axis.p99, std::fabs(s), 1e-5);
    // The surface is normal to x, so the projection changes nothing.
    CHECK_NEAR(r.normal.signed_mean, s, 1e-5);
    CHECK_NEAR(r.normal.mean_abs, std::fabs(s), 1e-5);
  }
}

// A surface oblique to the grid. v = 3x + 4y has |grad| = 5 and unit normal
// (0.6, 0.8, 0), so moving it by s along its own normal costs 5*s in value.
// The normal-projected figure must recover s on every surviving edge; the
// edge-aligned one must not, because it reads s/0.6 on x-edges and s/0.8 on
// y-edges. This is what the projection is for -- without it the metric reports
// a number that depends on how the sheet happens to sit on the voxel grid.
TEST(iso_oblique_surface_projects_onto_normal) {
  const Dims d{20, 20, 8};
  constexpr double kS = 0.2;
  Volume a = make(d);
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x)
        a.at(x, y, z) = 3.0f * static_cast<float>(x) + 4.0f * static_cast<float>(y);
  Volume b = a;
  for (float& f : b.data) f -= static_cast<float>(kS * 5.0);

  const IsoDisplacement r = isosurface_displacement(a.data.data(), b.data.data(), d, 60.5);
  REQUIRE(r.crossings > 0);
  CHECK_NEAR(r.normal.signed_mean, kS, 1e-5);
  CHECK_NEAR(r.normal.max_abs, kS, 1e-5);
  // x-edges read 0.2/0.6 = 0.333, y-edges 0.2/0.8 = 0.25; both exceed 0.2.
  CHECK(r.axis.mean_abs > kS + 0.04);
  CHECK_NEAR(r.axis.max_abs, kS / 0.6, 1e-5);
}

// The test that proves the metric measures displacement and not error
// magnitude: zero-mean noise jitters the surface without moving it, so the bias
// must vanish while the mean absolute displacement does not. A metric that
// reported |error| in disguise would show a bias here as readily as it would
// for a codec quietly shaving half a voxel off every sheet.
TEST(iso_zero_mean_noise_has_no_bias) {
  const Dims d{64, 64, 64};
  const Volume a = plane_wave(d);
  Volume b = a;
  Rng rng(4242);
  for (float& f : b.data) f += rng.range(-0.5f, 0.5f);

  const IsoDisplacement r = isosurface_displacement(a.data.data(), b.data.data(), d, 128.0);
  REQUIRE(r.crossings > 5000);
  CHECK(r.normal.mean_abs > 0.005);
  CHECK(r.axis.mean_abs > 0.005);
  // ~20k samples of a symmetric jitter with sd well under 0.1 puts the standard
  // error below 0.001. 0.01 is a wide margin that still fails on a real bias.
  CHECK(std::fabs(r.normal.signed_mean) < 0.01);
  CHECK(std::fabs(r.axis.signed_mean) < 0.01);
  // Some topology change is expected and is not a defect: an edge whose two
  // samples straddle the isovalue by less than the noise amplitude will stop
  // straddling it about half the time. Measured at 0.20 here. What the bound
  // catches is the metric mistaking jitter for wholesale surface destruction,
  // which is what a threshold-comparison bug looks like.
  CHECK(r.topology_frac < 0.35);
}

// A systematic shift and zero-mean noise of the same magnitude are worlds apart
// downstream -- bias accumulates along a traced sheet, jitter averages out --
// so the two must not produce the same report.
TEST(iso_bias_separates_from_jitter) {
  const Dims d{64, 64, 64};
  const Volume a = plane_wave(d);
  Volume jitter = a, shifted = a;
  Rng rng(99);
  for (float& f : jitter.data) f += rng.range(-0.5f, 0.5f);
  for (float& f : shifted.data) f += 0.5f;  // a DC offset: the surface really moves

  const IsoDisplacement rj = isosurface_displacement(a.data.data(), jitter.data.data(), d, 128.0);
  const IsoDisplacement rs =
      isosurface_displacement(a.data.data(), shifted.data.data(), d, 128.0);
  CHECK(rj.normal.mean_abs > 0.0);
  CHECK(std::fabs(rs.normal.signed_mean) > 10.0 * std::fabs(rj.normal.signed_mean));
  // Raising every sample moves the level set towards where the volume was
  // darker, which is a negative displacement by this metric's convention.
  CHECK(rs.normal.signed_mean < 0.0);
}

// A surface that vanishes is worse than one that moves and has to be counted
// separately rather than dropped. Flattening half the volume below the isovalue
// destroys every crossing there.
TEST(iso_topology_change_is_counted) {
  const Dims d{32, 32, 32};
  const Volume a = sinusoid_volume(d, 1.0f / 16.0f, 0.0f, 0.0f, 40.0f, 128.0f);
  Volume b = a;
  for (std::uint32_t z = d.z / 2; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x) b.at(x, y, z) = 0.0f;

  const IsoDisplacement r = isosurface_displacement(a.data.data(), b.data.data(), d, 128.0);
  REQUIRE(r.crossings > 0);
  CHECK(r.topology_frac > 0.3);
  CHECK(r.topology_frac < 0.8);
  // Where the volume survived it is untouched, so nothing moved.
  CHECK_EQ(r.axis.max_abs, 0.0);
}

// Otsu has to land in the valley of a bimodal volume, since that is the whole
// reason it is the default: the isovalue must be where someone tracing a sheet
// would put it, not at the mean of a distribution with no mass there.
TEST(iso_otsu_finds_the_valley) {
  const Dims d{32, 32, 32};
  Volume v = make(d);
  Rng rng(3);
  for (std::size_t i = 0; i < v.data.size(); ++i)
    v.data[i] = (i % 3 == 0) ? rng.range(190.0f, 210.0f) : rng.range(20.0f, 40.0f);
  const double t = otsu_threshold(v.data.data(), v.data.size());
  CHECK(t > 40.0);
  CHECK(t < 190.0);

  // And the default path through compute_metrics must use it rather than
  // leaving the isovalue at zero.
  const Metrics m = compute_metrics(v.data.data(), v.data.data(), d, DType::u8);
  CHECK_NEAR(m.iso.isovalue, t, 1e-9);
  CHECK_EQ(m.iso.axis.max_abs, 0.0);
}

// An explicit isovalue overrides Otsu, and the report says which was used.
TEST(iso_isovalue_override) {
  const Dims d{24, 12, 12};
  const Volume v = x_ramp(d);
  const Metrics m = compute_metrics(v.data.data(), v.data.data(), d, DType::u8, kRampIso);
  CHECK_NEAR(m.iso.isovalue, kRampIso, 1e-12);
  CHECK(m.iso.crossings > 0);
}

// A volume with no surface at the chosen isovalue must report nothing rather
// than divide by zero.
TEST(iso_no_crossings_is_safe) {
  const Dims d{16, 16, 16};
  const Volume v = constant_volume(d, 100.0f);
  const IsoDisplacement r = isosurface_displacement(v.data.data(), v.data.data(), d, 200.0);
  CHECK_EQ(r.crossings, static_cast<std::uint64_t>(0));
  CHECK_EQ(r.crossing_frac, 0.0);
  CHECK_EQ(r.topology_frac, 0.0);
  CHECK_EQ(r.axis.mean_abs, 0.0);
  CHECK(r.edges > 0);
}

// Volumes too small for the normal stencil must be handled, not indexed out of
// bounds: two voxels on an axis leaves no interior edge at all.
TEST(iso_tiny_volume_is_safe) {
  for (Dims d : {Dims{2, 2, 2}, Dims{3, 3, 3}, Dims{1, 8, 8}}) {
    Volume v = make(d);
    for (std::size_t i = 0; i < v.data.size(); ++i) v.data[i] = static_cast<float>(i % 7);
    const IsoDisplacement r = isosurface_displacement(v.data.data(), v.data.data(), d, 3.5);
    CHECK_EQ(r.crossings, static_cast<std::uint64_t>(0));
    CHECK_EQ(r.edges, static_cast<std::uint64_t>(0));
  }
}

TEST_MAIN()
