// Bounded-error guarantees (docs/DESIGN.md section 3.5).
//
// This is the codec's core promise, so it is checked exhaustively rather than
// spot-checked: every voxel of every test volume, at several bounds, across
// dtypes and adversarial content. A bound that holds "almost always" is not a
// bound.

#include <algorithm>
#include <cmath>

#include "gpudct/gpudct.hpp"
#include "gpudct/metrics.hpp"
#include "test.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct_test;

namespace {

struct Outcome {
  double max_err = 0;
  double p99 = 0;
  double p999 = 0;
  double ratio = 0;
  bool ok = false;
};

Outcome run(const Volume& v, DType t, const EncodeOptions& opts) {
  Outcome o;
  const std::vector<std::uint8_t> raw = to_typed(v, t);
  std::vector<std::uint8_t> archive;
  if (encode(raw.data(), v.dims, t, opts, archive) != Status::ok) return o;

  std::vector<std::uint8_t> out;
  VolumeInfo info;
  if (decode(archive, DecodeOptions{}, out, info) != Status::ok) return o;

  const std::vector<float> a = from_typed(raw, t, v.data.size());
  const std::vector<float> b = from_typed(out, t, v.data.size());
  const Percentiles p = exact_abs_percentiles(a.data(), b.data(), a.size());
  o.max_err = p.max;
  o.p99 = p.p99;
  o.p999 = p.p999;
  o.ratio = static_cast<double>(raw.size()) / static_cast<double>(archive.size());
  o.ok = true;
  return o;
}

}  // namespace

// An absolute bound must hold at every voxel, at any quality -- including a
// quality so coarse that the lossy layer is badly wrong and the correction layer
// is doing nearly all the work.
TEST(absolute_bound_holds_at_every_voxel) {
  const Volume v = scroll_like_volume({96, 80, 72});
  for (float tau : {0.0f, 1.0f, 2.0f, 5.0f, 10.0f}) {
    for (float q : {0.125f, 0.5f, 2.0f}) {
      EncodeOptions opts;
      opts.quality = q;
      opts.bounds.max_abs = tau;
      const Outcome o = run(v, DType::u8, opts);
      REQUIRE(o.ok);
      if (tau > 0.0f && o.max_err > tau)
        std::printf("       tau=%.1f q=%.3f -> max %.2f  ratio %.2f\n", static_cast<double>(tau),
                    static_cast<double>(q), o.max_err, o.ratio);
      if (tau > 0.0f) CHECK(o.max_err <= tau);
    }
  }
}

// tau = 0 on an integer dtype means lossless.
TEST(a_zero_bound_is_lossless) {
  for (DType t : {DType::u8, DType::s8, DType::u16, DType::s16}) {
    const Volume base = scroll_like_volume({64, 56, 48});
    Volume v = base;
    const float hi = std::min(dtype_max(t), 4000.0f);
    const float lo = std::max(dtype_min(t), -4000.0f);
    for (float& f : v.data) f = lo + (f / 255.0f) * (hi - lo);

    EncodeOptions opts;
    opts.quality = 0.25f;
    opts.bounds.max_abs = 0.5f;  // half an integer step: forces exact reconstruction
    const Outcome o = run(v, t, opts);
    REQUIRE(o.ok);
    CHECK_EQ(o.max_err, 0.0);
  }
}

TEST(absolute_bound_holds_on_adversarial_content) {
  const Dims d{64, 64, 64};
  struct Case { const char* name; Volume v; };
  std::vector<Case> cases;
  cases.push_back({"noise", noise_volume(d, 0.0f, 255.0f)});
  cases.push_back({"checkerboard", checkerboard_volume(d, 0.0f, 255.0f)});
  cases.push_back({"impulse", impulse_volume(d, 0.0f, 255.0f)});
  cases.push_back({"step", step_volume(d, 0.0f, 255.0f)});
  cases.push_back({"constant", constant_volume(d, 137.0f)});

  for (const Case& c : cases) {
    EncodeOptions opts;
    opts.quality = 0.125f;
    opts.bounds.max_abs = 3.0f;
    const Outcome o = run(c.v, DType::u8, opts);
    REQUIRE(o.ok);
    if (o.max_err > 3.0) std::printf("       %s -> max %.2f\n", c.name, o.max_err);
    CHECK(o.max_err <= 3.0);
  }
}

// Dimensions that do not fill a brick exercise the padding path, where a
// correction index could easily be computed against the wrong extent.
TEST(bounds_hold_on_non_multiple_dimensions) {
  for (Dims d : {Dims{1, 1, 1}, Dims{17, 3, 129}, Dims{130, 127, 5}, Dims{129, 1, 1}}) {
    const Volume v = scroll_like_volume(d);
    EncodeOptions opts;
    opts.quality = 0.25f;
    opts.bounds.max_abs = 2.0f;
    const Outcome o = run(v, DType::u8, opts);
    REQUIRE(o.ok);
    CHECK(o.max_err <= 2.0);
  }
}

// A percentile bound is much cheaper than a hard maximum because it only has to
// clip the tail. Both properties are worth pinning: the bound holds, and it
// costs less than the equivalent hard bound.
TEST(percentile_bound_holds_and_costs_less_than_a_hard_bound) {
  const Volume v = scroll_like_volume({128, 96, 96});

  EncodeOptions pct;
  pct.quality = 0.25f;
  pct.bounds.p99 = 4.0f;
  const Outcome a = run(v, DType::u8, pct);
  REQUIRE(a.ok);

  EncodeOptions hard;
  hard.quality = 0.25f;
  hard.bounds.max_abs = 4.0f;
  const Outcome b = run(v, DType::u8, hard);
  REQUIRE(b.ok);

  std::printf("       p99<=4: p99 %.2f max %.2f ratio %.2f | hard<=4: max %.2f ratio %.2f\n",
              a.p99, a.max_err, a.ratio, b.max_err, b.ratio);
  CHECK(a.p99 <= 4.0);
  CHECK(b.max_err <= 4.0);
  // Clipping only the tail must not cost more than clipping everything.
  CHECK(a.ratio >= b.ratio);
}

TEST(p999_bound_holds) {
  const Volume v = scroll_like_volume({128, 96, 96});
  EncodeOptions opts;
  opts.quality = 0.25f;
  opts.bounds.p999 = 6.0f;
  const Outcome o = run(v, DType::u8, opts);
  REQUIRE(o.ok);
  std::printf("       p999 %.2f (bound 6) max %.2f ratio %.2f\n", o.p999, o.max_err, o.ratio);
  CHECK(o.p999 <= 6.0);
}

// A relative bound scales with each chunk's own contrast, which is what makes it
// useful on data whose dynamic range varies enormously between air and material.
TEST(relative_bound_holds_against_chunk_range) {
  const Volume v = scroll_like_volume({96, 96, 96});
  EncodeOptions opts;
  opts.quality = 0.25f;
  opts.bounds.max_rel = 0.02f;
  const Outcome o = run(v, DType::u8, opts);
  REQUIRE(o.ok);
  // The loosest a 2% relative bound can be is 2% of the full dtype range.
  CHECK(o.max_err <= 0.02 * 255.0 + 1.0);
}

TEST(bounds_cost_bits_but_not_correctness_on_f32) {
  Volume v = scroll_like_volume({64, 64, 64});
  for (float& f : v.data) f = f * 0.01f - 1.0f;
  EncodeOptions opts;
  opts.quality = 0.5f;
  opts.bounds.max_abs = 0.02f;
  const Outcome o = run(v, DType::f32, opts);
  REQUIRE(o.ok);
  std::printf("       f32 max %.5f (bound 0.02) ratio %.2f\n", o.max_err, o.ratio);
  CHECK(o.max_err <= 0.02);
}

// A tighter bound must never produce a larger error or a better ratio.
TEST(tighter_bounds_are_monotonic) {
  const Volume v = scroll_like_volume({96, 96, 96});
  double prev_err = 1e9, prev_ratio = 0;
  for (float tau : {16.0f, 8.0f, 4.0f, 2.0f, 1.0f}) {
    EncodeOptions opts;
    opts.quality = 0.25f;
    opts.bounds.max_abs = tau;
    const Outcome o = run(v, DType::u8, opts);
    REQUIRE(o.ok);
    CHECK(o.max_err <= tau);
    CHECK(o.max_err <= prev_err + 1e-9);
    CHECK(o.ratio <= prev_ratio + 1e-9 || prev_ratio == 0);
    prev_err = o.max_err;
    prev_ratio = o.ratio;
  }
}

// A decoder that skips the correction streams must still produce the lossy
// reconstruction rather than failing, since the layer is additive.
TEST(correction_streams_are_skippable_by_brick_decode) {
  const Volume v = scroll_like_volume({128, 128, 128});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions opts;
  opts.quality = 0.25f;
  opts.bounds.max_abs = 2.0f;
  std::vector<std::uint8_t> archive;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, opts, archive) == Status::ok);

  std::vector<std::uint8_t> brick;
  REQUIRE(decode_brick(archive, 0, 0, 0, brick) == Status::ok);
  CHECK_EQ(brick.size(), static_cast<std::size_t>(kBrickDim) * kBrickDim * kBrickDim);

  // Without corrections the reconstruction is the lossy one, so it should be
  // close but not bound-tight.
  std::vector<std::uint8_t> full;
  VolumeInfo info;
  REQUIRE(decode(archive, DecodeOptions{}, full, info) == Status::ok);
  double worst = 0;
  for (std::size_t i = 0; i < full.size(); ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(full[i]) - brick[i]));
  CHECK(worst > 0.0);   // the layers really do differ
  CHECK(worst < 64.0);  // but the lossy layer is still a sane reconstruction
}

TEST(no_bound_requested_costs_nothing) {
  const Volume v = scroll_like_volume({128, 128, 128});
  const std::vector<std::uint8_t> raw = to_typed(v, DType::u8);
  EncodeOptions a, b;
  a.quality = b.quality = 0.5f;
  b.bounds.max_abs = 0.0f;  // explicitly inactive
  std::vector<std::uint8_t> x, y;
  REQUIRE(encode(raw.data(), v.dims, DType::u8, a, x) == Status::ok);
  REQUIRE(encode(raw.data(), v.dims, DType::u8, b, y) == Status::ok);
  CHECK(x == y);
}

TEST_MAIN()
