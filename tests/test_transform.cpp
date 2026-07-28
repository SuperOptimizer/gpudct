#include <numeric>

#include "core/transform.hpp"
#include "test.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct::detail;
using gpudct_test::Rng;

// The fast factorization and the naive matrix product must agree. This is the
// test that matters most in this file: dct16 is the one everything else uses,
// and dct16_ref is the only thing that says it is right.
TEST(dct16_matches_reference) {
  Rng rng(1234);
  for (int trial = 0; trial < 200; ++trial) {
    float x[16], a[16], b[16];
    for (float& v : x) v = rng.range(-1000.0f, 1000.0f);
    dct16(x, a);
    dct16_ref(x, b);
    for (int i = 0; i < 16; ++i) CHECK_NEAR(a[i], b[i], 2e-3);
  }
}

TEST(idct16_matches_reference) {
  Rng rng(4321);
  for (int trial = 0; trial < 200; ++trial) {
    float x[16], a[16], b[16];
    for (float& v : x) v = rng.range(-1000.0f, 1000.0f);
    idct16(x, a);
    idct16_ref(x, b);
    for (int i = 0; i < 16; ++i) CHECK_NEAR(a[i], b[i], 2e-3);
  }
}

TEST(dct16_round_trips) {
  Rng rng(99);
  for (int trial = 0; trial < 200; ++trial) {
    float x[16], c[16], y[16];
    for (float& v : x) v = rng.range(-1000.0f, 1000.0f);
    dct16(x, c);
    idct16(c, y);
    for (int i = 0; i < 16; ++i) CHECK_NEAR(x[i], y[i], 2e-3);
  }
}

// Orthonormality is not decoration: the quantizer's error model assumes error in
// the coefficient domain maps 1:1 to error in the voxel domain, which is exactly
// what Parseval gives us.
TEST(dct16_preserves_energy) {
  Rng rng(7);
  for (int trial = 0; trial < 100; ++trial) {
    float x[16], c[16];
    for (float& v : x) v = rng.range(-100.0f, 100.0f);
    dct16(x, c);
    double ex = 0, ec = 0;
    for (int i = 0; i < 16; ++i) {
      ex += static_cast<double>(x[i]) * x[i];
      ec += static_cast<double>(c[i]) * c[i];
    }
    CHECK_NEAR(ex, ec, ex * 1e-5 + 1e-6);
  }
}

TEST(dct16_basis_is_orthonormal) {
  for (int u = 0; u < 16; ++u) {
    float basis[16] = {};
    basis[u] = 1.0f;
    float spatial[16], back[16];
    idct16(basis, spatial);
    dct16(spatial, back);
    for (int v = 0; v < 16; ++v) CHECK_NEAR(back[v], (u == v) ? 1.0f : 0.0f, 1e-4);
  }
}

// A constant chunk must put all its energy in DC. For a constant c, each axis
// contributes a factor of 16 * alpha(0) = 4, so DC = 64c.
TEST(chunk_dc_of_constant) {
  std::vector<float> in(kChunkVox, 3.5f), out(kChunkVox);
  forward_chunk(in.data(), out.data());
  CHECK_NEAR(out[0], 64.0f * 3.5f, 1e-2);
  for (int i = 1; i < kChunkVox; ++i) CHECK_NEAR(out[i], 0.0f, 1e-2);
}

TEST(chunk_round_trips) {
  Rng rng(55);
  std::vector<float> in(kChunkVox), coef(kChunkVox), out(kChunkVox);
  for (float& v : in) v = rng.range(-128.0f, 127.0f);
  forward_chunk(in.data(), coef.data());
  inverse_chunk(coef.data(), out.data());
  double worst = 0;
  for (int i = 0; i < kChunkVox; ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(in[i] - out[i])));
  CHECK(worst < 1e-2);
}

TEST(chunk_matches_reference) {
  Rng rng(56);
  std::vector<float> in(kChunkVox), fast(kChunkVox), ref(kChunkVox);
  for (float& v : in) v = rng.range(-500.0f, 500.0f);
  forward_chunk(in.data(), fast.data());
  forward_chunk_ref(in.data(), ref.data());
  double worst = 0;
  for (int i = 0; i < kChunkVox; ++i)
    worst = std::max(worst, std::fabs(static_cast<double>(fast[i] - ref[i])));
  CHECK(worst < 5e-2);
}

// The significance hierarchy's index maths has to be a bijection, or masks and
// coefficients silently refer to different things.
TEST(subblock_indexing_is_a_bijection) {
  std::vector<int> seen(kChunkVox, 0);
  for (int sb = 0; sb < kSubCount; ++sb)
    for (int bit = 0; bit < kSubVox; ++bit) {
      const int idx = coeff_of_subblock_bit(sb, bit);
      REQUIRE(idx >= 0 && idx < kChunkVox);
      ++seen[static_cast<std::size_t>(idx)];
      CHECK_EQ(subblock_of_coeff(idx), sb);
      CHECK_EQ(bit_of_coeff(idx), bit);
    }
  for (int i = 0; i < kChunkVox; ++i) CHECK_EQ(seen[static_cast<std::size_t>(i)], 1);
}

TEST(bands_are_monotonic_in_frequency) {
  CHECK_EQ(band_of_subblock(0, 0, 0), 0);
  CHECK_EQ(band_of_subblock(3, 3, 3), kNumBands - 1);
  for (int sw = 0; sw < 4; ++sw)
    for (int sv = 0; sv < 4; ++sv)
      for (int su = 0; su < 3; ++su)
        CHECK(band_of_subblock(su, sv, sw) <= band_of_subblock(su + 1, sv, sw));
}

TEST_MAIN()
