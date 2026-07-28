// Synthetic test volumes (docs/QUALITY.md section 3).
//
// These exist for correctness, not for ratio claims: the adversarial ones in
// particular are shapes real data never takes, chosen because they break codecs.
#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

#include "gpudct/types.hpp"

namespace gpudct_test {

// A deterministic PRNG, so a failing test reproduces exactly. xorshift32 is
// plenty for generating test data and does not drag in <random>'s
// implementation-defined engines.
class Rng {
 public:
  explicit Rng(std::uint32_t seed = 0x9e3779b9u) : s_(seed ? seed : 1u) {}
  std::uint32_t next() {
    s_ ^= s_ << 13;
    s_ ^= s_ >> 17;
    s_ ^= s_ << 5;
    return s_;
  }
  float uniform() { return static_cast<float>(next()) * (1.0f / 4294967296.0f); }
  float range(float lo, float hi) { return lo + (hi - lo) * uniform(); }

 private:
  std::uint32_t s_;
};

struct Volume {
  gpudct::Dims dims{};
  std::vector<float> data;  // always f32 here; callers convert to the dtype under test

  float at(std::uint32_t x, std::uint32_t y, std::uint32_t z) const {
    return data[(static_cast<std::size_t>(z) * dims.y + y) * dims.x + x];
  }
  float& at(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return data[(static_cast<std::size_t>(z) * dims.y + y) * dims.x + x];
  }
};

inline Volume make(gpudct::Dims d) {
  Volume v;
  v.dims = d;
  v.data.assign(d.voxels(), 0.0f);
  return v;
}

inline Volume constant_volume(gpudct::Dims d, float value) {
  Volume v = make(d);
  for (float& f : v.data) f = value;
  return v;
}

inline Volume impulse_volume(gpudct::Dims d, float bg, float peak) {
  Volume v = constant_volume(d, bg);
  v.at(d.x / 2, d.y / 2, d.z / 2) = peak;
  return v;
}

inline Volume ramp_volume(gpudct::Dims d, float lo, float hi) {
  Volume v = make(d);
  const float n = static_cast<float>(d.x + d.y + d.z);
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x)
        v.at(x, y, z) = lo + (hi - lo) * (static_cast<float>(x + y + z) / n);
  return v;
}

inline Volume sinusoid_volume(gpudct::Dims d, float fx, float fy, float fz, float amp,
                              float dc) {
  Volume v = make(d);
  constexpr float tau = 2.0f * std::numbers::pi_v<float>;
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x)
        v.at(x, y, z) = dc + amp * std::sin(tau * (fx * static_cast<float>(x) +
                                                   fy * static_cast<float>(y) +
                                                   fz * static_cast<float>(z)));
  return v;
}

inline Volume noise_volume(gpudct::Dims d, float lo, float hi, std::uint32_t seed = 7) {
  Volume v = make(d);
  Rng rng(seed);
  for (float& f : v.data) f = rng.range(lo, hi);
  return v;
}

// Alternating extremes: the worst case for dynamic range and for any codec that
// assumes spatial correlation.
inline Volume checkerboard_volume(gpudct::Dims d, float lo, float hi) {
  Volume v = make(d);
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x) v.at(x, y, z) = ((x + y + z) & 1) ? hi : lo;
  return v;
}

// A sharp step through the middle of the volume: tests blocking artifacts and
// ringing at a discontinuity that does not align with a chunk boundary.
inline Volume step_volume(gpudct::Dims d, float lo, float hi) {
  Volume v = make(d);
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x)
        v.at(x, y, z) = (x * 3 + y * 2 + z > (d.x * 3 + d.y * 2 + d.z) / 2) ? hi : lo;
  return v;
}

// Smooth blobs plus fine-grained texture plus noise. A rough stand-in for the
// statistics of scroll CT: large soft structures carrying high-frequency fibre
// detail, which is exactly the combination a quant matrix has to balance.
inline Volume scroll_like_volume(gpudct::Dims d, std::uint32_t seed = 11) {
  Volume v = make(d);
  Rng rng(seed);
  constexpr int kBlobs = 24;
  struct Blob { float x, y, z, r, a; };
  std::vector<Blob> blobs;
  for (int i = 0; i < kBlobs; ++i)
    blobs.push_back({rng.range(0, static_cast<float>(d.x)), rng.range(0, static_cast<float>(d.y)),
                     rng.range(0, static_cast<float>(d.z)),
                     rng.range(6.0f, 0.35f * static_cast<float>(d.x) + 6.0f),
                     rng.range(40.0f, 160.0f)});

  constexpr float tau = 2.0f * std::numbers::pi_v<float>;
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x) {
        float acc = 30.0f;
        const float fx = static_cast<float>(x), fy = static_cast<float>(y),
                    fz = static_cast<float>(z);
        for (const Blob& b : blobs) {
          const float dx = fx - b.x, dy = fy - b.y, dz = fz - b.z;
          const float d2 = dx * dx + dy * dy + dz * dz;
          acc += b.a * std::exp(-d2 / (2.0f * b.r * b.r));
        }
        // Fibre-like texture: a directional carrier that a high-frequency
        // rolloff will visibly destroy.
        acc += 14.0f * std::sin(tau * (0.31f * fx + 0.07f * fy)) *
               std::exp(-0.0004f * (fz - 0.5f * static_cast<float>(d.z)) *
                        (fz - 0.5f * static_cast<float>(d.z)));
        acc += rng.range(-6.0f, 6.0f);
        v.at(x, y, z) = std::fmin(255.0f, std::fmax(0.0f, acc));
      }
  return v;
}

// A volume built from integer arithmetic only -- no exp, sin, or any other libm
// call.
//
// Golden-file tests hash the encoder's output, so their input has to be
// bit-reproducible across compilers and instruction sets. The generators above
// are not: clang vectorizes their transcendental calls to a different libm path
// under -mavx2, which moves a few values by one ULP, which flips a few
// round-to-nearest decisions, which changes the hash. That would make the golden
// test report a format change every time a build flag moved -- exactly the false
// alarm it exists to avoid.
//
// Every operation here is exact in f32: small integers, and division by a power
// of two.
inline Volume deterministic_volume(gpudct::Dims d, std::uint32_t seed = 1) {
  Volume v = make(d);
  Rng rng(seed);
  // A few smooth integer "blobs" plus a periodic integer carrier and noise.
  struct Blob { std::int32_t x, y, z, r2, a; };
  std::vector<Blob> blobs;
  for (int i = 0; i < 16; ++i) {
    const std::int32_t r = 4 + static_cast<std::int32_t>(rng.next() % 24);
    blobs.push_back({static_cast<std::int32_t>(rng.next() % (d.x ? d.x : 1)),
                     static_cast<std::int32_t>(rng.next() % (d.y ? d.y : 1)),
                     static_cast<std::int32_t>(rng.next() % (d.z ? d.z : 1)), r * r,
                     40 + static_cast<std::int32_t>(rng.next() % 120)});
  }
  for (std::uint32_t z = 0; z < d.z; ++z)
    for (std::uint32_t y = 0; y < d.y; ++y)
      for (std::uint32_t x = 0; x < d.x; ++x) {
        std::int32_t acc = 32;
        for (const Blob& b : blobs) {
          const std::int32_t dx = static_cast<std::int32_t>(x) - b.x;
          const std::int32_t dy = static_cast<std::int32_t>(y) - b.y;
          const std::int32_t dz = static_cast<std::int32_t>(z) - b.z;
          const std::int32_t d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < b.r2) acc += (b.a * (b.r2 - d2)) / b.r2;  // integer falloff
        }
        // Integer carrier with period 5 in x and 7 in y, and a little noise.
        acc += 12 * static_cast<std::int32_t>((x % 5)) - 24;
        acc += 6 * static_cast<std::int32_t>((y % 7)) - 18;
        acc += static_cast<std::int32_t>(rng.next() % 9) - 4;
        acc = acc < 0 ? 0 : (acc > 255 ? 255 : acc);
        v.at(x, y, z) = static_cast<float>(acc);
      }
  return v;
}

// Converts an f32 test volume into a typed buffer for the codec API.
inline std::vector<std::uint8_t> to_typed(const Volume& v, gpudct::DType t) {
  std::vector<std::uint8_t> out(v.data.size() * gpudct::dtype_size(t));
  for (std::size_t i = 0; i < v.data.size(); ++i) {
    const float c = std::fmin(gpudct::dtype_max(t), std::fmax(gpudct::dtype_min(t), v.data[i]));
    switch (t) {
      case gpudct::DType::u8:
        reinterpret_cast<std::uint8_t*>(out.data())[i] = static_cast<std::uint8_t>(std::lrintf(c));
        break;
      case gpudct::DType::s8:
        reinterpret_cast<std::int8_t*>(out.data())[i] = static_cast<std::int8_t>(std::lrintf(c));
        break;
      case gpudct::DType::u16:
        reinterpret_cast<std::uint16_t*>(out.data())[i] = static_cast<std::uint16_t>(std::lrintf(c));
        break;
      case gpudct::DType::s16:
        reinterpret_cast<std::int16_t*>(out.data())[i] = static_cast<std::int16_t>(std::lrintf(c));
        break;
      case gpudct::DType::u32:
        reinterpret_cast<std::uint32_t*>(out.data())[i] = static_cast<std::uint32_t>(std::llrintf(c));
        break;
      case gpudct::DType::s32:
        reinterpret_cast<std::int32_t*>(out.data())[i] = static_cast<std::int32_t>(std::llrintf(c));
        break;
      case gpudct::DType::f32:
        reinterpret_cast<float*>(out.data())[i] = v.data[i];
        break;
    }
  }
  return out;
}

inline std::vector<float> from_typed(const std::vector<std::uint8_t>& buf, gpudct::DType t,
                                     std::size_t n) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    switch (t) {
      case gpudct::DType::u8:  out[i] = reinterpret_cast<const std::uint8_t*>(buf.data())[i]; break;
      case gpudct::DType::s8:  out[i] = reinterpret_cast<const std::int8_t*>(buf.data())[i]; break;
      case gpudct::DType::u16: out[i] = reinterpret_cast<const std::uint16_t*>(buf.data())[i]; break;
      case gpudct::DType::s16: out[i] = reinterpret_cast<const std::int16_t*>(buf.data())[i]; break;
      case gpudct::DType::u32: out[i] = static_cast<float>(reinterpret_cast<const std::uint32_t*>(buf.data())[i]); break;
      case gpudct::DType::s32: out[i] = static_cast<float>(reinterpret_cast<const std::int32_t*>(buf.data())[i]); break;
      case gpudct::DType::f32: out[i] = reinterpret_cast<const float*>(buf.data())[i]; break;
    }
  }
  return out;
}

}  // namespace gpudct_test
