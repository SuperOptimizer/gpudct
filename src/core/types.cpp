#include "gpudct/types.hpp"

#include <algorithm>

#ifdef GPUDCT_HAVE_CUDA
namespace gpudct::cuda {
bool available();
}
#endif

namespace gpudct {

std::string_view dtype_name(DType t) noexcept {
  switch (t) {
    case DType::u8: return "u8";
    case DType::s8: return "s8";
    case DType::u16: return "u16";
    case DType::s16: return "s16";
    case DType::u32: return "u32";
    case DType::s32: return "s32";
    case DType::f32: return "f32";
  }
  return "?";
}

bool dtype_from_name(std::string_view name, DType& out) noexcept {
  static constexpr DType all[] = {DType::u8,  DType::s8,  DType::u16, DType::s16,
                                  DType::u32, DType::s32, DType::f32};
  for (DType t : all)
    if (dtype_name(t) == name) {
      out = t;
      return true;
    }
  return false;
}

std::string_view profile_name(Profile p) noexcept {
  switch (p) {
    case Profile::archival: return "archival";
    case Profile::balanced: return "balanced";
    case Profile::viewing: return "viewing";
    case Profile::custom: return "custom";
  }
  return "?";
}

bool profile_from_name(std::string_view name, Profile& out) noexcept {
  static constexpr Profile all[] = {Profile::archival, Profile::balanced, Profile::viewing,
                                    Profile::custom};
  for (Profile p : all)
    if (profile_name(p) == name) {
      out = p;
      return true;
    }
  return false;
}

QuantParams QuantParams::for_profile(Profile p, float quality) noexcept {
  QuantParams q;
  // Radial weighting: how hard high-frequency coefficients are attenuated.
  // archival keeps fibre texture for downstream models; viewing trades it for
  // ratio. These are M1 starting points -- docs/ROADMAP.md M4 tunes them against
  // real rate-distortion curves rather than intuition.
  switch (p) {
    case Profile::archival: q.base = 4.0f;  q.a = 0.8f;  q.b = 2.0f; break;
    case Profile::balanced: q.base = 8.0f;  q.a = 3.0f;  q.b = 2.0f; break;
    case Profile::viewing:  q.base = 16.0f; q.a = 10.0f; q.b = 2.0f; break;
    case Profile::custom:   q.base = 8.0f;  q.a = 3.0f;  q.b = 2.0f; break;
  }
  // Higher quality means a finer step. Clamped so a nonsense value cannot
  // produce a zero or infinite quantizer.
  const float qual = std::clamp(quality, 0.01f, 100.0f);
  q.base /= qual;
  return q;
}

std::string_view status_message(Status s) noexcept {
  switch (s) {
    case Status::ok: return "ok";
    case Status::invalid_argument: return "invalid argument";
    case Status::corrupt_bitstream: return "corrupt bitstream";
    case Status::unsupported_version: return "unsupported format version";
    case Status::unsupported_dtype: return "unsupported data type";
    case Status::truncated: return "truncated input";
    case Status::io_error: return "I/O error";
    case Status::out_of_memory: return "out of memory";
    case Status::not_implemented: return "not implemented";
    case Status::backend_unavailable: return "backend unavailable in this build";
  }
  return "unknown error";
}

std::string_view backend_name(Backend b) noexcept {
  switch (b) {
    case Backend::automatic: return "auto";
    case Backend::cpu_scalar: return "cpu-scalar";
    case Backend::cpu_simd: return "cpu-simd";
    case Backend::cuda: return "cuda";
  }
  return "?";
}

bool backend_available(Backend b) noexcept {
  switch (b) {
    case Backend::automatic:
    case Backend::cpu_scalar: return true;
    case Backend::cpu_simd:
#ifdef GPUDCT_HAVE_SIMD
      return true;
#else
      return false;
#endif
    case Backend::cuda:
#ifdef GPUDCT_HAVE_CUDA
      return gpudct::cuda::available();
#else
      return false;
#endif
  }
  return false;
}

}  // namespace gpudct
