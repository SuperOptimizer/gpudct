// Public shell for DeviceVolume. The work is in src/cuda/backend.cu; this file
// exists so the header stays free of CUDA types, and so a build without CUDA
// still links and reports backend_unavailable rather than failing to compile.
//
// The implementation type is opaque here -- this translation unit is compiled by
// the host compiler, which has never seen a CUDA type -- so the handle is a raw
// pointer released through an explicit close, and everything a caller can query
// is copied out as plain data when the volume is opened.
#include "gpudct/device_volume.hpp"

#include "gpudct/gpudct.hpp"

#ifdef GPUDCT_HAVE_CUDA
#include "cuda/cuda_backend.hpp"
#endif

namespace gpudct {

#ifdef GPUDCT_HAVE_CUDA

struct DeviceVolume::Impl {
  cuda::DeviceVolumeImpl* v = nullptr;
  cuda::DeviceVolumeInfo summary{};

  ~Impl() {
    if (v != nullptr) cuda::device_volume_close(v);
  }
};

DeviceVolume::DeviceVolume() : impl_(std::make_unique<Impl>()) {}
DeviceVolume::~DeviceVolume() = default;

Status DeviceVolume::open(std::span<const std::uint8_t> archive,
                          std::unique_ptr<DeviceVolume>& out) {
  if (!backend_available(Backend::cuda)) return Status::backend_unavailable;
  std::unique_ptr<DeviceVolume> dv(new DeviceVolume());
  cuda::DeviceVolumeImpl* raw = nullptr;
  const Status s = cuda::device_volume_open(archive, &raw, dv->impl_->summary);
  if (s != Status::ok) {
    if (raw != nullptr) cuda::device_volume_close(raw);
    return s;
  }
  dv->impl_->v = raw;
  out = std::move(dv);
  return Status::ok;
}

const VolumeInfo& DeviceVolume::info() const noexcept { return impl_->summary.info; }
std::size_t DeviceVolume::device_bytes() const noexcept { return impl_->summary.device_bytes; }
std::uint64_t DeviceVolume::brick_count() const noexcept { return impl_->summary.bricks; }
Dims DeviceVolume::brick_dims() const noexcept { return impl_->summary.grid; }
std::uint32_t DeviceVolume::max_batch() const noexcept { return impl_->summary.batch; }

Status DeviceVolume::decode_bricks(std::span<const std::uint32_t> indices, void* d_out) {
  return cuda::device_volume_decode(impl_->v, indices, d_out);
}

Status DeviceVolume::device_malloc(std::size_t bytes, void** out) {
  return cuda::device_malloc(bytes, out);
}
void DeviceVolume::device_free(void* p) { cuda::device_free(p); }
Status DeviceVolume::device_to_host(void* dst, const void* src, std::size_t bytes) {
  return cuda::device_to_host(dst, src, bytes);
}

#else

struct DeviceVolume::Impl {
  VolumeInfo info{};
};

DeviceVolume::DeviceVolume() : impl_(std::make_unique<Impl>()) {}
DeviceVolume::~DeviceVolume() = default;

Status DeviceVolume::open(std::span<const std::uint8_t>, std::unique_ptr<DeviceVolume>&) {
  return Status::backend_unavailable;
}
const VolumeInfo& DeviceVolume::info() const noexcept { return impl_->info; }
std::size_t DeviceVolume::device_bytes() const noexcept { return 0; }
std::uint64_t DeviceVolume::brick_count() const noexcept { return 0; }
Dims DeviceVolume::brick_dims() const noexcept { return {}; }
std::uint32_t DeviceVolume::max_batch() const noexcept { return 0; }
Status DeviceVolume::decode_bricks(std::span<const std::uint32_t>, void*) {
  return Status::backend_unavailable;
}
Status DeviceVolume::device_malloc(std::size_t, void**) { return Status::backend_unavailable; }
void DeviceVolume::device_free(void*) {}
Status DeviceVolume::device_to_host(void*, const void*, std::size_t) {
  return Status::backend_unavailable;
}

#endif

}  // namespace gpudct
