// Host-visible entry points for the CUDA backend.
//
// Declared separately from the .cu so that codec.cpp -- compiled by the host
// compiler, not nvcc -- can call into it without seeing any CUDA types.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/format.hpp"
#include "core/models.hpp"
#include "core/quant.hpp"
#include "gpudct/gpudct.hpp"
#include "gpudct/types.hpp"

namespace gpudct::cuda {

[[nodiscard]] bool available();

// Decodes a whole archive on the GPU.
//
// Returns Status::backend_unavailable when the archive uses something this path
// does not implement (raw-mode bricks, the correction layer), so the caller can
// fall back to the CPU rather than receive a wrong answer.
// Persistent device-resident decoder; see include/gpudct/device_volume.hpp.
// The struct is opaque to everything outside backend.cu.
struct DeviceVolumeImpl;

// Everything the public shell needs to answer queries without seeing the
// implementation type, copied out at open time.
struct DeviceVolumeInfo {
  VolumeInfo info{};
  Dims grid{};
  std::uint64_t bricks = 0;
  std::uint32_t batch = 0;
  std::size_t device_bytes = 0;
};

[[nodiscard]] Status device_volume_open(std::span<const std::uint8_t> archive,
                                        DeviceVolumeImpl** out, DeviceVolumeInfo& summary);
void device_volume_close(DeviceVolumeImpl* v);
[[nodiscard]] Status device_malloc(std::size_t bytes, void** out);
void device_free(void* p);
[[nodiscard]] Status device_to_host(void* dst, const void* src, std::size_t bytes);
[[nodiscard]] Status device_volume_decode(DeviceVolumeImpl* v,
                                          std::span<const std::uint32_t> indices, void* d_out);

// `deblock_rms` is the quantizer's predicted voxel RMS; the device combines it
// with an activity statistic it measures itself, exactly as the host filter
// does. Pass strength <= 0 to skip filtering. When filtering, the whole volume
// must be reconstructed before any of it is read back, so this also disables
// the pipelined readback -- worth roughly 5-8%, against the 2.8x the host-side
// filter was costing.
[[nodiscard]] Status decode_archive(std::span<const std::uint8_t> archive,
                                    const detail::FileHeader& h, const detail::QuantMatrix& qm,
                                    const detail::ModelSet& ms, std::span<std::uint8_t> out,
                                    float deblock_rms = 0.0f, float deblock_strength = 0.0f);

// Encodes every brick of a volume on the GPU, filling `payloads` in brick-index
// order. The caller assembles the header and index around them.
//
// Returns Status::backend_unavailable when a chunk is too dense for the device
// scratch or a stream slab overflows, so the caller re-encodes on the CPU rather
// than shipping a truncated archive.
[[nodiscard]] Status encode_bricks(const void* data, Dims dims, DType dtype, float scale,
                                   float offset, const detail::QuantMatrix& qm, float deadzone,
                                   const detail::ModelSet& ms, std::uint32_t P, bool rdo,
                                   bool per_brick_tables,
                                   std::vector<std::vector<std::uint8_t>>& payloads);

}  // namespace gpudct::cuda
