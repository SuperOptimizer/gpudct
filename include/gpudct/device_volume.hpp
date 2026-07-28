// Compressed-in-VRAM random access (docs/DESIGN.md section 2).
//
// The motivating use case is a viewer that keeps a whole scroll *compressed* in
// device memory and decodes only the bricks it is about to draw, so effective
// VRAM is `ratio` times larger than the card. That is a latency problem, not a
// throughput one, and the ordinary decode entry points are the wrong shape for
// it: each call uploads the archive, allocates its scratch, uploads tables,
// parses every brick header on the host, and copies the result back to system
// memory. For a batch of eight bricks that fixed cost is most of the time, and
// the copy back is pure waste when the consumer is a renderer on the same device.
//
// DeviceVolume pays all of it once. The archive lives in device memory, the
// brick descriptors and entropy tables are parsed and uploaded at open time, the
// scratch buffers persist, and `decode_bricks` writes decoded voxels straight
// into a device pointer the caller owns.
#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "gpudct/gpudct.hpp"
#include "gpudct/types.hpp"

namespace gpudct {

class DeviceVolume {
 public:
  // Uploads `archive` to device memory and prepares everything decoding needs.
  // The span need not outlive the call.
  //
  // Returns backend_unavailable when this build has no CUDA, when no device is
  // present, or when the archive uses a feature the device path declines (raw
  // bricks or a correction layer) -- the caller should fall back to the CPU
  // decoder rather than receive a partially decoded volume.
  [[nodiscard]] static Status open(std::span<const std::uint8_t> archive,
                                   std::unique_ptr<DeviceVolume>& out);

  ~DeviceVolume();
  DeviceVolume(const DeviceVolume&) = delete;
  DeviceVolume& operator=(const DeviceVolume&) = delete;

  [[nodiscard]] const VolumeInfo& info() const noexcept;

  // Bytes of device memory the compressed volume and its metadata occupy. This
  // is the number a VRAM budget is drawn against.
  [[nodiscard]] std::size_t device_bytes() const noexcept;

  // Number of bricks, and the brick grid, so a caller can turn a view frustum
  // into brick indices. Index order is x fastest, then y, then z.
  [[nodiscard]] std::uint64_t brick_count() const noexcept;
  [[nodiscard]] Dims brick_dims() const noexcept;

  // Largest batch `decode_bricks` accepts, set by the persistent scratch.
  [[nodiscard]] std::uint32_t max_batch() const noexcept;

  // Decodes the listed bricks into `d_out`, which must be a device pointer with
  // room for `indices.size() * kBrickDim^3` elements of the archive's dtype.
  //
  // Output is brick-major and densely packed in the order given: brick
  // `indices[i]` lands at element offset `i * kBrickDim^3`. That is the layout a
  // brick-pool renderer wants, and it means the caller chooses the residency
  // policy -- this class holds no decoded-brick cache of its own.
  //
  // Voxels past the volume boundary in an edge brick are edge-replicated
  // padding, exactly as decode_brick returns them.
  //
  // Synchronous: returns once the decode has completed on the device.
  [[nodiscard]] Status decode_bricks(std::span<const std::uint32_t> indices, void* d_out);

  // Device memory helpers.
  //
  // decode_bricks writes to a device pointer, so a caller needs a way to make
  // one -- and the point of this header is that it can be used without linking
  // or including CUDA. A caller that already has its own device allocator should
  // use that instead; these exist so that one does not become a requirement.
  [[nodiscard]] static Status device_malloc(std::size_t bytes, void** out);
  static void device_free(void* d_ptr);
  [[nodiscard]] static Status device_to_host(void* dst, const void* d_src, std::size_t bytes);

 private:
  DeviceVolume();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpudct
