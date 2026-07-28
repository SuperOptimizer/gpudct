// gpudct — public C++ API.
//
// A lossy (and optionally bounded-error) codec for large 3D scalar volumes.
// See docs/DESIGN.md for the format and docs/QUALITY.md for what "quality" is
// measured to mean here.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "gpudct/types.hpp"

namespace gpudct {

// What an archive says about itself, without decoding any voxels.
struct VolumeInfo {
  Dims dims{};
  DType dtype = DType::u8;
  Profile profile = Profile::balanced;
  std::uint32_t version = 0;
  std::uint64_t brick_count = 0;
  std::uint8_t streams_per_brick = 0;
  QuantParams quant{};
};

[[nodiscard]] Status inspect(std::span<const std::uint8_t> archive, VolumeInfo& out);

// --------------------------------------------------------------------------
// Whole-volume encode / decode.
//
// `data` points at dims.voxels() elements of `dtype`, in x-fastest order.
// --------------------------------------------------------------------------
[[nodiscard]] Status encode(const void* data, Dims dims, DType dtype,
                            const EncodeOptions& opts, std::vector<std::uint8_t>& out,
                            Backend backend = Backend::automatic);

// Decodes into `out`, which is resized to dims.voxels() * dtype_size(dtype).
//
// Convenience wrapper over decode_into. Note that resizing a std::vector
// value-initializes it, and zeroing a large output is not cheap -- 241 ms for a
// 1 GiB volume, measured, against 31 ms of actual GPU decode. Callers that
// decode repeatedly, or that care about latency, should allocate once and use
// decode_into.
[[nodiscard]] Status decode(std::span<const std::uint8_t> archive,
                            const DecodeOptions& opts, std::vector<std::uint8_t>& out,
                            VolumeInfo& info, Backend backend = Backend::automatic);

// Decodes into a caller-provided buffer, which must be exactly
// dims.voxels() * dtype_size(dtype) bytes. The buffer need not be zeroed: every
// voxel inside the volume is written.
//
// This is the entry point to prefer. It lets the caller own the allocation --
// reuse it across volumes, allocate it uninitialized, or point it at mapped
// memory -- none of which is possible when the codec owns the vector.
[[nodiscard]] Status decode_into(std::span<const std::uint8_t> archive,
                                 const DecodeOptions& opts, std::span<std::uint8_t> out,
                                 VolumeInfo& info, Backend backend = Backend::automatic);

// --------------------------------------------------------------------------
// Random access. The brick is the archive's independently decodable unit -- see
// docs/DESIGN.md section 2 for why the entropy unit is not the 16^3 chunk.
//
// `out` receives kBrickDim^3 elements of the archive's dtype. Voxels beyond the
// volume boundary are edge-replicated padding and should be discarded.
// --------------------------------------------------------------------------
[[nodiscard]] Status decode_brick(std::span<const std::uint8_t> archive,
                                  std::uint32_t bx, std::uint32_t by, std::uint32_t bz,
                                  std::vector<std::uint8_t>& out,
                                  Backend backend = Backend::automatic);

// --------------------------------------------------------------------------
// Sub-brick random access: one 16^3 chunk, at chunk coordinates (ceil(dims/16)
// grid). `out` receives kChunkVox elements of the archive's dtype.
//
// Cheaper than decode_brick without being independent of it. Chunk `ci` within a
// brick is coded into stream `ci % streams_per_brick`, and a stream's chunks
// appear in increasing index order, so this decodes one stream's prefix -- about
// `ci / P` chunks -- instead of all 512. The inverse transform runs once, for the
// requested chunk only.
//
// The saving therefore scales with `streams_per_brick`, which makes that setting
// a random-access knob as well as a GPU-parallelism one: at the recommended 16 it
// is roughly a sixteenth of a brick decode, at the default 4 roughly a quarter.
// For a viewer that caches at 16^3 granularity, this is what a cache miss costs
// instead of a whole 2 MiB brick.
// --------------------------------------------------------------------------
[[nodiscard]] Status decode_chunk(std::span<const std::uint8_t> archive,
                                  std::uint32_t cx, std::uint32_t cy, std::uint32_t cz,
                                  std::vector<std::uint8_t>& out,
                                  Backend backend = Backend::automatic);

}  // namespace gpudct
