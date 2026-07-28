// Host-visible entry points for the CUDA backend.
//
// Declared separately from the .cu so that codec.cpp -- compiled by the host
// compiler, not nvcc -- can call into it without seeing any CUDA types.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/format.hpp"
#include "core/models.hpp"
#include "core/quant.hpp"
#include "gpudct/types.hpp"

namespace gpudct::cuda {

[[nodiscard]] bool available();

// Decodes a whole archive on the GPU.
//
// Returns Status::backend_unavailable when the archive uses something this path
// does not implement (raw-mode bricks, the correction layer), so the caller can
// fall back to the CPU rather than receive a wrong answer.
[[nodiscard]] Status decode_archive(std::span<const std::uint8_t> archive,
                                    const detail::FileHeader& h, const detail::QuantMatrix& qm,
                                    const detail::ModelSet& ms, std::span<std::uint8_t> out);

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
