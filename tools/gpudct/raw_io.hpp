// Raw binary volume I/O for the CLI.
//
// "Raw" means exactly what it says: dims.voxels() elements of the given dtype,
// x fastest, no header. It is the lowest common denominator every volumetric
// tool can produce, which makes it the right thing for a CLI to speak while the
// Zarr integration (docs/ROADMAP.md M8) is still ahead of us.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpudct/types.hpp"

namespace gpudct_cli {

bool read_file(const std::string& path, std::vector<std::uint8_t>& out, std::string& err);
bool write_file(const std::string& path, const std::vector<std::uint8_t>& data,
                std::string& err);

// Parses "512,512,256" into dimensions.
bool parse_dims(const std::string& s, gpudct::Dims& out);

}  // namespace gpudct_cli
