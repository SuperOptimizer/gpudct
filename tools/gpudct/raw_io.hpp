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

// Memory-mapped whole-volume I/O.
//
// A raw volume is the largest thing this tool touches and reading it into a
// vector charges the process for every byte twice over: once as anonymous
// pages that count against the working set and can only be swapped to the page
// file, and once as the file cache the read came through. A mapping is the same
// bytes with neither cost -- clean, file-backed pages the OS can drop and
// refetch under pressure. On a 1 GiB volume that is the difference between a
// process the machine can host several of and one it cannot.
//
// Both types are move-only and unmap in their destructor.
class MappedFile {
 public:
  MappedFile() = default;
  ~MappedFile();
  MappedFile(MappedFile&& o) noexcept;
  MappedFile& operator=(MappedFile&& o) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  bool open(const std::string& path, std::string& err);
  [[nodiscard]] const std::uint8_t* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

 private:
  void reset() noexcept;
  const std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
  void* h_file_ = nullptr;
  void* h_map_ = nullptr;
  int fd_ = -1;
};

// A writable mapping over a file created at exactly `bytes` long, so a decode
// can land straight in the page cache instead of filling a heap buffer that is
// then copied out to disk.
class MappedOutput {
 public:
  MappedOutput() = default;
  ~MappedOutput();
  MappedOutput(MappedOutput&& o) noexcept;
  MappedOutput& operator=(MappedOutput&& o) noexcept;
  MappedOutput(const MappedOutput&) = delete;
  MappedOutput& operator=(const MappedOutput&) = delete;

  bool create(const std::string& path, std::size_t bytes, std::string& err);
  [[nodiscard]] std::uint8_t* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  // Flushes and unmaps. Reports failures a destructor could only swallow.
  bool close(std::string& err);

 private:
  void reset() noexcept;
  std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
  void* h_file_ = nullptr;
  void* h_map_ = nullptr;
  int fd_ = -1;
};

// Parses "512,512,256" into dimensions.
bool parse_dims(const std::string& s, gpudct::Dims& out);

}  // namespace gpudct_cli
