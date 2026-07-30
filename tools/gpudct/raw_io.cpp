#include "raw_io.hpp"

#include <cstdio>
#include <sstream>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace gpudct_cli {

bool read_file(const std::string& path, std::vector<std::uint8_t>& out, std::string& err) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    err = "cannot open " + path;
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  if (n < 0) {
    std::fclose(f);
    err = "cannot size " + path;
    return false;
  }
  std::fseek(f, 0, SEEK_SET);
  out.resize(static_cast<std::size_t>(n));
  const std::size_t got = out.empty() ? 0 : std::fread(out.data(), 1, out.size(), f);
  std::fclose(f);
  if (got != out.size()) {
    err = "short read on " + path;
    return false;
  }
  return true;
}

bool write_file(const std::string& path, const std::vector<std::uint8_t>& data,
                std::string& err) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    err = "cannot create " + path;
    return false;
  }
  const std::size_t put = data.empty() ? 0 : std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  if (put != data.size()) {
    err = "short write on " + path;
    return false;
  }
  return true;
}

bool parse_dims(const std::string& s, gpudct::Dims& out) {
  std::istringstream is(s);
  char c1 = 0, c2 = 0;
  unsigned long x = 0, y = 0, z = 0;
  if (!(is >> x >> c1 >> y >> c2 >> z)) return false;
  if (c1 != ',' && c1 != 'x') return false;
  if (c2 != ',' && c2 != 'x') return false;
  if (x == 0 || y == 0 || z == 0) return false;
  out.x = static_cast<std::uint32_t>(x);
  out.y = static_cast<std::uint32_t>(y);
  out.z = static_cast<std::uint32_t>(z);
  return true;
}

// --------------------------------------------------------------------------
// Memory-mapped I/O
// --------------------------------------------------------------------------

namespace {
#ifdef _WIN32
constexpr void* kNoHandle = nullptr;
bool win_is_valid(void* h) { return h != nullptr && h != INVALID_HANDLE_VALUE; }
#endif
}  // namespace

void MappedFile::reset() noexcept {
#ifdef _WIN32
  if (data_) UnmapViewOfFile(data_);
  if (win_is_valid(h_map_)) CloseHandle(h_map_);
  if (win_is_valid(h_file_)) CloseHandle(h_file_);
  h_map_ = h_file_ = kNoHandle;
#else
  if (data_ && size_) munmap(const_cast<std::uint8_t*>(data_), size_);
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
#endif
  data_ = nullptr;
  size_ = 0;
}

MappedFile::~MappedFile() { reset(); }

MappedFile::MappedFile(MappedFile&& o) noexcept
    : data_(o.data_), size_(o.size_), h_file_(o.h_file_), h_map_(o.h_map_), fd_(o.fd_) {
  o.data_ = nullptr;
  o.size_ = 0;
  o.h_file_ = o.h_map_ = nullptr;
  o.fd_ = -1;
}

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
  if (this != &o) {
    reset();
    data_ = o.data_;
    size_ = o.size_;
    h_file_ = o.h_file_;
    h_map_ = o.h_map_;
    fd_ = o.fd_;
    o.data_ = nullptr;
    o.size_ = 0;
    o.h_file_ = o.h_map_ = nullptr;
    o.fd_ = -1;
  }
  return *this;
}

bool MappedFile::open(const std::string& path, std::string& err) {
  reset();
#ifdef _WIN32
  h_file_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
  if (!win_is_valid(h_file_)) {
    err = "cannot open " + path;
    reset();
    return false;
  }
  LARGE_INTEGER li{};
  if (!GetFileSizeEx(h_file_, &li)) {
    err = "cannot size " + path;
    reset();
    return false;
  }
  size_ = static_cast<std::size_t>(li.QuadPart);
  // A zero-length file cannot be mapped, and is not an error to report here --
  // the caller's size check will reject it with a better message.
  if (size_ == 0) return true;
  h_map_ = CreateFileMappingA(h_file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!win_is_valid(h_map_)) {
    err = "cannot map " + path;
    reset();
    return false;
  }
  data_ = static_cast<const std::uint8_t*>(MapViewOfFile(h_map_, FILE_MAP_READ, 0, 0, 0));
  if (!data_) {
    err = "cannot map " + path;
    reset();
    return false;
  }
#else
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    err = "cannot open " + path;
    return false;
  }
  struct stat st {};
  if (fstat(fd_, &st) != 0) {
    err = "cannot size " + path;
    reset();
    return false;
  }
  size_ = static_cast<std::size_t>(st.st_size);
  if (size_ == 0) return true;
  void* p = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (p == MAP_FAILED) {
    err = "cannot map " + path;
    reset();
    return false;
  }
  data_ = static_cast<const std::uint8_t*>(p);
#endif
  return true;
}

void MappedOutput::reset() noexcept {
#ifdef _WIN32
  if (data_) UnmapViewOfFile(data_);
  if (win_is_valid(h_map_)) CloseHandle(h_map_);
  if (win_is_valid(h_file_)) CloseHandle(h_file_);
  h_map_ = h_file_ = nullptr;
#else
  if (data_ && size_) munmap(data_, size_);
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
#endif
  data_ = nullptr;
  size_ = 0;
}

MappedOutput::~MappedOutput() { reset(); }

MappedOutput::MappedOutput(MappedOutput&& o) noexcept
    : data_(o.data_), size_(o.size_), h_file_(o.h_file_), h_map_(o.h_map_), fd_(o.fd_) {
  o.data_ = nullptr;
  o.size_ = 0;
  o.h_file_ = o.h_map_ = nullptr;
  o.fd_ = -1;
}

MappedOutput& MappedOutput::operator=(MappedOutput&& o) noexcept {
  if (this != &o) {
    reset();
    data_ = o.data_;
    size_ = o.size_;
    h_file_ = o.h_file_;
    h_map_ = o.h_map_;
    fd_ = o.fd_;
    o.data_ = nullptr;
    o.size_ = 0;
    o.h_file_ = o.h_map_ = nullptr;
    o.fd_ = -1;
  }
  return *this;
}

bool MappedOutput::create(const std::string& path, std::size_t bytes, std::string& err) {
  reset();
  if (bytes == 0) {
    err = "refusing to create an empty volume";
    return false;
  }
#ifdef _WIN32
  h_file_ = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
  if (!win_is_valid(h_file_)) {
    err = "cannot create " + path;
    reset();
    return false;
  }
  LARGE_INTEGER li{};
  li.QuadPart = static_cast<LONGLONG>(bytes);
  h_map_ = CreateFileMappingA(h_file_, nullptr, PAGE_READWRITE, li.HighPart, li.LowPart, nullptr);
  if (!win_is_valid(h_map_)) {
    err = "cannot size " + path;
    reset();
    return false;
  }
  data_ = static_cast<std::uint8_t*>(MapViewOfFile(h_map_, FILE_MAP_WRITE, 0, 0, 0));
  if (!data_) {
    err = "cannot map " + path;
    reset();
    return false;
  }
#else
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd_ < 0) {
    err = "cannot create " + path;
    return false;
  }
  if (ftruncate(fd_, static_cast<off_t>(bytes)) != 0) {
    err = "cannot size " + path;
    reset();
    return false;
  }
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (p == MAP_FAILED) {
    err = "cannot map " + path;
    reset();
    return false;
  }
  data_ = static_cast<std::uint8_t*>(p);
#endif
  size_ = bytes;
  return true;
}

bool MappedOutput::close(std::string& err) {
  if (!data_) return true;
  bool ok = true;
#ifdef _WIN32
  if (!FlushViewOfFile(data_, 0)) ok = false;
#else
  if (msync(data_, size_, MS_SYNC) != 0) ok = false;
#endif
  reset();
  if (!ok) err = "flush failed";
  return ok;
}

}  // namespace gpudct_cli
