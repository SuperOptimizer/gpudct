#include "raw_io.hpp"

#include <cstdio>
#include <sstream>

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

}  // namespace gpudct_cli
