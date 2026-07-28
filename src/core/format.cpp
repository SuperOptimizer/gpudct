#include "format.hpp"

namespace gpudct::detail {

void write_header(const FileHeader& h, std::vector<std::uint8_t>& out) {
  const std::size_t start = out.size();
  out.insert(out.end(), kMagic, kMagic + 8);
  put_u32(out, h.version);
  put_u32(out, h.flags);
  put_u32(out, h.dims.x);
  put_u32(out, h.dims.y);
  put_u32(out, h.dims.z);
  out.push_back(static_cast<std::uint8_t>(h.dtype));
  out.push_back(static_cast<std::uint8_t>(h.profile));
  out.push_back(h.streams_per_brick);
  out.push_back(h.table_version);
  put_f32(out, h.q_base);
  put_f32(out, h.q_a);
  put_f32(out, h.q_b);
  put_f32(out, h.deadzone);
  put_f32(out, h.data_scale);
  put_f32(out, h.data_offset);
  put_u64(out, h.table_offset);
  put_u64(out, h.table_size);
  put_u64(out, h.index_offset);
  put_u64(out, h.brick_count);
  put_f32(out, h.corr_step);
  (void)start;
}

Status read_header(std::span<const std::uint8_t> in, FileHeader& out) {
  if (in.size() < FileHeader::kSize) return Status::truncated;
  for (int i = 0; i < 8; ++i)
    if (in[static_cast<std::size_t>(i)] != static_cast<std::uint8_t>(kMagic[i]))
      return Status::corrupt_bitstream;

  std::size_t p = 8;
  out.version = get_u32(in, p); p += 4;
  if (out.version != kFormatVersion) return Status::unsupported_version;
  out.flags = get_u32(in, p); p += 4;
  out.dims.x = get_u32(in, p); p += 4;
  out.dims.y = get_u32(in, p); p += 4;
  out.dims.z = get_u32(in, p); p += 4;

  const std::uint8_t dt = in[p++];
  const std::uint8_t pf = in[p++];
  out.streams_per_brick = in[p++];
  out.table_version = in[p++];
  if (dt > static_cast<std::uint8_t>(DType::f32)) return Status::unsupported_dtype;
  if (pf > static_cast<std::uint8_t>(Profile::custom)) return Status::corrupt_bitstream;
  out.dtype = static_cast<DType>(dt);
  out.profile = static_cast<Profile>(pf);
  if (out.streams_per_brick == 0 || out.streams_per_brick > 64)
    return Status::corrupt_bitstream;

  out.q_base = get_f32(in, p); p += 4;
  out.q_a = get_f32(in, p); p += 4;
  out.q_b = get_f32(in, p); p += 4;
  out.deadzone = get_f32(in, p); p += 4;
  out.data_scale = get_f32(in, p); p += 4;
  out.data_offset = get_f32(in, p); p += 4;
  out.table_offset = get_u64(in, p); p += 8;
  out.table_size = get_u64(in, p); p += 8;
  out.index_offset = get_u64(in, p); p += 8;
  out.brick_count = get_u64(in, p); p += 8;
  out.corr_step = get_f32(in, p); p += 4;

  // Structural sanity. A decoder that trusts these fields is a decoder that can
  // be made to read out of bounds by a hand-edited file.
  if (out.dims.empty()) return Status::corrupt_bitstream;
  if (!(out.q_base > 0.0f) || !(out.data_scale != 0.0f)) return Status::corrupt_bitstream;
  if (!(out.corr_step > 0.0f)) return Status::corrupt_bitstream;
  const Dims g = brick_grid(out.dims);
  const std::uint64_t expect =
      static_cast<std::uint64_t>(g.x) * static_cast<std::uint64_t>(g.y) *
      static_cast<std::uint64_t>(g.z);
  if (out.brick_count != expect) return Status::corrupt_bitstream;
  if (out.index_offset < FileHeader::kSize) return Status::corrupt_bitstream;
  if (out.index_offset + out.brick_count * BrickEntry::kSize > in.size())
    return Status::truncated;
  if (out.table_size > 0 &&
      (out.table_offset < FileHeader::kSize || out.table_offset + out.table_size > in.size()))
    return Status::corrupt_bitstream;
  return Status::ok;
}

}  // namespace gpudct::detail
