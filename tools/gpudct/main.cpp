// gpudct command-line interface.

#include <algorithm>
#include <chrono>
#include <memory>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpudct/device_volume.hpp"
#include "gpudct/gpudct.hpp"
#include "gpudct/metrics.hpp"
#include "core/train.hpp"
#include "raw_io.hpp"

using namespace gpudct;
using namespace gpudct_cli;

namespace {

int usage() {
  std::printf(R"(gpudct - 3D DCT compression for large volumetric data

usage:
  gpudct compress   <in.raw> <out.gdct> --dims X,Y,Z --dtype T [options]
  gpudct decompress <in.gdct> <out.raw>
  gpudct inspect    <in.gdct>
  gpudct bench      <in.raw> --dims X,Y,Z --dtype T [options]
  gpudct eval       <in.raw> --dims X,Y,Z --dtype T [options]
  gpudct metrics    <a.raw> <b.raw> --dims X,Y,Z --dtype T
  gpudct bits       <in.raw> --dims X,Y,Z --dtype T [options]
  gpudct randread   <in.gdct>                    (random-access cost)

dtypes:
  u8 s8 u16 s16 u32 s32 f32          (no 64-bit types; see docs/DESIGN.md R7)

compression options:
  --profile P      archival | balanced | viewing      (default balanced)
  --quality Q      higher is finer; scales the quantizer (default 1.0)
  --effort E       fast | normal | high               (default normal)
  --streams N      rANS streams per brick, 1..64      (default 16)
  --threads N      0 = hardware concurrency           (default 0)
  --backend B      auto | cpu-scalar | cpu-simd | cuda
  --deadzone D     dead-zone width as a fraction of a step (default 0.34)
  --qshape B       radial exponent of the quant matrix (default 2.0)
  --qamp A         radial amplitude of the quant matrix (default per profile)
  --max-abs T      hard bound on absolute error, in input units. On an integer
                   dtype, 0.5 is lossless: half a step forces exact rounding.
  --no-deblock     skip the chunk-boundary deblocking filter. It is on by
                   default: it costs no bits, raises PSNR, and removes the
                   16-voxel seam lattice. Use this to measure the format's
                   defined reconstruction on its own.
  --deblock-strength S   scale the filter thresholds; 1.0 is calibrated,
                   0 disables. Implies --deblock when > 0.
  --reps N         bench: repeat each point N times and keep the best

structure:
  chunk 16^3 (transform unit), brick 128^3 (entropy and random-access unit)
)");
  return 2;
}

struct Args {
  std::string cmd, in, out;
  Dims dims{};
  DType dtype = DType::u8;
  bool have_dims = false, have_dtype = false, have_profile = false;
  bool have_quality = false;
  int reps = 1;
  EncodeOptions enc{};
  Backend backend = Backend::automatic;
  bool deblock = true;
  float deblock_strength = 1.0f;
  float deadzone = -1.0f;  // <0 = profile default
  float qshape = -1.0f;    // <0 = profile default (radial exponent b)
  float qamp = -1.0f;      // <0 = profile default (radial amplitude a)
};

bool parse(int argc, char** argv, Args& a) {
  if (argc < 2) return false;
  a.cmd = argv[1];
  int positional = 0;
  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s needs a value\n", name);
        return nullptr;
      }
      return argv[++i];
    };
    if (s == "--dims") {
      const char* v = next("--dims");
      if (!v || !parse_dims(v, a.dims)) return false;
      a.have_dims = true;
    } else if (s == "--dtype") {
      const char* v = next("--dtype");
      if (!v || !dtype_from_name(v, a.dtype)) return false;
      a.have_dtype = true;
    } else if (s == "--profile") {
      const char* v = next("--profile");
      if (!v || !profile_from_name(v, a.enc.profile)) return false;
      a.have_profile = true;
    } else if (s == "--quality") {
      const char* v = next("--quality");
      if (!v) return false;
      a.enc.quality = std::strtof(v, nullptr);
      a.have_quality = true;
    } else if (s == "--reps") {
      const char* v = next("--reps");
      if (!v) return false;
      a.reps = std::atoi(v);
    } else if (s == "--effort") {
      const char* v = next("--effort");
      if (!v) return false;
      if (std::strcmp(v, "fast") == 0) a.enc.effort = Effort::fast;
      else if (std::strcmp(v, "normal") == 0) a.enc.effort = Effort::normal;
      else if (std::strcmp(v, "high") == 0) a.enc.effort = Effort::high;
      else return false;
    } else if (s == "--streams") {
      const char* v = next("--streams");
      if (!v) return false;
      a.enc.streams_per_brick = static_cast<std::uint8_t>(std::atoi(v));
    } else if (s == "--threads") {
      const char* v = next("--threads");
      if (!v) return false;
      a.enc.threads = std::atoi(v);
    } else if (s == "--backend") {
      const char* v = next("--backend");
      if (!v) return false;
      if (std::strcmp(v, "auto") == 0) a.backend = Backend::automatic;
      else if (std::strcmp(v, "cpu-scalar") == 0) a.backend = Backend::cpu_scalar;
      else if (std::strcmp(v, "cpu-simd") == 0) a.backend = Backend::cpu_simd;
      else if (std::strcmp(v, "cuda") == 0) a.backend = Backend::cuda;
      else return false;
    } else if (s == "--deadzone") {
      const char* v = next("--deadzone");
      if (!v) return false;
      a.deadzone = std::strtof(v, nullptr);
    } else if (s == "--qamp") {
      const char* v = next("--qamp");
      if (!v) return false;
      a.qamp = std::strtof(v, nullptr);
    } else if (s == "--qshape") {
      const char* v = next("--qshape");
      if (!v) return false;
      a.qshape = std::strtof(v, nullptr);
    } else if (s == "--max-abs") {
      const char* v = next("--max-abs");
      if (!v) return false;
      a.enc.bounds.max_abs = std::strtof(v, nullptr);
    } else if (s == "--deblock") {
      a.deblock = true;
    } else if (s == "--no-deblock") {
      a.deblock = false;
    } else if (s == "--deblock-strength") {
      const char* v = next("--deblock-strength");
      if (!v) return false;
      a.deblock_strength = std::strtof(v, nullptr);
      a.deblock = a.deblock_strength > 0.0f;
    } else if (s.rfind("--", 0) == 0) {
      std::fprintf(stderr, "error: unknown option %s\n", s.c_str());
      return false;
    } else {
      if (positional == 0) a.in = s;
      else if (positional == 1) a.out = s;
      else return false;
      ++positional;
    }
  }
  return true;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Force backend initialization before timing.
//
// Creating a CUDA context and loading kernels costs on the order of a second,
// once per process. Including that in a throughput number says more about the
// driver than the codec, and a library caller pays it once no matter how many
// volumes follow. The CLI reports it separately instead.
double warm_up_backend(Backend b) {
  if (b != Backend::cuda) return 0.0;
  const auto t0 = std::chrono::steady_clock::now();
  (void)backend_available(b);
  return seconds_since(t0);
}

void print_size(const char* label, std::size_t bytes) {
  const double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
  std::printf("%-14s %12zu bytes (%.2f MiB)\n", label, bytes, mb);
}

int cmd_compress(const Args& a) {
  if (a.in.empty() || a.out.empty() || !a.have_dims || !a.have_dtype) return usage();

  MappedFile raw;
  std::string err;
  if (!raw.open(a.in, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  const std::size_t expect = a.dims.voxels() * dtype_size(a.dtype);
  if (raw.size() != expect) {
    std::fprintf(stderr, "error: %s is %zu bytes, but %ux%ux%u of %s needs %zu\n", a.in.c_str(),
                 raw.size(), a.dims.x, a.dims.y, a.dims.z,
                 std::string(dtype_name(a.dtype)).c_str(), expect);
    return 1;
  }

  const double warm = warm_up_backend(a.backend);
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::uint8_t> archive;
  EncodeOptions eo = a.enc;
  eo.deadzone_override = a.deadzone;
  eo.qshape_override = a.qshape;
  eo.qamp_override = a.qamp;
  const Status s = encode(raw.data(), a.dims, a.dtype, eo, archive, a.backend);
  const double dt = seconds_since(t0);
  if (s != Status::ok) {
    std::fprintf(stderr, "error: encode failed: %s\n", std::string(status_message(s)).c_str());
    return 1;
  }
  if (!write_file(a.out, archive, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }

  print_size("input", raw.size());
  print_size("output", archive.size());
  std::printf("%-14s %12.2fx\n", "ratio",
              static_cast<double>(raw.size()) / static_cast<double>(archive.size()));
  std::printf("%-14s %12.3f bits/voxel\n", "rate",
              8.0 * static_cast<double>(archive.size()) / static_cast<double>(a.dims.voxels()));
  std::printf("%-14s %12.3f s (%.1f MB/s)\n", "encode", dt,
              static_cast<double>(raw.size()) / dt / 1e6);
  // Reported separately, never folded into the throughput above: it is a
  // once-per-process driver cost, not something the codec does per volume.
  if (warm > 0.0) std::printf("%-14s %12.3f s\n", "backend init", warm);
  return 0;
}

int cmd_decompress(const Args& a) {
  if (a.in.empty() || a.out.empty()) return usage();
  std::vector<std::uint8_t> archive;
  std::string err;
  if (!read_file(a.in, archive, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }

  DecodeOptions opts;
  opts.threads = a.enc.threads;
  opts.deblock = a.deblock;
  opts.deblock_strength = a.deblock_strength;
  VolumeInfo probe;
  if (const Status ps = inspect(archive, probe); ps != Status::ok) {
    std::fprintf(stderr, "error: %s\n", std::string(status_message(ps)).c_str());
    return 1;
  }
  // Decode straight into the mapped output file. Previously this allocated an
  // uninitialized heap block, decoded into it, copied the whole thing into a
  // vector, and handed that to write_file -- three full-volume buffers live at
  // once for a job that needs none of them. The mapping is the decode target and
  // the file at the same time.
  const std::size_t nbytes = probe.dims.voxels() * dtype_size(probe.dtype);
  MappedOutput out;
  if (!out.create(a.out, nbytes, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }

  const double warm = warm_up_backend(a.backend);
  const auto t0 = std::chrono::steady_clock::now();
  VolumeInfo info;
  const Status s = decode_into(archive, opts, {out.data(), nbytes}, info, a.backend);
  const double dt = seconds_since(t0);
  if (s != Status::ok) {
    std::fprintf(stderr, "error: decode failed: %s\n", std::string(status_message(s)).c_str());
    return 1;
  }
  if (!out.close(err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }

  std::printf("%-14s %ux%ux%u %s\n", "volume", info.dims.x, info.dims.y, info.dims.z,
              std::string(dtype_name(info.dtype)).c_str());
  print_size("output", nbytes);
  std::printf("%-14s %12.3f s (%.1f MB/s)\n", "decode", dt,
              static_cast<double>(nbytes) / dt / 1e6);
  if (warm > 0.0) std::printf("%-14s %12.3f s\n", "backend init", warm);
  return 0;
}

int cmd_inspect(const Args& a) {
  if (a.in.empty()) return usage();
  std::vector<std::uint8_t> archive;
  std::string err;
  if (!read_file(a.in, archive, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  VolumeInfo info;
  const Status s = inspect(archive, info);
  if (s != Status::ok) {
    std::fprintf(stderr, "error: %s\n", std::string(status_message(s)).c_str());
    return 1;
  }
  const std::size_t raw = info.dims.voxels() * dtype_size(info.dtype);
  std::printf("%-18s %u\n", "format version", info.version);
  std::printf("%-18s %ux%ux%u\n", "dimensions", info.dims.x, info.dims.y, info.dims.z);
  std::printf("%-18s %s\n", "dtype", std::string(dtype_name(info.dtype)).c_str());
  std::printf("%-18s %s\n", "profile", std::string(profile_name(info.profile)).c_str());
  std::printf("%-18s %llu (128^3 each)\n", "bricks",
              static_cast<unsigned long long>(info.brick_count));
  std::printf("%-18s %u\n", "streams per brick", info.streams_per_brick);
  std::printf("%-18s base=%.3f a=%.3f b=%.3f deadzone=%.3f\n", "quantizer", info.quant.base,
              info.quant.a, info.quant.b, info.quant.deadzone);
  std::printf("%-18s %.2fx (%.3f bits/voxel)\n", "ratio",
              static_cast<double>(raw) / static_cast<double>(archive.size()),
              8.0 * static_cast<double>(archive.size()) /
                  static_cast<double>(info.dims.voxels()));
  return 0;
}

int cmd_bench(const Args& a) {
  if (a.in.empty() || !a.have_dims || !a.have_dtype) return usage();
  MappedFile raw;
  std::string err;
  if (!raw.open(a.in, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  const std::size_t expect = a.dims.voxels() * dtype_size(a.dtype);
  if (raw.size() != expect) {
    std::fprintf(stderr, "error: size mismatch (%zu vs %zu)\n", raw.size(), expect);
    return 1;
  }

  std::printf("%-10s %10s %10s %12s %12s\n", "quality", "ratio", "bpv", "enc MB/s", "dec MB/s");
  // An explicit --quality pins the sweep to that one point, which is what a
  // throughput experiment wants; the six-point sweep is for rate-distortion.
  const std::vector<float> qualities =
      a.have_quality ? std::vector<float>{a.enc.quality}
                     : std::vector<float>{0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f};
  const int reps = a.reps > 0 ? a.reps : 1;

  for (float q : qualities) {
    EncodeOptions opts = a.enc;
    opts.quality = q;
    opts.deadzone_override = a.deadzone;
    opts.qshape_override = a.qshape;
    opts.qamp_override = a.qamp;

    // Best of N, not mean: this machine's run-to-run spread is wide enough that
    // a mean mostly measures thermal drift (docs/QUALITY.md).
    std::vector<std::uint8_t> decoded(raw.size());
    std::vector<std::uint8_t> archive;
    double te = 1e30, td = 1e30;
    for (int r = 0; r < reps; ++r) {
      std::vector<std::uint8_t> a2;
      const auto t0 = std::chrono::steady_clock::now();
      if (encode(raw.data(), a.dims, a.dtype, opts, a2, a.backend) != Status::ok) return 1;
      te = std::min(te, seconds_since(t0));
      archive = std::move(a2);

      // decode_into a buffer allocated once, not decode() into a fresh vector.
      // std::vector value-initializes, so a fresh 1 GB output costs a full
      // zero-fill plus page faults -- measured at more than every other stage of
      // decode combined. That cost is real for a caller who needs a new buffer,
      // but it is allocator behaviour rather than codec throughput, and the API
      // already offers decode_into for callers who can reuse one.
      DecodeOptions dopts;
      dopts.threads = a.enc.threads;
      VolumeInfo info;
      const auto t1 = std::chrono::steady_clock::now();
      if (decode_into(archive, dopts, decoded, info, a.backend) != Status::ok) return 1;
      td = std::min(td, seconds_since(t1));
    }

    std::printf("%-10.2f %10.2f %10.4f %12.1f %12.1f\n", static_cast<double>(q),
                static_cast<double>(raw.size()) / static_cast<double>(archive.size()),
                8.0 * static_cast<double>(archive.size()) /
                    static_cast<double>(a.dims.voxels()),
                static_cast<double>(raw.size()) / te / 1e6,
                static_cast<double>(raw.size()) / td / 1e6);
  }
  return 0;
}

// Converts a typed buffer to f32 for the metrics code.
std::vector<float> to_float(const std::uint8_t* buf, DType t, std::size_t n) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    switch (t) {
      case DType::u8:  out[i] = buf[i]; break;
      case DType::s8:  out[i] = reinterpret_cast<const std::int8_t*>(buf)[i]; break;
      case DType::u16: out[i] = reinterpret_cast<const std::uint16_t*>(buf)[i]; break;
      case DType::s16: out[i] = reinterpret_cast<const std::int16_t*>(buf)[i]; break;
      case DType::u32: out[i] = static_cast<float>(reinterpret_cast<const std::uint32_t*>(buf)[i]); break;
      case DType::s32: out[i] = static_cast<float>(reinterpret_cast<const std::int32_t*>(buf)[i]); break;
      case DType::f32: out[i] = reinterpret_cast<const float*>(buf)[i]; break;
    }
  }
  return out;
}

// Full rate-distortion sweep with the complete metric set, on a real file.
// This is the command that answers "is this ratio actually usable", which a
// ratio number on its own never does.
int cmd_eval(const Args& a) {
  if (a.in.empty() || !a.have_dims || !a.have_dtype) return usage();
  MappedFile raw;
  std::string err;
  if (!raw.open(a.in, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  const std::size_t expect = a.dims.voxels() * dtype_size(a.dtype);
  if (raw.size() != expect) {
    std::fprintf(stderr, "error: %s is %zu bytes, but %ux%ux%u of %s needs %zu\n", a.in.c_str(),
                 raw.size(), a.dims.x, a.dims.y, a.dims.z,
                 std::string(dtype_name(a.dtype)).c_str(), expect);
    return 1;
  }
  const std::vector<float> orig = to_float(raw.data(), a.dtype, a.dims.voxels());

  std::printf("%s  %ux%ux%u %s  (%.1f MiB)\n", a.in.c_str(), a.dims.x, a.dims.y, a.dims.z,
              std::string(dtype_name(a.dtype)).c_str(),
              static_cast<double>(raw.size()) / 1048576.0);
  std::printf("%-9s %-7s %8s %7s %8s %8s %7s %6s %6s %6s %6s %s\n", "profile", "qual", "ratio",
              "bpv", "psnr", "ssim3d", "mae", "p95", "p99", "p999", "max", "band energy lo->hi");

  // An explicit --profile narrows the sweep to that one curve. Tuning runs read
  // a single profile and the other two triple their cost for nothing.
  const std::vector<Profile> profiles =
      a.have_profile ? std::vector<Profile>{a.enc.profile}
                     : std::vector<Profile>{Profile::archival, Profile::balanced,
                                            Profile::viewing};
  for (Profile p : profiles) {
    for (float q : {0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f}) {
      EncodeOptions opts = a.enc;
      opts.profile = p;
      opts.quality = q;
      opts.deadzone_override = a.deadzone;
      opts.qshape_override = a.qshape;
      opts.qamp_override = a.qamp;
      std::vector<std::uint8_t> archive;
      if (encode(raw.data(), a.dims, a.dtype, opts, archive, a.backend) != Status::ok) continue;

      DecodeOptions dopts;
      dopts.threads = a.enc.threads;
      dopts.deblock = a.deblock;
      dopts.deblock_strength = a.deblock_strength;
      std::vector<std::uint8_t> out;
      VolumeInfo info;
      if (decode(archive, dopts, out, info, a.backend) != Status::ok) continue;

      const std::vector<float> dec = to_float(out.data(), a.dtype, a.dims.voxels());
      Metrics m = compute_metrics(orig.data(), dec.data(), a.dims, a.dtype);
      m.ratio = static_cast<double>(raw.size()) / static_cast<double>(archive.size());
      m.bits_per_voxel =
          8.0 * static_cast<double>(archive.size()) / static_cast<double>(a.dims.voxels());

      std::printf("%-9s %-7.3f %7.2fx %7.4f %8.2f %8.5f %7.3f %6.0f %6.0f %6.0f %6.0f  %.2f %.2f %.2f %.2f\n",
                  std::string(profile_name(p)).c_str(), static_cast<double>(q), m.ratio,
                  m.bits_per_voxel, m.psnr_range, m.ssim, m.mae, m.abs_err.p95, m.abs_err.p99,
                  m.abs_err.p999, m.abs_err.max, m.band_energy_ratio[0], m.band_energy_ratio[1],
                  m.band_energy_ratio[2], m.band_energy_ratio[3]);
    }
  }
  return 0;
}

int cmd_metrics(const Args& a) {
  if (a.in.empty() || a.out.empty() || !a.have_dims || !a.have_dtype) return usage();
  MappedFile ba, bb;
  std::string err;
  if (!ba.open(a.in, err) || !bb.open(a.out, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  const std::size_t expect = a.dims.voxels() * dtype_size(a.dtype);
  if (ba.size() != expect || bb.size() != expect) {
    std::fprintf(stderr, "error: inputs must both be %zu bytes\n", expect);
    return 1;
  }
  const std::vector<float> fa = to_float(ba.data(), a.dtype, a.dims.voxels());
  const std::vector<float> fb = to_float(bb.data(), a.dtype, a.dims.voxels());
  const Metrics m = compute_metrics(fa.data(), fb.data(), a.dims, a.dtype);
  std::printf("%s", format_report(m, true).c_str());

  const std::vector<BrickScore> worst = worst_bricks(fa.data(), fb.data(), a.dims, 10);
  if (worst.size() > 1) {
    std::printf("worst bricks by rmse\n");
    for (const BrickScore& b : worst)
      std::printf("  (%4u,%4u,%4u)  rmse %8.4f  p99 %8.2f  max %8.2f\n", b.bx, b.by, b.bz,
                  b.rmse, b.p99, b.max_abs);
  }
  return 0;
}

// Where the coded bits go, by stage. Answers "what should I optimize next".
int cmd_bits(const Args& a) {
  if (a.in.empty() || !a.have_dims || !a.have_dtype) return usage();
  MappedFile raw;
  std::string err;
  if (!raw.open(a.in, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  if (raw.size() != a.dims.voxels() * dtype_size(a.dtype)) {
    std::fprintf(stderr, "error: size mismatch\n");
    return 1;
  }
  const gpudct::detail::BitBreakdown b =
      gpudct::detail::measure_bits(raw.data(), a.dims, a.dtype, a.enc);
  const double tot = b.total();
  if (tot <= 0) return 1;

  struct Row { const char* name; double bits; };
  const Row rows[] = {
      {"L2 masks (coeff significance)", b.l2_masks},
      {"AC levels", b.levels},
      {"mantissas (bypass)", b.mantissas},
      {"signs (bypass)", b.signs},
      {"DC magnitudes", b.dc},
      {"L1 masks (sub-block significance)", b.l1_masks},
      {"escape bit-lengths", b.escapes},
      {"chunk flags", b.chunk_flags},
      {"exponents", b.exponents},
  };
  std::printf("%-36s %12s %8s %10s\n", "stage", "bits", "share", "bits/voxel");
  for (const Row& r : rows)
    std::printf("%-36s %12.0f %7.1f%% %10.4f\n", r.name, r.bits, 100.0 * r.bits / tot,
                r.bits / static_cast<double>(b.voxels));
  std::printf("%-36s %12.0f %7.1f%% %10.4f\n", "total", tot, 100.0,
              tot / static_cast<double>(b.voxels));
  return 0;
}

// What a random read actually costs, per chunk and per brick.
//
// The question a viewer cares about is not throughput but the cost of a cache
// miss. Both units are measured over the same random positions so the comparison
// is not confounded by which part of the volume gets touched.
int cmd_randread(const Args& a) {
  if (a.in.empty()) return usage();
  std::vector<std::uint8_t> archive;
  std::string err;
  if (!read_file(a.in, archive, err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  VolumeInfo info;
  if (inspect(archive, info) != Status::ok) {
    std::fprintf(stderr, "error: not a gpudct archive\n");
    return 1;
  }

  const Dims bg = brick_grid(info.dims);
  const Dims cg{(info.dims.x + kChunkDim - 1) / kChunkDim,
                (info.dims.y + kChunkDim - 1) / kChunkDim,
                (info.dims.z + kChunkDim - 1) / kChunkDim};
  const int n = 200;
  // A fixed LCG, not a random device: the two loops must visit the same places,
  // and a rerun must be comparable to the last one.
  std::uint32_t seed = 12345;
  auto next = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };

  std::vector<std::uint32_t> pos(static_cast<std::size_t>(n) * 3);
  for (int i = 0; i < n * 3; ++i) pos[static_cast<std::size_t>(i)] = next();

  std::vector<std::uint8_t> buf;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i)
    if (decode_chunk(archive, pos[i * 3 + 0] % cg.x, pos[i * 3 + 1] % cg.y,
                     pos[i * 3 + 2] % cg.z, buf, a.backend) != Status::ok)
      return 1;
  const double tc = seconds_since(t0);

  const auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i)
    if (decode_brick(archive, (pos[i * 3 + 0] % cg.x) / kBrickChunks,
                     (pos[i * 3 + 1] % cg.y) / kBrickChunks,
                     (pos[i * 3 + 2] % cg.z) / kBrickChunks, buf, a.backend) != Status::ok)
      return 1;
  const double tb = seconds_since(t1);

  std::printf("%ux%ux%u  %u streams/brick  %u bricks, %u chunks\n", info.dims.x, info.dims.y,
              info.dims.z, info.streams_per_brick, bg.x * bg.y * bg.z, cg.x * cg.y * cg.z);
  std::printf("%-22s %10.3f ms/read\n", "decode_chunk (16^3)", tc / n * 1000.0);
  std::printf("%-22s %10.3f ms/read\n", "decode_brick (128^3)", tb / n * 1000.0);
  std::printf("%-22s %10.2fx cheaper\n", "chunk vs brick", tb / tc);

  // The VRAM-residency path: archive resident on the device, decode straight
  // into device memory, nothing copied back. This is the number a viewer that
  // keeps the volume compressed in VRAM actually lives with.
  std::unique_ptr<DeviceVolume> dv;
  if (DeviceVolume::open(archive, dv) == Status::ok) {
    const std::size_t brick_vox = static_cast<std::size_t>(kBrickDim) * kBrickDim * kBrickDim;
    const std::size_t esz = dtype_size(info.dtype);
    for (std::uint32_t batch : {1u, 8u, 32u}) {
      if (batch > dv->brick_count() || batch > dv->max_batch()) continue;
      void* d_out = nullptr;
      if (DeviceVolume::device_malloc(batch * brick_vox * esz, &d_out) != Status::ok) break;
      std::vector<std::uint32_t> want(batch);
      for (std::uint32_t i = 0; i < batch; ++i)
        want[i] = static_cast<std::uint32_t>(next() % dv->brick_count());
      // One warm-up: the first call still faults in lazily-created device state,
      // and reporting that as steady state would measure the driver.
      (void)dv->decode_bricks(want, d_out);

      const int iters = 20;
      const auto t2 = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        for (std::uint32_t k = 0; k < batch; ++k)
          want[k] = static_cast<std::uint32_t>(next() % dv->brick_count());
        if (dv->decode_bricks(want, d_out) != Status::ok) break;
      }
      const double dt = seconds_since(t2) / iters;
      DeviceVolume::device_free(d_out);
      std::printf("DeviceVolume %2u brick%s  %10.3f ms  %9.1f MB/s (device-resident)\n", batch,
                  batch == 1 ? " " : "s", dt * 1000.0,
                  static_cast<double>(batch * brick_vox * esz) / dt / 1e6);
    }
    std::printf("%-22s %10.1f MiB compressed in VRAM\n", "device footprint",
                static_cast<double>(dv->device_bytes()) / 1048576.0);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse(argc, argv, a)) return usage();
  if (a.cmd == "compress") return cmd_compress(a);
  if (a.cmd == "decompress") return cmd_decompress(a);
  if (a.cmd == "inspect") return cmd_inspect(a);
  if (a.cmd == "bench") return cmd_bench(a);
  if (a.cmd == "eval") return cmd_eval(a);
  if (a.cmd == "metrics") return cmd_metrics(a);
  if (a.cmd == "bits") return cmd_bits(a);
  if (a.cmd == "randread") return cmd_randread(a);
  if (a.cmd == "-h" || a.cmd == "--help" || a.cmd == "help") {
    usage();
    return 0;
  }
  return usage();
}
