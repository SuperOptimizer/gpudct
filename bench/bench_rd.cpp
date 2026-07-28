// Rate-distortion sweep (docs/ROADMAP.md M2).
//
// The point of this tool is to make every later ratio decision answerable with a
// curve instead of an argument. It sweeps quality across a set of volumes and
// prints rate against each quality metric, so a change to the quant matrix or
// the entropy model shows up as a curve moving, not as a single number that
// might be noise.
//
// Baseline codecs (3ddct, ZFP, SZ3, blosc2+zstd) plug in behind the same
// interface; they are not wired up here yet because none of them are vendored.
// The column layout already has room for them.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "gpudct/gpudct.hpp"
#include "gpudct/metrics.hpp"
#include "volumes.hpp"

using namespace gpudct;
using namespace gpudct_test;

namespace {

struct Row {
  float quality;
  double ratio, bpv, psnr, ssim, mae, p99, maxerr, blockiness, laplacian;
  std::array<double, 4> bands;
  double mae_high_grad;
  double enc_mbs, dec_mbs;
};

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool sweep_one(const Volume& v, DType dtype, Profile profile, float quality, Row& row) {
  const std::vector<std::uint8_t> raw = to_typed(v, dtype);

  EncodeOptions opts;
  opts.profile = profile;
  opts.quality = quality;

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::uint8_t> archive;
  if (encode(raw.data(), v.dims, dtype, opts, archive) != Status::ok) return false;
  const double te = seconds_since(t0);

  const auto t1 = std::chrono::steady_clock::now();
  std::vector<std::uint8_t> out;
  VolumeInfo info;
  if (decode(archive, DecodeOptions{}, out, info) != Status::ok) return false;
  const double td = seconds_since(t1);

  const std::vector<float> orig = from_typed(raw, dtype, v.data.size());
  const std::vector<float> dec = from_typed(out, dtype, v.data.size());
  Metrics m = compute_metrics(orig.data(), dec.data(), v.dims, dtype);

  row.quality = quality;
  row.ratio = static_cast<double>(raw.size()) / static_cast<double>(archive.size());
  row.bpv = 8.0 * static_cast<double>(archive.size()) / static_cast<double>(v.dims.voxels());
  row.psnr = m.psnr_range;
  row.ssim = m.ssim;
  row.mae = m.mae;
  row.p99 = m.abs_err.p99;
  row.maxerr = m.abs_err.max;
  row.blockiness = m.blockiness;
  row.laplacian = m.laplacian_ratio;
  row.bands = m.band_energy_ratio;
  row.mae_high_grad = m.mae_high_gradient;
  row.enc_mbs = static_cast<double>(raw.size()) / te / 1e6;
  row.dec_mbs = static_cast<double>(raw.size()) / td / 1e6;
  return true;
}

void print_header(const char* title) {
  std::printf("\n== %s ==\n", title);
  std::printf("%7s %8s %7s %8s %8s %8s %8s %7s %7s %6s %s\n", "quality", "ratio", "bpv",
              "psnr", "ssim3d", "mae", "p99", "maxerr", "block", "lapl", "band energy (lo->hi)");
}

void print_row(const Row& r) {
  std::printf("%7.2f %8.2f %7.4f %8.2f %8.5f %8.4f %8.2f %7.1f %7.3f %6.3f  %.3f %.3f %.3f %.3f\n",
              static_cast<double>(r.quality), r.ratio, r.bpv, r.psnr, r.ssim, r.mae, r.p99,
              r.maxerr, r.blockiness, r.laplacian, r.bands[0], r.bands[1], r.bands[2],
              r.bands[3]);
}

}  // namespace

int main(int argc, char** argv) {
  Dims dims{128, 128, 128};
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) {
      quick = true;
      dims = {64, 64, 64};
    }
  }

  const float qualities_full[] = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
  const float qualities_quick[] = {0.25f, 1.0f, 4.0f};
  const std::span<const float> qualities =
      quick ? std::span<const float>(qualities_quick) : std::span<const float>(qualities_full);

  struct Corpus {
    const char* name;
    Volume vol;
  };
  std::vector<Corpus> corpus;
  corpus.push_back({"scroll-like (blobs + fibre texture + noise)", scroll_like_volume(dims)});
  corpus.push_back({"smooth ramp", ramp_volume(dims, 10.0f, 240.0f)});
  corpus.push_back({"sharp step", step_volume(dims, 20.0f, 230.0f)});
  corpus.push_back({"uniform noise (incompressible)", noise_volume(dims, 0.0f, 255.0f)});

  for (const Corpus& c : corpus) {
    for (Profile p : {Profile::archival, Profile::balanced, Profile::viewing}) {
      const std::string title =
          std::string(c.name) + "  [" + std::string(profile_name(p)) + "]";
      print_header(title.c_str());
      for (float q : qualities) {
        Row r{};
        if (!sweep_one(c.vol, DType::u8, p, q, r)) {
          std::fprintf(stderr, "sweep failed at quality %f\n", static_cast<double>(q));
          return 1;
        }
        print_row(r);
      }
    }
  }

  // Throughput is reported separately and at one operating point, because it is
  // dominated by volume size rather than quality and the per-row numbers above
  // are too noisy at these sizes to mean anything.
  {
    Row r{};
    if (sweep_one(corpus[0].vol, DType::u8, Profile::balanced, 1.0f, r))
      std::printf("\nthroughput (scroll-like, balanced, q=1): encode %.1f MB/s, decode %.1f MB/s\n",
                  r.enc_mbs, r.dec_mbs);
  }
  return 0;
}
