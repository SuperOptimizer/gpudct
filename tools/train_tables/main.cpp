// Trains the static entropy models on a corpus (docs/ROADMAP.md M3).
//
// Static models are the price of parallel decode: the decoder cannot adapt, so
// the tables have to be right before coding starts. The shipped defaults are
// analytic priors -- educated guesses about how coefficient significance falls
// off with frequency -- and this tool replaces them with measurements.
//
// Counts are gathered through the production encoder (detail::collect_counts),
// not a reimplementation of it, so what gets trained is exactly what gets coded.
//
// Usage:
//   train_tables --dims X,Y,Z --dtype T [--quality Q] out.inc in1.raw in2.raw ...
//
// The emitted .inc is committed and compiled in; see src/core/models.cpp.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/models.hpp"
#include "core/train.hpp"
#include "gpudct/types.hpp"

using namespace gpudct;
using namespace gpudct::detail;

namespace {

bool read_file(const std::string& path, std::vector<std::uint8_t>& out) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n < 0) {
    std::fclose(f);
    return false;
  }
  out.resize(static_cast<std::size_t>(n));
  const std::size_t got = out.empty() ? 0 : std::fread(out.data(), 1, out.size(), f);
  std::fclose(f);
  return got == out.size();
}

bool parse_dims(const char* s, Dims& d) {
  unsigned long x = 0, y = 0, z = 0;
  if (std::sscanf(s, "%lu,%lu,%lu", &x, &y, &z) != 3) return false;
  if (x == 0 || y == 0 || z == 0) return false;
  d = {static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y),
       static_cast<std::uint32_t>(z)};
  return true;
}

const char* model_name(std::uint16_t i) {
  if (i == ModelIndex::bypass) return "bypass";
  if (i == ModelIndex::exponent) return "exponent";
  if (i == ModelIndex::chunk_nonzero) return "chunk_nonzero";
  if (i >= ModelIndex::l1_base && i < ModelIndex::l1_base + ModelIndex::l1_count) return "l1";
  if (i >= ModelIndex::l2_base && i < ModelIndex::l2_base + ModelIndex::l2_count) return "l2";
  if (i >= ModelIndex::level_base && i < ModelIndex::level_base + ModelIndex::level_count)
    return "level";
  if (i == ModelIndex::dc_len) return "dc_len";
  if (i == ModelIndex::esc_len) return "esc_len";
  if (i == ModelIndex::corr_flag) return "corr_flag";
  if (i == ModelIndex::corr_mag) return "corr_mag";
  return "?";
}

}  // namespace

int main(int argc, char** argv) {
  Dims dims{};
  DType dtype = DType::u8;
  bool have_dims = false;
  float quality = 1.0f;
  std::string out_path;
  std::vector<std::string> inputs;

  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "--dims" && i + 1 < argc) {
      if (!parse_dims(argv[++i], dims)) return 2;
      have_dims = true;
    } else if (s == "--dtype" && i + 1 < argc) {
      if (!dtype_from_name(argv[++i], dtype)) return 2;
    } else if (s == "--quality" && i + 1 < argc) {
      quality = std::strtof(argv[++i], nullptr);
    } else if (s.rfind("--", 0) == 0) {
      std::fprintf(stderr, "unknown option %s\n", s.c_str());
      return 2;
    } else if (out_path.empty()) {
      out_path = s;
    } else {
      inputs.push_back(s);
    }
  }

  if (!have_dims || out_path.empty() || inputs.empty()) {
    std::fprintf(stderr,
                 "usage: train_tables --dims X,Y,Z --dtype T [--quality Q] out.inc in.raw...\n");
    return 2;
  }

  // Train across a sweep of quality settings rather than one. The coefficient
  // statistics at 20:1 look nothing like those at 3:1, and a table trained at a
  // single operating point is mistuned everywhere else.
  const float sweep[] = {0.25f, 0.5f, 1.0f, 2.0f};

  CountCollector total;
  for (const std::string& path : inputs) {
    std::vector<std::uint8_t> raw;
    if (!read_file(path, raw)) {
      std::fprintf(stderr, "cannot read %s\n", path.c_str());
      return 1;
    }
    const std::size_t expect = dims.voxels() * dtype_size(dtype);
    if (raw.size() != expect) {
      std::fprintf(stderr, "%s: %zu bytes, expected %zu\n", path.c_str(), raw.size(), expect);
      return 1;
    }
    for (float q : sweep) {
      for (Profile p : {Profile::archival, Profile::balanced, Profile::viewing}) {
        for (Effort e : {Effort::normal}) {
          EncodeOptions opts;
          opts.profile = p;
          opts.quality = q * quality;
          opts.effort = e;
          collect_counts(raw.data(), dims, dtype, opts, total);
        }
      }
    }
    std::fprintf(stderr, "trained on %s\n", path.c_str());
  }

  // Normalizing through ModelSet keeps the emitted table byte-identical to what
  // the codec would build from these counts at runtime.
  const ModelSet trained = ModelSet::from_counts(total.counts);

  std::FILE* f = std::fopen(out_path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
    return 1;
  }
  std::fprintf(f,
               "// Generated by tools/train_tables. Do not edit.\n"
               "//\n"
               "// Static entropy model frequencies, normalized to kProbScale (4096) and\n"
               "// measured on a real corpus rather than assumed. Regenerate with:\n"
               "//   train_tables --dims X,Y,Z --dtype u8 tables_v2.inc corpus/*.raw\n"
               "//\n"
               "// Changing this file changes how every archive coded against table\n"
               "// version 2 decodes, so it is a format-versioned artifact: add a new\n"
               "// version rather than editing this one in place.\n\n"
               "namespace {\n\n");

  for (std::uint16_t i = 0; i < ModelIndex::total; ++i) {
    const Model& m = trained[i];
    std::fprintf(f, "// model %u (%s), %u symbols\n", i, model_name(i), m.nsym());
    std::fprintf(f, "constexpr std::uint16_t kTrained%u[] = {", i);
    for (std::uint32_t v = 0; v < m.nsym(); ++v) {
      if (v % 16 == 0) std::fprintf(f, "\n    ");
      std::fprintf(f, "%u,", m.freq(v));
    }
    std::fprintf(f, "\n};\n\n");
  }

  std::fprintf(f, "constexpr std::span<const std::uint16_t> kTrainedTables[] = {\n");
  for (std::uint16_t i = 0; i < ModelIndex::total; ++i)
    std::fprintf(f, "    kTrained%u,\n", i);
  std::fprintf(f, "};\n\n}  // namespace\n");
  std::fclose(f);

  // Report where the corpus actually disagreed with the analytic priors, since
  // that is the interesting output: a model whose measured cost is far from the
  // prior's is one the prior was wrong about.
  const ModelSet priors = ModelSet::defaults(1);
  std::fprintf(stderr, "\n%-16s %6s %12s %12s %10s\n", "model", "idx", "prior bits",
               "trained bits", "delta");
  for (std::uint16_t i = 0; i < ModelIndex::total; ++i) {
    std::uint64_t n = 0;
    for (std::uint32_t c : total.counts[i]) n += c;
    if (n == 0) continue;
    double bp = 0, bt = 0;
    for (std::uint32_t v = 0; v < priors[i].nsym() && v < total.counts[i].size(); ++v) {
      const double w = static_cast<double>(total.counts[i][v]) / static_cast<double>(n);
      if (w > 0) {
        bp += w * priors[i].bit_cost(v);
        bt += w * trained[i].bit_cost(v);
      }
    }
    std::fprintf(stderr, "%-16s %6u %12.4f %12.4f %9.1f%%\n", model_name(i), i, bp, bt,
                 (bp > 0) ? 100.0 * (bt - bp) / bp : 0.0);
  }
  std::fprintf(stderr, "\nwrote %s\n", out_path.c_str());
  return 0;
}
