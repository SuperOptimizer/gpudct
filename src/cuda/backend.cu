// CUDA host integration (docs/ROADMAP.md M7).
//
// Drives the two kernels in kernels.cu and owns everything they need on the
// device: the flattened model tables, the archive bytes, the per-brick stream
// descriptors, and the sparse coefficient scratch.
//
// Bricks are processed in batches sized to a device-memory budget rather than
// all at once, because the sparse scratch is the dominant allocation and a
// whole-volume archive would not fit. Batching is also what lets this stream:
// each batch is independent.
//
// Not handled on the GPU, and deliberately: raw-mode bricks (a memcpy, not worth
// a kernel) and the bounded-error correction layer. An archive carrying
// corrections falls back to the CPU decoder, because shipping a bound we cannot
// verify on this path would be worse than being slower.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <atomic>
#include <thread>
#include <cstdlib>

#include <algorithm>
#include <cstring>
#include <vector>

#include "core/format.hpp"
#include "core/models.hpp"
#include "core/quant.hpp"
#include "core/brick_tables.hpp"
#include "core/rdo.hpp"
#include "gpudct/gpudct.hpp"
#include "gpudct/device_volume.hpp"
#include "cuda/cuda_backend.hpp"

namespace gpudct::cuda {

// Declared in kernels.cu.
struct DeviceModels;
struct BrickDesc;

}  // namespace gpudct::cuda

// Pull in the kernel definitions directly. One translation unit keeps the
// device-side structs and the host code that fills them impossible to get out of
// sync, which is the failure mode that matters here.
#include "kernels.cu"
#include "encode.cu"

namespace gpudct::cuda {
namespace {

using detail::FileHeader;
using detail::ModelSet;
using detail::QuantMatrix;

#define CUDA_OK(call)                                   \
  do {                                                  \
    const cudaError_t e_ = (call);                      \
    if (e_ != cudaSuccess) return Status::io_error;     \
  } while (0)

// Flattened, device-resident copy of the model set.
struct ModelUpload {
  std::uint32_t* fc = nullptr;
  std::uint8_t* slot = nullptr;
  std::uint32_t* offsets = nullptr;

  ~ModelUpload() {
    cudaFree(fc);
    cudaFree(slot);
    cudaFree(offsets);
  }
};

Status upload_models(const ModelSet& ms, ModelUpload& up, DeviceModels& dm) {
  const std::uint32_t n = detail::ModelIndex::total;
  std::vector<std::uint32_t> fc, offsets(n);
  std::vector<std::uint8_t> slot(static_cast<std::size_t>(n) * kProbScale);

  for (std::uint32_t i = 0; i < n; ++i) {
    offsets[i] = static_cast<std::uint32_t>(fc.size());
    const detail::Model& m = ms[static_cast<std::uint16_t>(i)];
    for (std::uint32_t v = 0; v < m.nsym(); ++v)
      fc.push_back((static_cast<std::uint32_t>(m.freq(v)) << 16) | m.cum(v));
    for (std::uint32_t s = 0; s < kProbScale; ++s)
      slot[static_cast<std::size_t>(i) * kProbScale + s] =
          static_cast<std::uint8_t>(m.sym_of_slot(s));
  }

  CUDA_OK(cudaMalloc(&up.fc, fc.size() * 4));
  CUDA_OK(cudaMalloc(&up.slot, slot.size()));
  CUDA_OK(cudaMalloc(&up.offsets, offsets.size() * 4));
  CUDA_OK(cudaMemcpy(up.fc, fc.data(), fc.size() * 4, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(up.slot, slot.data(), slot.size(), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(up.offsets, offsets.data(), offsets.size() * 4, cudaMemcpyHostToDevice));

  dm.fc = up.fc;
  dm.slot = up.slot;
  dm.offset_freq = up.offsets;
  dm.model_count = n;
  return Status::ok;
}

// Upload the transform constants the device IDCT reads from __constant__.
Status upload_transform_constants() {
  const auto& c = detail::dct_constants();
  CUDA_OK(cudaMemcpyToSymbol(c_alpha_inv, c.inv_alpha.data(), 16 * sizeof(float)));
  CUDA_OK(cudaMemcpyToSymbol(c_s16, c.s16.data(), 8 * sizeof(float)));
  CUDA_OK(cudaMemcpyToSymbol(c_s8, c.s8.data(), 4 * sizeof(float)));
  CUDA_OK(cudaMemcpyToSymbol(c_s4, c.s4.data(), 2 * sizeof(float)));
  CUDA_OK(cudaMemcpyToSymbol(c_s2, c.s2.data(), 1 * sizeof(float)));
  return Status::ok;
}

// Parses one brick payload's stream table. Returns false for raw-mode bricks and
// for anything malformed; both are handed back to the CPU path.
bool parse_brick(std::span<const std::uint8_t> payload, std::uint32_t P, BrickDesc& bd,
                 std::uint64_t device_base_offset, detail::BrickTableView& tables) {
  if (payload.empty() || payload[0] != 0) return false;
  if (P > 64) return false;
  if (payload.size() < 3) return false;

  std::size_t pos = 1;
  const std::uint8_t table_flag = payload[pos++];
  if (table_flag > 1) return false;
  bd.has_tables = table_flag;
  if (table_flag == 1) {
    if (pos + 4 > payload.size()) return false;
    const std::uint32_t blob_size = detail::get_u32(payload, pos);
    pos += 4;
    if (blob_size > payload.size() - pos) return false;
    if (!detail::parse_brick_tables(payload.subspan(pos, blob_size), tables)) return false;
    pos += blob_size;
  }
  if (payload.size() < pos + 4ull * P + 1) return false;
  std::uint64_t sum = 0;
  for (std::uint32_t s = 0; s < P; ++s) {
    bd.stream_size[s] = detail::get_u32(payload, pos);
    pos += 4;
    sum += bd.stream_size[s];
  }
  const std::uint8_t has_corr = payload[pos++];
  if (has_corr) return false;  // corrections are CPU-only for now
  if (pos + sum != payload.size()) return false;

  for (std::uint32_t s = 0; s < P; ++s) {
    bd.stream_offset[s] = static_cast<std::uint32_t>(pos);
    pos += bd.stream_size[s];
  }
  bd.payload = reinterpret_cast<const std::uint8_t*>(device_base_offset);
  bd.payload_size = static_cast<std::uint32_t>(payload.size());
  bd.streams = P;
  bd.mode = 0;
  return true;
}

}  // namespace

// Per-stage timing, off unless GPUDCT_CUDA_PROFILE is set. Optimizing the GPU
// path without knowing the K1/K2 split is guesswork, and the two stages have
// completely different bottlenecks -- K1 is latency-bound on a serial parse, K2
// is bandwidth- and occupancy-bound.
struct StageTimer {
  bool on = false;
  cudaEvent_t a{}, b{}, c{}, d{};
  float upload_ms = 0, k1_ms = 0, k2_ms = 0, download_ms = 0;
  // Host-side wall clock and an explicit unaccounted remainder.
  //
  // The kernel timers alone covered 40% of decode and 32% of encode, and the
  // silent remainder is where the real cost turned out to live -- it is what
  // sent three previous optimizations at the wrong stage (see docs/QUALITY.md).
  // A profile that cannot be checked against its own total is not a profile.
  std::chrono::steady_clock::time_point t_start{};
  double host_ms = 0;

  StageTimer() {
    on = std::getenv("GPUDCT_CUDA_PROFILE") != nullptr;
    t_start = std::chrono::steady_clock::now();
    if (!on) return;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventCreate(&c);
    cudaEventCreate(&d);
  }
  ~StageTimer() {
    if (!on) return;
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaEventDestroy(c);
    cudaEventDestroy(d);
    const double total =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() *
        1000.0;
    std::fprintf(stderr,
                 "[cuda] k1 (entropy) %8.2f ms   k2 (transform) %8.2f ms   "
                 "readback wait %8.2f ms\n"
                 "[cuda] host prep %8.2f ms   total %8.2f ms   unaccounted %8.2f ms\n",
                 k1_ms, k2_ms, download_ms, host_ms, total,
                 total - k1_ms - k2_ms - download_ms - host_ms);
  }
  // Wall-clock span for host work, which CUDA events cannot see.
  struct HostSpan {
    StageTimer& t;
    std::chrono::steady_clock::time_point t0;
    bool running = true;
    explicit HostSpan(StageTimer& s) : t(s), t0(std::chrono::steady_clock::now()) {}
    void stop() {
      if (!running) return;
      running = false;
      if (t.on)
        t.host_ms +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1000.0;
    }
    ~HostSpan() { stop(); }
  };
  // The stream matters. Kernels run on the decoder's compute stream, so an
  // event recorded on the null stream sits in a different queue and measures
  // nothing -- it reported 0.01 ms for kernels that plainly take tens of
  // milliseconds, and quietly moved their cost into `unaccounted`.
  void mark(cudaEvent_t e, cudaStream_t s = nullptr) {
    if (on) cudaEventRecord(e, s);
  }
  void accumulate(float& into, cudaEvent_t from, cudaEvent_t to) {
    if (!on) return;
    cudaEventSynchronize(to);
    float ms = 0;
    cudaEventElapsedTime(&ms, from, to);
    into += ms;
  }
};

// Touches every page of a buffer, in parallel, so it is resident before the
// driver tries to pin it.
//
// This is not optional bookkeeping. Registering a gigabyte of never-touched
// pages costs ~255 ms and leaves the subsequent DMA faulting its way through the
// buffer at a quarter speed (428 ms against 93). The cost is the same first
// touch either way; doing it here, across threads, is simply the cheapest place
// to pay it. It was previously paid by accident, single-threaded, by the
// zero-fill of the output vector.
void prefault(std::uint8_t* p, std::size_t n) {
  if (n == 0) return;
  constexpr std::size_t kPage = 4096;
  unsigned t = std::thread::hardware_concurrency();
  if (t == 0) t = 4;
  t = std::min<unsigned>(t, 16);
  const std::size_t span = (n + t - 1) / t;

  std::vector<std::thread> pool;
  pool.reserve(t);
  for (unsigned i = 0; i < t; ++i) {
    const std::size_t lo = i * span;
    if (lo >= n) break;
    const std::size_t hi = std::min(n, lo + span);
    pool.emplace_back([p, lo, hi] {
      for (std::size_t off = lo; off < hi; off += kPage) p[off] = 0;
      p[hi - 1] = 0;
    });
  }
  for (auto& th : pool) th.join();
}

bool available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Whole-archive decode. Returns Status::backend_unavailable when the archive
// uses a feature this path does not implement, so the caller can fall back
// rather than get a wrong answer.
namespace {

// Distinguishes "the sparse buffer was too small" from a genuinely corrupt
// stream, so the retry above only fires for the former.
constexpr std::uint32_t kErrCorrupt = 1;
constexpr std::uint32_t kErrOverflow = 2;

// How many batches to split a whole-volume decode into, so the readback of one
// slab overlaps the decode of the next. Swept on a 1 GiB volume; see
// docs/QUALITY.md.
constexpr int kDecodeBatchSplit = 4;

Status decode_archive_attempt(std::span<const std::uint8_t> archive, const FileHeader& h,
                              const QuantMatrix& qm, const ModelSet& ms,
                              std::uint32_t attempt_max_nz, std::span<std::uint8_t> out,
                              bool& overflowed) {
  overflowed = false;
  if (!available()) return Status::backend_unavailable;
  if (h.flags & detail::kFlagCorrections) return Status::backend_unavailable;

  ModelUpload up;
  DeviceModels dm{};
  if (const Status s = upload_models(ms, up, dm); s != Status::ok) return s;
  if (const Status s = upload_transform_constants(); s != Status::ok) return s;

  // The archive is uploaded once; brick descriptors point into it.
  std::uint8_t* d_archive = nullptr;
  CUDA_OK(cudaMalloc(&d_archive, archive.size()));
  CUDA_OK(cudaMemcpy(d_archive, archive.data(), archive.size(), cudaMemcpyHostToDevice));

  float* d_quant = nullptr;
  CUDA_OK(cudaMalloc(&d_quant, kChunkVox * sizeof(float)));
  CUDA_OK(cudaMemcpy(d_quant, qm.q.data(), kChunkVox * sizeof(float), cudaMemcpyHostToDevice));

  const std::size_t esz = dtype_size(h.dtype);
  const std::size_t brick_vox = static_cast<std::size_t>(kBrickDim) * kBrickDim * kBrickDim;

  // Sparse-scratch capacity per chunk.
  //
  // The worst case is 4096 -- every coefficient nonzero -- but real chunks hold
  // a few dozen to a few hundred at any useful rate, and sizing for the worst
  // case meant allocating over half a gigabyte of device memory that was never
  // touched. cudaMalloc of that size cost more than the kernels it fed.
  //
  // So the buffer is sized for the realistic case and K1 reports an overflow
  // instead of writing past the end; `decode_archive` retries at full capacity.
  // A first pass that overflows costs one wasted launch, which is far cheaper
  // than the allocation it avoids on every normal decode.
  const std::uint32_t max_nz = attempt_max_nz;
  const std::size_t per_brick =
      static_cast<std::size_t>(kChunksPerBrick) * max_nz * (2 + 2) +
      static_cast<std::size_t>(kChunksPerBrick) * 4 + brick_vox * esz +
      sizeof(BrickDesc);
  std::size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  const std::size_t budget = (free_mem > (512u << 20)) ? (free_mem - (512u << 20)) : (256u << 20);
  std::size_t batch = std::max<std::size_t>(1, budget / per_brick);
  batch = std::min<std::size_t>(batch, static_cast<std::size_t>(h.brick_count));

  // Round the batch down to a whole z-layer of bricks.
  //
  // Bricks are indexed x fastest, so a run of grid.x * grid.y consecutive
  // bricks is exactly one 128-voxel z-slab of the volume -- and in the volume's
  // own layout that slab is one contiguous byte range. That is what makes a
  // per-batch readback a single DMA instead of a scatter, which in turn is what
  // lets the readback overlap the next batch's kernels. Without the rounding a
  // batch straddles slabs and the copy would have to be split per brick row.
  const Dims grid = brick_grid(h.dims);
  const std::size_t bricks_per_layer = static_cast<std::size_t>(grid.x) * grid.y;
  const bool slab_pipelined = batch >= bricks_per_layer;
  if (slab_pipelined) batch -= batch % bricks_per_layer;

  // Deliberately split the work into several batches even when it would all
  // fit at once. Sizing the batch purely by what device memory allows produces
  // exactly one batch for any volume that fits, and a single batch has nothing
  // to overlap: the readback is issued after the last kernel and simply waited
  // on. Splitting costs a little per-batch launch and table-upload overhead and
  // buys the readback of slab k running underneath the decode of slab k+1.
  if (slab_pipelined) {
    static const int split = [] {
      const char* e = std::getenv("GPUDCT_BATCH_SPLIT");
      const int v = e ? std::atoi(e) : kDecodeBatchSplit;
      return v > 0 ? v : kDecodeBatchSplit;
    }();
    const std::size_t layers = static_cast<std::size_t>(grid.z);
    const std::size_t want = std::max<std::size_t>(1, layers / static_cast<std::size_t>(split));
    batch = std::min(batch, want * bricks_per_layer);
  }

  // Pinned staging for the readback. Pageable memory forces the driver to stage
  // through its own pinned buffer, which roughly halves achievable PCIe
  // bandwidth -- and the readback measured comparable to the entropy kernel, so
  // it is worth the pinned allocation.
  std::uint8_t* host_bricks = nullptr;
  const std::size_t host_bytes = batch * brick_vox * esz;
  bool host_pinned = cudaMallocHost(&host_bricks, host_bytes) == cudaSuccess;
  std::vector<std::uint8_t> host_fallback;
  if (!host_pinned) {
    host_fallback.resize(host_bytes);
    host_bricks = host_fallback.data();
  }

  std::int16_t* d_levels = nullptr;
  std::uint16_t* d_indices = nullptr;
  std::uint32_t* d_counts = nullptr;
  std::uint32_t* d_offsets = nullptr;
  std::uint8_t* d_packed = nullptr;
  std::uint8_t* d_out = nullptr;
  BrickDesc* d_descs = nullptr;
  std::uint32_t* d_error = nullptr;
  // Whole output volume on device, when it fits. This is what lets K2 write
  // final coordinates and turns the readback into a single contiguous copy.
  std::uint8_t* d_volume = nullptr;
  std::uint32_t* d_origins = nullptr;
  std::uint16_t* d_tab_freq = nullptr;
  std::uint32_t* d_tab_fc = nullptr;
  std::uint8_t* d_tab_slot = nullptr;
  std::uint8_t* d_tab_map = nullptr;
  std::uint32_t* d_tab_base = nullptr;
  std::uint32_t* d_tab_count = nullptr;

  // Two non-blocking streams. Non-blocking matters: a stream created with the
  // default flags implicitly synchronizes with the null stream, and the small
  // scratch uploads still go there, so blocking streams would serialize the
  // readback right back against the kernels it is meant to overlap.
  cudaStream_t s_compute = nullptr, s_copy = nullptr;
  const bool streams_ok =
      cudaStreamCreateWithFlags(&s_compute, cudaStreamNonBlocking) == cudaSuccess &&
      cudaStreamCreateWithFlags(&s_copy, cudaStreamNonBlocking) == cudaSuccess;
  bool out_registered = false;

  auto cleanup = [&] {
    if (s_copy) cudaStreamSynchronize(s_copy);
    if (out_registered) cudaHostUnregister(out.data());
    if (s_compute) cudaStreamDestroy(s_compute);
    if (s_copy) cudaStreamDestroy(s_copy);
    if (host_pinned && host_bricks) cudaFreeHost(host_bricks);
    cudaFree(d_levels);
    cudaFree(d_indices);
    cudaFree(d_counts);
    cudaFree(d_offsets);
    cudaFree(d_packed);
    cudaFree(d_out);
    cudaFree(d_descs);
    cudaFree(d_error);
    cudaFree(d_volume);
    cudaFree(d_origins);
    cudaFree(d_tab_freq);
    cudaFree(d_tab_fc);
    cudaFree(d_tab_slot);
    cudaFree(d_tab_map);
    cudaFree(d_tab_base);
    cudaFree(d_tab_count);
    cudaFree(d_quant);
    cudaFree(d_archive);
  };

  const std::size_t volume_bytes = h.dims.voxels() * esz;
  const bool direct = cudaMalloc(&d_volume, volume_bytes) == cudaSuccess &&
                      cudaMalloc(&d_origins, batch * 3 * sizeof(std::uint32_t)) == cudaSuccess;
  if (!direct) {
    cudaFree(d_volume);
    cudaFree(d_origins);
    d_volume = nullptr;
    d_origins = nullptr;
  }

  if (cudaMalloc(&d_levels, batch * kChunksPerBrick * max_nz * 2) != cudaSuccess ||
      cudaMalloc(&d_indices, batch * kChunksPerBrick * max_nz * 2) != cudaSuccess ||
      cudaMalloc(&d_counts, batch * kChunksPerBrick * 4) != cudaSuccess ||
      (!direct && cudaMalloc(&d_out, batch * brick_vox * esz) != cudaSuccess) ||
      cudaMalloc(&d_descs, batch * sizeof(BrickDesc)) != cudaSuccess ||
      cudaMalloc(&d_error, 4) != cudaSuccess ||
      cudaMalloc(&d_tab_freq, batch * detail::kMaxBrickTables * 256 * 2) != cudaSuccess ||
      cudaMalloc(&d_tab_fc, batch * detail::kMaxBrickTables * 256 * 4) != cudaSuccess ||
      cudaMalloc(&d_tab_slot, batch * detail::kMaxBrickTables * kProbScale) != cudaSuccess ||
      cudaMalloc(&d_tab_map, batch * detail::ModelIndex::total) != cudaSuccess ||
      cudaMalloc(&d_tab_base, batch * 4) != cudaSuccess ||
      cudaMalloc(&d_tab_count, batch * 4) != cudaSuccess) {
    cleanup();
    return Status::out_of_memory;
  }

  if (out.size() != h.dims.voxels() * esz) return Status::invalid_argument;
  StageTimer timer;
  std::vector<BrickDesc> descs(batch);
  std::vector<std::uint32_t> origins(batch * 3);
  std::vector<std::uint16_t> tab_freq;
  std::vector<std::uint8_t> tab_map;
  std::vector<std::uint32_t> tab_base;
  std::vector<std::uint32_t> tab_count;
  tab_freq.reserve(batch * detail::kMaxBrickTables * 256);

  const float lo = dtype_min(h.dtype), hi = dtype_max(h.dtype);
  const std::int32_t max_level = detail::max_plausible_level(qm);

  // Per-block shared memory available for the low-latency K1 launch. The default
  // cap is 48 KB; a brick's slot tables can want more than that, so the opt-in
  // limit is requested once and the launch falls back to the flat kernel if the
  // device will not give it.
  std::size_t shared_limit = 48u << 10;
  {
    int dev = 0, per_block_optin = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) ==
            cudaSuccess &&
        per_block_optin > 0) {
      if (cudaFuncSetAttribute(k1_entropy_decode_shared,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               per_block_optin) == cudaSuccess)
        shared_limit = static_cast<std::size_t>(per_block_optin);
    }
  }

  // Pipelined readback: each batch's slab is copied out on `s_copy` while the
  // next batch is prepared and decoded on `s_compute`. Measured on a warm 1 GiB
  // decode the readback and the compute side were 140 ms and 134 ms, so almost
  // the whole copy hides behind work that has to happen anyway.
  //
  // The destination is pinned once here rather than around the final copy.
  // cudaMemcpyAsync into pageable memory is not actually asynchronous, so
  // without this the pipeline would silently degrade to the serial behaviour.
  const bool pipelined = direct && slab_pipelined && streams_ok;
  if (pipelined) {
    prefault(out.data(), volume_bytes);
    out_registered =
        cudaHostRegister(out.data(), volume_bytes, cudaHostRegisterDefault) == cudaSuccess;
  }
  const cudaStream_t ks = streams_ok ? s_compute : cudaStream_t{nullptr};
  static const bool profile = std::getenv("GPUDCT_CUDA_PROFILE") != nullptr;
  if (profile)
    std::fprintf(stderr,
                 "[cuda] pipelined=%d direct=%d streams=%d registered=%d batch=%zu "
                 "layer=%zu batches=%llu\n",
                 static_cast<int>(pipelined), static_cast<int>(direct),
                 static_cast<int>(streams_ok), static_cast<int>(out_registered), batch,
                 bricks_per_layer,
                 static_cast<unsigned long long>((h.brick_count + batch - 1) / batch));

  for (std::uint64_t b0 = 0; b0 < h.brick_count; b0 += batch) {
    const std::uint32_t n = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(batch, h.brick_count - b0));

    StageTimer::HostSpan prep(timer);

    tab_freq.clear();
    tab_base.assign(n, 0);
    tab_count.assign(n, 0);
    std::uint32_t max_tab_count = 0;
    tab_map.assign(static_cast<std::size_t>(n) * detail::ModelIndex::total,
                   detail::kUseGlobalTable);

    // Any brick this path cannot handle aborts the whole decode, so the caller
    // falls back rather than getting a partially GPU-decoded volume.
    for (std::uint32_t i = 0; i < n; ++i) {
      const std::size_t e =
          static_cast<std::size_t>(h.index_offset + (b0 + i) * detail::BrickEntry::kSize);
      const std::uint64_t off = detail::get_u64(archive, e);
      const std::uint32_t size = detail::get_u32(archive, e + 8);
      if (off > archive.size() || size > archive.size() - off) {
        cleanup();
        return Status::corrupt_bitstream;
      }
      detail::BrickTableView tv;
      if (!parse_brick(archive.subspan(static_cast<std::size_t>(off), size),
                       h.streams_per_brick, descs[i],
                       reinterpret_cast<std::uint64_t>(d_archive) + off, tv)) {
        cleanup();
        return Status::backend_unavailable;
      }
      tab_base[i] = static_cast<std::uint32_t>(tab_freq.size() / 256);
      tab_count[i] = 0;
      if (descs[i].has_tables) {
        tab_count[i] = static_cast<std::uint32_t>(tv.freqs.size() / 256);
        max_tab_count = std::max(max_tab_count, tab_count[i]);
        tab_freq.insert(tab_freq.end(), tv.freqs.begin(), tv.freqs.end());
        std::memcpy(tab_map.data() + static_cast<std::size_t>(i) * detail::ModelIndex::total,
                    tv.model_map.data(), detail::ModelIndex::total);
      } else {
        std::memset(tab_map.data() + static_cast<std::size_t>(i) * detail::ModelIndex::total,
                    detail::kUseGlobalTable, detail::ModelIndex::total);
      }
      if (direct) {
        const std::uint64_t bi = b0 + i;
        origins[i * 3 + 0] = static_cast<std::uint32_t>(bi % grid.x) * kBrickDim;
        origins[i * 3 + 1] = static_cast<std::uint32_t>((bi / grid.x) % grid.y) * kBrickDim;
        origins[i * 3 + 2] =
            static_cast<std::uint32_t>(bi / (static_cast<std::uint64_t>(grid.x) * grid.y)) *
            kBrickDim;
      }
    }
    if (direct &&
        cudaMemcpy(d_origins, origins.data(), n * 3 * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      cleanup();
      return Status::io_error;
    }

    // Upload the compact frequency tables and expand them on the device.
    const std::uint32_t ntab = static_cast<std::uint32_t>(tab_freq.size() / 256);
    if (ntab > 0) {
      if (cudaMemcpy(d_tab_freq, tab_freq.data(), tab_freq.size() * 2,
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        cleanup();
        return Status::io_error;
      }
      k0_build_brick_tables<<<ntab, 256, 0, ks>>>(d_tab_freq, d_tab_fc, d_tab_slot, ntab);
    }
    if (cudaMemcpy(d_tab_map, tab_map.data(), tab_map.size(), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(d_tab_base, tab_base.data(), n * 4, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(d_tab_count, tab_count.data(), n * 4, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(d_descs, descs.data(), n * sizeof(BrickDesc), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemset(d_error, 0, 4) != cudaSuccess ||
        cudaMemset(d_counts, 0, static_cast<std::size_t>(n) * kChunksPerBrick * 4) !=
            cudaSuccess) {
      cleanup();
      return Status::io_error;
    }

    prep.stop();
    timer.mark(timer.a, ks);
    {
      DeviceBrickTables bt{d_tab_fc, d_tab_slot, d_tab_map, d_tab_base};
      // Two launches, chosen by what limits this batch.
      //
      // The flat one maximizes occupancy and wins when there are enough bricks
      // to saturate the device. The per-brick one gives each block its own
      // shared-memory copy of the brick's slot tables, which is worth far more
      // when a batch is small: with a handful of bricks there are only a few
      // hundred threads and nothing hides the slot lookup's global latency.
      //
      // The crossover is on brick count, and P has to be at least a warp for the
      // per-brick launch not to waste most of every block.
      const std::size_t shared_bytes =
          static_cast<std::size_t>(max_tab_count) * kProbScale +
          static_cast<std::size_t>(h.streams_per_brick) * kRansStates * 4 + 4;
      static const bool no_shared = std::getenv("GPUDCT_NO_SHARED_K1") != nullptr;
      const bool use_shared = !no_shared && n <= kSharedLaunchMaxBricks &&
                              h.streams_per_brick >= 32 &&
                              shared_bytes <= shared_limit;
      if (use_shared) {
        k1_entropy_decode_shared<<<n, h.streams_per_brick, shared_bytes, ks>>>(
            d_descs, dm, bt, d_tab_count, max_level, d_levels, d_indices, d_counts, max_nz, n,
            d_error);
      } else {
        const std::uint32_t total_threads = n * h.streams_per_brick;
        const std::uint32_t blocks = (total_threads + kK1Threads - 1) / kK1Threads;
        k1_entropy_decode<<<blocks, kK1Threads, 0, ks>>>(d_descs, dm, bt, max_level, d_levels,
                                                  d_indices, d_counts, max_nz,
                                                  h.streams_per_brick, n, d_error);
      }
    }

    timer.mark(timer.b, ks);
    if (direct) {
      const dim3 blocks(n * kChunksPerBrick);
#define GPUDCT_K2_DIRECT(T)                                                                  \
  k2_inverse_transform_direct<T><<<blocks, 256, 0, ks>>>(                                           \
      d_levels, d_indices, d_counts, max_nz, d_quant, h.data_scale, h.data_offset, lo, hi,   \
      d_origins, h.dims.x, h.dims.y, h.dims.z, reinterpret_cast<T*>(d_volume))
      switch (h.dtype) {
        case DType::u8:  GPUDCT_K2_DIRECT(std::uint8_t); break;
        case DType::s8:  GPUDCT_K2_DIRECT(std::int8_t); break;
        case DType::u16: GPUDCT_K2_DIRECT(std::uint16_t); break;
        case DType::s16: GPUDCT_K2_DIRECT(std::int16_t); break;
        case DType::u32: GPUDCT_K2_DIRECT(std::uint32_t); break;
        case DType::s32: GPUDCT_K2_DIRECT(std::int32_t); break;
        case DType::f32: GPUDCT_K2_DIRECT(float); break;
      }
#undef GPUDCT_K2_DIRECT
    } else {
      const dim3 blocks(n * kChunksPerBrick);
      switch (h.dtype) {
        case DType::u8:
          k2_inverse_transform<std::uint8_t><<<blocks, 256, 0, ks>>>(
              d_levels, d_indices, d_counts, max_nz, d_quant, h.data_scale, h.data_offset, lo,
              hi, reinterpret_cast<std::uint8_t*>(d_out));
          break;
        case DType::s8:
          k2_inverse_transform<std::int8_t><<<blocks, 256, 0, ks>>>(
              d_levels, d_indices, d_counts, max_nz, d_quant, h.data_scale, h.data_offset, lo,
              hi, reinterpret_cast<std::int8_t*>(d_out));
          break;
        case DType::u16:
          k2_inverse_transform<std::uint16_t><<<blocks, 256, 0, ks>>>(
              d_levels, d_indices, d_counts, max_nz, d_quant, h.data_scale, h.data_offset, lo,
              hi, reinterpret_cast<std::uint16_t*>(d_out));
          break;
        case DType::s16:
          k2_inverse_transform<std::int16_t><<<blocks, 256, 0, ks>>>(
              d_levels, d_indices, d_counts, max_nz, d_quant, h.data_scale, h.data_offset, lo,
              hi, reinterpret_cast<std::int16_t*>(d_out));
          break;
        case DType::u32:
          k2_inverse_transform<std::uint32_t><<<blocks, 256, 0, ks>>>(
              d_levels, d_indices, d_counts, max_nz, d_quant, h.data_scale, h.data_offset, lo,
              hi, reinterpret_cast<std::uint32_t*>(d_out));
          break;
        case DType::s32:
          k2_inverse_transform<std::int32_t><<<blocks, 256, 0, ks>>>(
              d_levels, d_indices, d_counts, max_nz, d_quant, h.data_scale, h.data_offset, lo,
              hi, reinterpret_cast<std::int32_t*>(d_out));
          break;
        case DType::f32:
          k2_inverse_transform<float><<<blocks, 256, 0, ks>>>(
              d_levels, d_indices, d_counts, max_nz, d_quant, h.data_scale, h.data_offset, lo,
              hi, reinterpret_cast<float*>(d_out));
          break;
      }
    }

    timer.mark(timer.c, ks);
    // Only the compute stream: waiting on the whole device here would also wait
    // on the previous batch's readback, which is exactly the overlap we want.
    // The scratch buffers are safe to reuse once these kernels have retired.
    if ((streams_ok ? cudaStreamSynchronize(s_compute) : cudaDeviceSynchronize()) !=
        cudaSuccess) {
      cleanup();
      return Status::io_error;
    }
    std::uint32_t err = 0;
    cudaMemcpy(&err, d_error, 4, cudaMemcpyDeviceToHost);
    if (err == kErrOverflow) {
      overflowed = true;
      cleanup();
      return Status::corrupt_bitstream;  // caller retries at full capacity
    }
    if (err) {
      cleanup();
      return Status::corrupt_bitstream;
    }

    if (direct) {
      if (pipelined) {
        // This batch is a whole number of z-slabs, so its output is one
        // contiguous range of both d_volume and out.
        const std::uint64_t z0 = (b0 / bricks_per_layer) * kBrickDim;
        const std::uint64_t z1 =
            std::min<std::uint64_t>(h.dims.z, z0 + (n / bricks_per_layer) * kBrickDim);
        const std::size_t plane = static_cast<std::size_t>(h.dims.y) * h.dims.x * esz;
        const std::size_t off = static_cast<std::size_t>(z0) * plane;
        const std::size_t len = static_cast<std::size_t>(z1 - z0) * plane;
        if (cudaMemcpyAsync(out.data() + off, d_volume + off, len, cudaMemcpyDeviceToHost,
                            s_copy) != cudaSuccess) {
          cleanup();
          return Status::io_error;
        }
        // Flush the submission. Under WDDM the driver batches commands in user
        // space and does not hand them to the GPU until something forces it, so
        // without this the whole point of the copy stream is lost: the copies
        // sit unsubmitted and all run at the final synchronize. cudaStreamQuery
        // is the documented way to push the batch without blocking.
        (void)cudaStreamQuery(s_copy);
      }
      timer.mark(timer.d);
      timer.accumulate(timer.k1_ms, timer.a, timer.b);
      timer.accumulate(timer.k2_ms, timer.b, timer.c);
      continue;  // the volume stays on device until every batch is done
    }

    if (cudaMemcpy(host_bricks, d_out, static_cast<std::size_t>(n) * brick_vox * esz,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      cleanup();
      return Status::io_error;
    }
    timer.mark(timer.d);
    timer.accumulate(timer.k1_ms, timer.a, timer.b);
    timer.accumulate(timer.k2_ms, timer.b, timer.c);
    timer.accumulate(timer.download_ms, timer.c, timer.d);

    // Crop each padded brick into the output volume.
    for (std::uint32_t i = 0; i < n; ++i) {
      const std::uint64_t bi = b0 + i;
      const std::uint32_t bx = static_cast<std::uint32_t>(bi % grid.x);
      const std::uint32_t by = static_cast<std::uint32_t>((bi / grid.x) % grid.y);
      const std::uint32_t bz =
          static_cast<std::uint32_t>(bi / (static_cast<std::uint64_t>(grid.x) * grid.y));
      const std::uint8_t* src = host_bricks + static_cast<std::size_t>(i) * brick_vox * esz;

      const std::uint32_t nx = std::min<std::uint32_t>(kBrickDim, h.dims.x - bx * kBrickDim);
      const std::uint32_t ny = std::min<std::uint32_t>(kBrickDim, h.dims.y - by * kBrickDim);
      const std::uint32_t nz = std::min<std::uint32_t>(kBrickDim, h.dims.z - bz * kBrickDim);
      for (std::uint32_t z = 0; z < nz; ++z)
        for (std::uint32_t y = 0; y < ny; ++y) {
          const std::size_t so =
              ((static_cast<std::size_t>(z) * kBrickDim) + y) * kBrickDim;
          const std::size_t doff =
              ((static_cast<std::size_t>(bz * kBrickDim + z) * h.dims.y + (by * kBrickDim + y)) *
                   h.dims.x +
               bx * kBrickDim);
          std::memcpy(out.data() + doff * esz, src + so * esz, nx * esz);
        }
    }
  }

  if (direct && pipelined) {
    timer.mark(timer.c);
    const cudaError_t copy = cudaStreamSynchronize(s_copy);
    timer.mark(timer.d);
    timer.accumulate(timer.download_ms, timer.c, timer.d);
    if (copy != cudaSuccess) {
      cleanup();
      return Status::io_error;
    }
  } else if (direct) {
    timer.mark(timer.c);
    // Pin the destination in place rather than staging through a separate
    // pinned buffer: a D2H copy into pageable memory runs at roughly half rate
    // (measured 25 ms vs 6 ms for 134 MB), and staging would cost a full extra
    // host-side memcpy of the whole volume to avoid it.
    const auto t_reg = std::chrono::steady_clock::now();
    prefault(out.data(), volume_bytes);
    const bool registered =
        cudaHostRegister(out.data(), volume_bytes, cudaHostRegisterDefault) == cudaSuccess;
    const double reg_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_reg).count();
    const cudaError_t copy = cudaMemcpy(out.data(), d_volume, volume_bytes,
                                        cudaMemcpyDeviceToHost);
    if (registered) cudaHostUnregister(out.data());
    if (std::getenv("GPUDCT_CUDA_PROFILE"))
      std::fprintf(stderr, "[cuda] prefault+pin for readback %8.2f ms\n", reg_s * 1000.0);
    if (copy != cudaSuccess) {
      cleanup();
      return Status::io_error;
    }
    timer.mark(timer.d);
    timer.accumulate(timer.download_ms, timer.c, timer.d);
  }

  cleanup();
  return Status::ok;
}

}  // namespace

// Typical chunk occupancy at the rates this codec targets, with headroom. A
// chunk denser than this is legal, just rare enough not to size every buffer for.
constexpr std::uint32_t kTypicalMaxNonzero = 768;

Status decode_archive(std::span<const std::uint8_t> archive, const FileHeader& h,
                      const QuantMatrix& qm, const ModelSet& ms, std::span<std::uint8_t> out) {
  bool overflowed = false;
  const Status s =
      decode_archive_attempt(archive, h, qm, ms, kTypicalMaxNonzero, out, overflowed);
  if (!overflowed) return s;
  return decode_archive_attempt(archive, h, qm, ms, kChunkVox, out, overflowed);
}


// --------------------------------------------------------------------------
// Encode
// --------------------------------------------------------------------------

namespace {

struct EncModelUpload {
  DeviceEncSymbol* sym = nullptr;
  std::uint32_t* offset = nullptr;
  ~EncModelUpload() {
    cudaFree(sym);
    cudaFree(offset);
  }
};

Status upload_enc_models(const ModelSet& ms, EncModelUpload& up, DeviceEncModels& dm) {
  const std::uint32_t n = detail::ModelIndex::total;
  std::vector<DeviceEncSymbol> sym;
  std::vector<std::uint32_t> offset(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    offset[i] = static_cast<std::uint32_t>(sym.size());
    const detail::Model& m = ms[static_cast<std::uint16_t>(i)];
    for (std::uint32_t v = 0; v < m.nsym(); ++v) {
      const detail::Model::EncSymbol& e = m.enc(v);
      sym.push_back({e.rcp_freq, e.freq, e.bias, e.cmpl_freq, e.rcp_shift});
    }
  }
  CUDA_OK(cudaMalloc(&up.sym, sym.size() * sizeof(DeviceEncSymbol)));
  CUDA_OK(cudaMalloc(&up.offset, offset.size() * 4));
  CUDA_OK(cudaMemcpy(up.sym, sym.data(), sym.size() * sizeof(DeviceEncSymbol),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(up.offset, offset.data(), offset.size() * 4, cudaMemcpyHostToDevice));
  dm.sym = up.sym;
  dm.offset = up.offset;
  dm.brick_sym = nullptr;
  dm.brick_map = nullptr;
  dm.brick_base = nullptr;
  return Status::ok;
}

// Symbol scratch per chunk, tried in increasing order.
//
// This buffer dominates device memory: at 512 bricks in flight, a 6144-symbol
// capacity is 6.4 GB, and allocating that cost an order of magnitude more than
// the kernels it fed. A typical chunk at 20:1 emits on the order of a thousand
// symbols, so the first attempt is sized for that and the encoder retries larger
// only when a chunk actually overflows. The true worst case -- every coefficient
// nonzero and escaped -- is never allocated for.
constexpr int kEncSymCapacities[] = {1536, 6144, 24576};

}  // namespace

namespace {

Status encode_bricks_attempt(const void* data, Dims dims, DType dtype, float scale,
                             float offset, const QuantMatrix& qm, float deadzone,
                             const ModelSet& ms, std::uint32_t P, int kEncSymCapacity,
                             bool rdo, bool per_brick_tables,
                             std::vector<std::vector<std::uint8_t>>& payloads,
                             bool& overflowed) {
  overflowed = false;
  if (!available()) return Status::backend_unavailable;

  EncModelUpload up;
  DeviceEncModels dm{};
  if (const Status st = upload_enc_models(ms, up, dm); st != Status::ok) return st;
  if (rdo) {
    // Precompute the rate estimates the device RDO needs, using the same
    // Model::bit_cost the CPU version calls, so the two make identical
    // decisions wherever float arithmetic allows.
    std::vector<float> bits(
        static_cast<std::size_t>(kNumBandsDev) * kLevelCtxDev * kRdoMagTable, 0.0f);
    std::vector<float> dc_bits(kRdoMagTable, 0.0f);
    std::vector<float> overhead(kNumBandsDev, 0.0f);
    for (int b = 0; b < kNumBandsDev; ++b) {
      overhead[b] = detail::subblock_overhead_bits(ms, b);
      for (int c = 0; c < kLevelCtxDev; ++c)
        for (int m = 1; m < kRdoMagTable; ++m)
          bits[(static_cast<std::size_t>(b) * kLevelCtxDev + c) * kRdoMagTable + m] =
              detail::magnitude_bits(ms, b, c, m, false);
    }
    for (int m = 1; m < kRdoMagTable; ++m)
      dc_bits[m] = detail::magnitude_bits(ms, 0, 0, m, true);
    const float lam = detail::rdo_lambda_scale();

    CUDA_OK(cudaMemcpyToSymbol(c_rdo_bits, bits.data(), bits.size() * sizeof(float)));
    CUDA_OK(cudaMemcpyToSymbol(c_rdo_dc_bits, dc_bits.data(), dc_bits.size() * sizeof(float)));
    CUDA_OK(cudaMemcpyToSymbol(c_rdo_sub_overhead, overhead.data(),
                               overhead.size() * sizeof(float)));
    CUDA_OK(cudaMemcpyToSymbol(c_rdo_lambda_scale, &lam, sizeof(float)));
  }
  {
    const auto& c = detail::dct_constants();
    CUDA_OK(cudaMemcpyToSymbol(c_alpha, c.alpha.data(), 16 * sizeof(float)));
    CUDA_OK(cudaMemcpyToSymbol(c_s16, c.s16.data(), 8 * sizeof(float)));
    CUDA_OK(cudaMemcpyToSymbol(c_s8, c.s8.data(), 4 * sizeof(float)));
    CUDA_OK(cudaMemcpyToSymbol(c_s4, c.s4.data(), 2 * sizeof(float)));
    CUDA_OK(cudaMemcpyToSymbol(c_s2, c.s2.data(), 1 * sizeof(float)));
  }

  const std::size_t esz = dtype_size(dtype);
  const Dims grid = brick_grid(dims);
  const std::uint64_t brick_count =
      static_cast<std::uint64_t>(grid.x) * grid.y * grid.z;
  payloads.assign(static_cast<std::size_t>(brick_count), {});

  std::uint8_t* d_volume = nullptr;
  float* d_rquant = nullptr;
  CUDA_OK(cudaMalloc(&d_volume, dims.voxels() * esz));
  {
    // Pin the source in place for the upload. A pageable H2D copy runs at about
    // half rate, and at these volume sizes the upload is a large fraction of
    // encode time -- 150 ms of 600 for a gigabyte.
    void* src = const_cast<void*>(data);
    const std::size_t bytes = dims.voxels() * esz;
    const bool registered =
        cudaHostRegister(src, bytes, cudaHostRegisterReadOnly) == cudaSuccess;
    const cudaError_t e = cudaMemcpy(d_volume, data, bytes, cudaMemcpyHostToDevice);
    if (registered) cudaHostUnregister(src);
    if (e != cudaSuccess) {
      cudaFree(d_volume);
      return Status::io_error;
    }
  }
  CUDA_OK(cudaMalloc(&d_rquant, kChunkVox * sizeof(float)));
  CUDA_OK(cudaMemcpy(d_rquant, qm.rq.data(), kChunkVox * sizeof(float),
                     cudaMemcpyHostToDevice));
  float* d_quant = nullptr;
  CUDA_OK(cudaMalloc(&d_quant, kChunkVox * sizeof(float)));
  CUDA_OK(cudaMemcpy(d_quant, qm.q.data(), kChunkVox * sizeof(float),
                     cudaMemcpyHostToDevice));

  // Per-stream output slab. rANS can expand incompressible input slightly, so
  // this is sized above the raw share of a brick; a stream that still runs out
  // reports failure and the host falls back rather than truncating.
  const std::size_t brick_vox = static_cast<std::size_t>(kBrickDim) * kBrickDim * kBrickDim;
  const std::uint32_t stream_cap =
      static_cast<std::uint32_t>(brick_vox * esz / P + (64u << 10));

  const std::size_t per_brick =
      static_cast<std::size_t>(kChunksPerBrick) * kChunkVox * 2 +  // levels
      static_cast<std::size_t>(P) * stream_cap +                   // output slabs
      static_cast<std::size_t>(kChunksPerBrick) * kEncSymCapacity * sizeof(EncSym) +
      static_cast<std::size_t>(kChunksPerBrick) * 4;  // per-chunk symbol counts
  std::size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  const std::size_t budget = (free_mem > (768u << 20)) ? (free_mem - (768u << 20)) : (256u << 20);
  std::size_t batch = std::max<std::size_t>(1, budget / per_brick);
  batch = std::min<std::size_t>(batch, static_cast<std::size_t>(brick_count));

  std::int16_t* d_levels = nullptr;
  std::uint8_t* d_bytes = nullptr;
  std::uint32_t* d_sizes = nullptr;
  std::uint32_t* d_origins = nullptr;
  std::uint32_t* d_error = nullptr;
  EncSym* d_syms = nullptr;
  std::uint32_t* d_counts = nullptr;
  std::uint32_t* d_hist = nullptr;
  DeviceEncSymbol* d_btab = nullptr;
  std::uint8_t* d_bmap = nullptr;
  std::uint32_t* d_bbase = nullptr;
  std::uint32_t* d_offsets = nullptr;
  std::uint8_t* d_packed = nullptr;

  auto cleanup = [&] {
    cudaFree(d_levels);
    cudaFree(d_bytes);
    cudaFree(d_sizes);
    cudaFree(d_origins);
    cudaFree(d_error);
    cudaFree(d_syms);
    cudaFree(d_counts);
    cudaFree(d_hist);
    cudaFree(d_btab);
    cudaFree(d_bmap);
    cudaFree(d_bbase);
    cudaFree(d_offsets);
    cudaFree(d_packed);
    cudaFree(d_rquant);
    cudaFree(d_quant);
    cudaFree(d_volume);
  };

  const std::size_t threads_total = batch * P;
  if (cudaMalloc(&d_levels, batch * kChunksPerBrick * kChunkVox * 2) != cudaSuccess ||
      cudaMalloc(&d_bytes, threads_total * stream_cap) != cudaSuccess ||
      cudaMalloc(&d_sizes, threads_total * 4) != cudaSuccess ||
      cudaMalloc(&d_origins, batch * 3 * 4) != cudaSuccess ||
      cudaMalloc(&d_error, 4) != cudaSuccess ||
      cudaMalloc(&d_syms, batch * kChunksPerBrick * static_cast<std::size_t>(kEncSymCapacity) *
                              sizeof(EncSym)) != cudaSuccess ||
      cudaMalloc(&d_counts, batch * kChunksPerBrick * 4) != cudaSuccess ||
      (per_brick_tables &&
       (cudaMalloc(&d_hist, batch * detail::ModelIndex::total * 256 * 4) != cudaSuccess ||
        cudaMalloc(&d_btab, batch * detail::kMaxBrickTables * 256 * sizeof(DeviceEncSymbol)) !=
            cudaSuccess ||
        cudaMalloc(&d_bmap, batch * detail::ModelIndex::total) != cudaSuccess ||
        cudaMalloc(&d_bbase, batch * 4) != cudaSuccess)) ||
      cudaMalloc(&d_offsets, threads_total * 4) != cudaSuccess ||
      cudaMalloc(&d_packed, threads_total * stream_cap) != cudaSuccess) {
    cleanup();
    return Status::out_of_memory;
  }

  std::vector<std::uint32_t> origins(batch * 3);
  std::vector<std::uint32_t> sizes(threads_total);
  std::vector<std::uint8_t> packed;
  std::vector<std::uint32_t> hist;
  std::vector<DeviceEncSymbol> btab;
  std::vector<std::uint8_t> bmap;
  std::vector<std::uint32_t> bbase;

  // Stage timing, same switch as the decode path.
  const bool prof = std::getenv("GPUDCT_CUDA_PROFILE") != nullptr;
  cudaEvent_t ea{}, eb{}, ec{}, ed{};
  float ms_ke1 = 0, ms_ke2a = 0, ms_ke2b = 0;
  if (prof) {
    cudaEventCreate(&ea); cudaEventCreate(&eb);
    cudaEventCreate(&ec); cudaEventCreate(&ed);
  }
  auto stamp = [&](cudaEvent_t e) { if (prof) cudaEventRecord(e); };
  auto add = [&](float& into, cudaEvent_t f, cudaEvent_t t) {
    if (!prof) return;
    cudaEventSynchronize(t);
    float v = 0; cudaEventElapsedTime(&v, f, t); into += v;
  };
  // Wall clock for the whole call, so the kernel timings can be checked against
  // a total rather than believed. See the note on StageTimer.
  const auto t_enc_start = std::chrono::steady_clock::now();
  double ms_tables = 0, ms_payload = 0;

  for (std::uint64_t b0 = 0; b0 < brick_count; b0 += batch) {
    const std::uint32_t n =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(batch, brick_count - b0));
    for (std::uint32_t i = 0; i < n; ++i) {
      const std::uint64_t bi = b0 + i;
      origins[i * 3 + 0] = static_cast<std::uint32_t>(bi % grid.x) * kBrickDim;
      origins[i * 3 + 1] = static_cast<std::uint32_t>((bi / grid.x) % grid.y) * kBrickDim;
      origins[i * 3 + 2] =
          static_cast<std::uint32_t>(bi / (static_cast<std::uint64_t>(grid.x) * grid.y)) *
          kBrickDim;
    }
    if (cudaMemcpy(d_origins, origins.data(), n * 3 * 4, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemset(d_error, 0, 4) != cudaSuccess) {
      cleanup();
      return Status::io_error;
    }

    stamp(ea);
    {
      const dim3 blocks(n * kChunksPerBrick);
#define GPUDCT_KE1(T)                                                                    \
  ke1_forward<T><<<blocks, 256>>>(reinterpret_cast<const T*>(d_volume), dims.x, dims.y,  \
                                  dims.z, d_origins, scale, offset, d_rquant, deadzone,  \
                                  d_quant, rdo, d_levels)
      switch (dtype) {
        case DType::u8:  GPUDCT_KE1(std::uint8_t); break;
        case DType::s8:  GPUDCT_KE1(std::int8_t); break;
        case DType::u16: GPUDCT_KE1(std::uint16_t); break;
        case DType::s16: GPUDCT_KE1(std::int16_t); break;
        case DType::u32: GPUDCT_KE1(std::uint32_t); break;
        case DType::s32: GPUDCT_KE1(std::int32_t); break;
        case DType::f32: GPUDCT_KE1(float); break;
      }
#undef GPUDCT_KE1
    }

    stamp(eb);
    if (per_brick_tables &&
        cudaMemset(d_hist, 0,
                   static_cast<std::size_t>(n) * detail::ModelIndex::total * 256 * 4) !=
            cudaSuccess) {
      cleanup();
      return Status::io_error;
    }
    {
      const std::uint32_t chunks = n * kChunksPerBrick;
      ke2a_build_symbols<<<(chunks + kKE2AThreads - 1) / kKE2AThreads, kKE2AThreads>>>(
          d_levels, chunks, d_syms, kEncSymCapacity, d_counts,
          per_brick_tables ? d_hist : nullptr, d_error);
    }
    stamp(ec);
    // Per-brick tables: histogram on the device, cluster on the host, upload.
    // The clustering is a few hundred microseconds of host work on a few
    // megabytes of counts, against a symbol stream that never leaves the device.
    std::vector<std::vector<std::uint8_t>> blobs(n);
    const auto t_tab = std::chrono::steady_clock::now();
    if (per_brick_tables) {
      const auto t_h2 = std::chrono::steady_clock::now();
      hist.resize(static_cast<std::size_t>(n) * detail::ModelIndex::total * 256);
      if (cudaDeviceSynchronize() != cudaSuccess ||
          cudaMemcpy(hist.data(), d_hist, hist.size() * 4, cudaMemcpyDeviceToHost) !=
              cudaSuccess) {
        cleanup();
        return Status::io_error;
      }
      btab.assign(static_cast<std::size_t>(n) * detail::kMaxBrickTables * 256,
                  DeviceEncSymbol{});
      bmap.assign(static_cast<std::size_t>(n) * detail::ModelIndex::total,
                  detail::kUseGlobalTable);
      bbase.assign(n, 0);

      // Clustering is per brick and independent, so it runs across threads --
      // serially it cost more than every kernel in the encoder combined.
      std::vector<detail::BrickTables> bts(n);
      std::vector<detail::BrickTableView> tvs(n);
      std::vector<std::uint8_t> ok(n, 0);
      {
        unsigned nthr = std::thread::hardware_concurrency();
        if (nthr == 0) nthr = 4;
        std::atomic<std::uint32_t> next{0};
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < nthr; ++t)
          pool.emplace_back([&] {
            // Sized once per worker, refilled per brick: allocating 75 vectors
            // for every brick was most of this stage.
            std::vector<std::vector<std::uint64_t>> h(detail::ModelIndex::total);
            for (std::uint16_t mi = 0; mi < detail::ModelIndex::total; ++mi)
              h[mi].resize(ms[mi].nsym());
            for (;;) {
              const std::uint32_t i = next.fetch_add(1, std::memory_order_relaxed);
              if (i >= n) return;
              for (std::uint16_t mi = 0; mi < detail::ModelIndex::total; ++mi) {
                const std::uint32_t* src =
                    hist.data() +
                    (static_cast<std::size_t>(i) * detail::ModelIndex::total + mi) * 256;
                for (std::uint32_t v = 0; v < ms[mi].nsym(); ++v) h[mi][v] = src[v];
              }
              bts[i] = detail::build_brick_tables(ms, h, /*want_models=*/false);
              if (bts[i].used && detail::parse_brick_tables(bts[i].blob, tvs[i])) ok[i] = 1;
            }
          });
        for (auto& th : pool) th.join();
      }
      if (prof)
        std::fprintf(stderr, "[cuda]   clustering %7.2f ms\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t_h2)
                             .count() * 1000.0);

      std::uint32_t next_table = 0;
      for (std::uint32_t i = 0; i < n; ++i) {
        if (!ok[i]) continue;
        const detail::BrickTableView& tv = tvs[i];
        const detail::BrickTables& bt = bts[i];
        bbase[i] = next_table;
        for (std::uint32_t t = 0; t < tv.ntables; ++t) {
          std::uint32_t running = 0;
          for (std::uint32_t v = 0; v < 256; ++v) {
            const std::uint16_t f = tv.freqs[t * 256 + v];
            const detail::Model::EncSymbol e = detail::Model::enc_symbol_of(
                f, static_cast<std::uint16_t>(running));
            running += f;
            btab[(static_cast<std::size_t>(next_table) + t) * 256 + v] = {
                e.rcp_freq, e.freq, e.bias, e.cmpl_freq, e.rcp_shift};
          }
        }
        next_table += tv.ntables;
        std::memcpy(bmap.data() + static_cast<std::size_t>(i) * detail::ModelIndex::total,
                    tv.model_map.data(), detail::ModelIndex::total);
        blobs[i] = bt.blob;
      }

      if (cudaMemcpy(d_btab, btab.data(), btab.size() * sizeof(DeviceEncSymbol),
                     cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(d_bmap, bmap.data(), bmap.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(d_bbase, bbase.data(), n * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        cleanup();
        return Status::io_error;
      }
      dm.brick_sym = d_btab;
      dm.brick_map = d_bmap;
      dm.brick_base = d_bbase;
    }
    if (prof)
      ms_tables +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t_tab).count() *
          1000.0;

    {
      const std::uint32_t total_threads = n * P;
      ke2b_range_encode<<<(total_threads + 127) / 128, 128>>>(
          d_syms, kEncSymCapacity, d_counts, dm, P, n, d_bytes, stream_cap, d_sizes, d_error);
    }
    stamp(ed);
    add(ms_ke1, ea, eb);
    add(ms_ke2a, eb, ec);
    add(ms_ke2b, ec, ed);

    if (cudaDeviceSynchronize() != cudaSuccess) {
      cleanup();
      return Status::io_error;
    }
    std::uint32_t err = 0;
    cudaMemcpy(&err, d_error, 4, cudaMemcpyDeviceToHost);
    if (err) {
      overflowed = true;
      cleanup();
      return Status::backend_unavailable;  // retry larger, then fall back to CPU
    }

    const std::uint32_t nstreams = n * P;
    if (cudaMemcpy(sizes.data(), d_sizes, static_cast<std::size_t>(nstreams) * 4,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      cleanup();
      return Status::io_error;
    }

    // Pack on device, then move only the bytes that exist.
    std::vector<std::uint32_t> offsets(nstreams);
    std::uint32_t running = 0;
    for (std::uint32_t i = 0; i < nstreams; ++i) {
      offsets[i] = running;
      running += sizes[i];
    }
    if (cudaMemcpy(d_offsets, offsets.data(), nstreams * 4, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      cleanup();
      return Status::io_error;
    }
    ke3_compact<<<nstreams, 128>>>(d_bytes, stream_cap, d_sizes, d_offsets, nstreams, d_packed);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      cleanup();
      return Status::io_error;
    }
    packed.resize(running);
    if (running > 0 &&
        cudaMemcpy(packed.data(), d_packed, running, cudaMemcpyDeviceToHost) != cudaSuccess) {
      cleanup();
      return Status::io_error;
    }

    const auto t_pay = std::chrono::steady_clock::now();
    // Assemble each brick payload in the container's layout. The kernel wrote
    // each stream's bytes at the *end* of its slab, so the live bytes start at
    // stream_cap - size.
    for (std::uint32_t i = 0; i < n; ++i) {
      std::vector<std::uint8_t>& out = payloads[static_cast<std::size_t>(b0 + i)];
      std::size_t total = 3 + 4ull * P;
      for (std::uint32_t sIdx = 0; sIdx < P; ++sIdx) total += sizes[i * P + sIdx];
      const std::vector<std::uint8_t>& blob = blobs[i];
      out.clear();
      out.reserve(total + 5 + blob.size());
      out.push_back(0);  // coded mode
      out.push_back(blob.empty() ? 0 : 1);
      if (!blob.empty()) {
        detail::put_u32(out, static_cast<std::uint32_t>(blob.size()));
        out.insert(out.end(), blob.begin(), blob.end());
      }
      for (std::uint32_t sIdx = 0; sIdx < P; ++sIdx)
        detail::put_u32(out, sizes[i * P + sIdx]);
      out.push_back(0);  // no correction streams
      for (std::uint32_t sIdx = 0; sIdx < P; ++sIdx) {
        const std::uint32_t idx = i * P + sIdx;
        const std::uint32_t sz = sizes[idx];
        out.insert(out.end(), packed.begin() + static_cast<std::ptrdiff_t>(offsets[idx]),
                   packed.begin() + static_cast<std::ptrdiff_t>(offsets[idx] + sz));
      }
    }
    if (prof)
      ms_payload +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pay).count() *
          1000.0;
  }

  if (prof) {
    const double total =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_enc_start).count() *
        1000.0;
    std::fprintf(stderr,
                 "[cuda] ke1 (transform) %8.2f ms   ke2a (symbols) %8.2f ms   "
                 "ke2b (rANS) %8.2f ms\n"
                 "[cuda] tables %8.2f ms   payload %8.2f ms   total %8.2f ms\n",
                 ms_ke1, ms_ke2a, ms_ke2b, ms_tables, ms_payload, total);
    cudaEventDestroy(ea);
    cudaEventDestroy(eb);
    cudaEventDestroy(ec);
    cudaEventDestroy(ed);
  }
  cleanup();
  return Status::ok;
}

}  // namespace

Status encode_bricks(const void* data, Dims dims, DType dtype, float scale, float offset,
                     const QuantMatrix& qm, float deadzone, const ModelSet& ms,
                     std::uint32_t P, bool rdo, bool per_brick_tables,
                     std::vector<std::vector<std::uint8_t>>& payloads) {
  for (int cap : kEncSymCapacities) {
    bool overflowed = false;
    const Status s = encode_bricks_attempt(data, dims, dtype, scale, offset, qm, deadzone, ms,
                                           P, cap, rdo, per_brick_tables, payloads, overflowed);
    if (!overflowed) return s;
  }
  return Status::backend_unavailable;
}

// --------------------------------------------------------------------------
// DeviceVolume: the compressed archive stays in VRAM and decoding writes
// straight into device memory. See include/gpudct/device_volume.hpp.
//
// Everything the ordinary decode path redoes on every call -- uploading the
// archive, allocating scratch, expanding entropy tables, parsing every brick
// header on the host -- happens once here, at open time. What is left per call
// is a small descriptor copy and the two kernels.
// --------------------------------------------------------------------------

struct DeviceVolumeImpl {
  detail::FileHeader h{};
  VolumeInfo info{};
  Dims grid{};
  std::uint32_t P = 0;
  std::int32_t max_level = 0;
  float lo = 0.0f, hi = 0.0f;
  std::uint32_t max_nz = 0;
  std::uint32_t batch = 0;
  std::size_t device_bytes = 0;
  std::size_t shared_limit = 48u << 10;
  std::uint32_t max_tab_count = 0;

  ModelUpload models_up;
  DeviceModels dm{};

  // Device memory owned for the object's lifetime.
  std::uint8_t* d_archive = nullptr;
  float* d_quant = nullptr;
  std::uint32_t* d_error = nullptr;
  std::uint32_t* d_tab_fc = nullptr;
  std::uint8_t* d_tab_slot = nullptr;
  // Per-call scratch, sized for `batch` bricks.
  BrickDesc* d_descs = nullptr;
  std::int16_t* d_levels = nullptr;
  std::uint16_t* d_indices = nullptr;
  std::uint32_t* d_counts = nullptr;
  std::uint8_t* d_map = nullptr;
  std::uint32_t* d_tab_base = nullptr;
  std::uint32_t* d_tab_count = nullptr;

  // Host-side metadata for every brick, gathered per call into the scratch.
  std::vector<BrickDesc> descs;
  std::vector<std::uint32_t> tab_base_all, tab_count_all;
  std::vector<std::uint8_t> tab_map;

  ~DeviceVolumeImpl() {
    cudaFree(d_archive);
    cudaFree(d_quant);
    cudaFree(d_error);
    cudaFree(d_tab_fc);
    cudaFree(d_tab_slot);
    cudaFree(d_descs);
    cudaFree(d_levels);
    cudaFree(d_indices);
    cudaFree(d_counts);
    cudaFree(d_map);
    cudaFree(d_tab_base);
    cudaFree(d_tab_count);
  }
};

Status device_volume_open_impl(std::span<const std::uint8_t> archive,
                               std::unique_ptr<DeviceVolumeImpl>& out) {
  auto v = std::make_unique<DeviceVolumeImpl>();
  if (const Status s = detail::read_header(archive, v->h); s != Status::ok) return s;
  if (const Status s = inspect(archive, v->info); s != Status::ok) return s;
  // A correction layer is not implemented on the device, and quietly dropping it
  // would break the error bound the archive advertises.
  if ((v->h.flags & detail::kFlagCorrections) != 0) return Status::backend_unavailable;
  if (v->h.brick_count == 0) return Status::corrupt_bitstream;

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return Status::backend_unavailable;
  if (upload_transform_constants() != Status::ok) return Status::backend_unavailable;

  const detail::QuantMatrix qm =
      detail::QuantMatrix::build({v->h.q_base, v->h.q_a, v->h.q_b, v->h.deadzone});
  const ModelSet ms = ModelSet::defaults(v->h.table_version);
  if (upload_models(ms, v->models_up, v->dm) != Status::ok) return Status::backend_unavailable;

  v->grid = brick_grid(v->h.dims);
  v->P = v->h.streams_per_brick;
  v->max_level = detail::max_plausible_level(qm);
  v->lo = dtype_min(v->h.dtype);
  v->hi = dtype_max(v->h.dtype);
  // Sparse scratch is sized for typical chunk occupancy, not the worst case, and
  // grown on demand. Sizing for 4096 nonzeros per chunk costs 8.4 MB of device
  // memory per brick of batch -- which on a volume whose whole point is to stay
  // compressed in VRAM is absurd: the scratch would dwarf the archive.
  v->max_nz = kTypicalMaxNonzero;

  if (cudaMalloc(&v->d_archive, archive.size()) != cudaSuccess ||
      cudaMemcpy(v->d_archive, archive.data(), archive.size(), cudaMemcpyHostToDevice) !=
          cudaSuccess)
    return Status::out_of_memory;
  v->device_bytes += archive.size();

  const std::size_t nb = static_cast<std::size_t>(v->h.brick_count);
  v->descs.resize(nb);
  v->tab_base_all.assign(nb, 0);
  v->tab_count_all.assign(nb, 0);
  v->tab_map.assign(nb * detail::ModelIndex::total, detail::kUseGlobalTable);

  std::vector<std::uint16_t> tab_freq;
  for (std::size_t i = 0; i < nb; ++i) {
    const std::size_t e =
        static_cast<std::size_t>(v->h.index_offset) + i * detail::BrickEntry::kSize;
    if (e + detail::BrickEntry::kSize > archive.size()) return Status::corrupt_bitstream;
    const std::uint64_t off = detail::get_u64(archive, e);
    const std::uint32_t size = detail::get_u32(archive, e + 8);
    if (off > archive.size() || size > archive.size() - off) return Status::corrupt_bitstream;

    detail::BrickTableView tv;
    if (!parse_brick(archive.subspan(static_cast<std::size_t>(off), size), v->P, v->descs[i],
                     reinterpret_cast<std::uint64_t>(v->d_archive) + off, tv))
      return Status::backend_unavailable;  // raw brick or unsupported; caller falls back

    v->tab_base_all[i] = static_cast<std::uint32_t>(tab_freq.size() / 256);
    if (v->descs[i].has_tables) {
      v->tab_count_all[i] = static_cast<std::uint32_t>(tv.freqs.size() / 256);
      v->max_tab_count = std::max(v->max_tab_count, v->tab_count_all[i]);
      tab_freq.insert(tab_freq.end(), tv.freqs.begin(), tv.freqs.end());
      std::memcpy(v->tab_map.data() + i * detail::ModelIndex::total, tv.model_map.data(),
                  detail::ModelIndex::total);
    }
  }

  const std::uint32_t ntab = static_cast<std::uint32_t>(tab_freq.size() / 256);
  if (ntab > 0) {
    std::uint16_t* d_freq = nullptr;
    const bool ok =
        cudaMalloc(&d_freq, tab_freq.size() * 2) == cudaSuccess &&
        cudaMalloc(&v->d_tab_fc, static_cast<std::size_t>(ntab) * 256 * 4) == cudaSuccess &&
        cudaMalloc(&v->d_tab_slot, static_cast<std::size_t>(ntab) * kProbScale) == cudaSuccess &&
        cudaMemcpy(d_freq, tab_freq.data(), tab_freq.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess;
    if (!ok) {
      cudaFree(d_freq);
      return Status::out_of_memory;
    }
    k0_build_brick_tables<<<ntab, 256>>>(d_freq, v->d_tab_fc, v->d_tab_slot, ntab);
    const cudaError_t e = cudaDeviceSynchronize();
    cudaFree(d_freq);
    if (e != cudaSuccess) return Status::io_error;
    v->device_bytes += static_cast<std::size_t>(ntab) * (256 * 4 + kProbScale);
  }

  // Batch size for the persistent scratch, which is what caps one call. The
  // sparse coefficient buffers dominate it.
  std::size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  const std::size_t per_brick =
      static_cast<std::size_t>(kChunksPerBrick) * v->max_nz * 4 +
      static_cast<std::size_t>(kChunksPerBrick) * 4 + sizeof(BrickDesc) +
      detail::ModelIndex::total + 8;
  const std::size_t budget = (free_mem > (512u << 20)) ? (free_mem / 8) : (64u << 20);
  // Capped at the shared-launch threshold as well as by memory: past it the
  // low-latency kernel stops being the right one, and a caller asking for more
  // bricks is served by looping rather than by a bigger allocation.
  v->batch = static_cast<std::uint32_t>(std::clamp<std::size_t>(
      budget / per_brick, 1, std::min<std::size_t>(nb, kSharedLaunchMaxBricks)));

  const std::size_t b = v->batch;
  if (cudaMalloc(&v->d_quant, kChunkVox * sizeof(float)) != cudaSuccess ||
      cudaMemcpy(v->d_quant, qm.q.data(), kChunkVox * sizeof(float), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMalloc(&v->d_error, 4) != cudaSuccess ||
      cudaMalloc(&v->d_descs, b * sizeof(BrickDesc)) != cudaSuccess ||
      cudaMalloc(&v->d_levels, b * kChunksPerBrick * v->max_nz * 2) != cudaSuccess ||
      cudaMalloc(&v->d_indices, b * kChunksPerBrick * v->max_nz * 2) != cudaSuccess ||
      cudaMalloc(&v->d_counts, b * kChunksPerBrick * 4) != cudaSuccess ||
      cudaMalloc(&v->d_map, b * detail::ModelIndex::total) != cudaSuccess ||
      cudaMalloc(&v->d_tab_base, b * 4) != cudaSuccess ||
      cudaMalloc(&v->d_tab_count, b * 4) != cudaSuccess)
    return Status::out_of_memory;
  v->device_bytes += b * (static_cast<std::size_t>(kChunksPerBrick) * v->max_nz * 4 +
                          kChunksPerBrick * 4 + sizeof(BrickDesc) + detail::ModelIndex::total + 8);

  int optin = 0;
  if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) ==
          cudaSuccess &&
      optin > 0 &&
      cudaFuncSetAttribute(k1_entropy_decode_shared, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           optin) == cudaSuccess)
    v->shared_limit = static_cast<std::size_t>(optin);

  out = std::move(v);
  return Status::ok;
}

Status device_volume_decode_impl(DeviceVolumeImpl& v, std::span<const std::uint32_t> indices,
                                 void* d_out) {
  if (indices.empty()) return Status::ok;
  if (d_out == nullptr) return Status::invalid_argument;
  const std::size_t esz = dtype_size(v.h.dtype);
  const std::size_t brick_vox = static_cast<std::size_t>(kBrickDim) * kBrickDim * kBrickDim;

  std::vector<BrickDesc> bd;
  std::vector<std::uint32_t> base, count;
  std::vector<std::uint8_t> map;

  for (std::size_t off = 0; off < indices.size(); off += v.batch) {
    const std::uint32_t n =
        static_cast<std::uint32_t>(std::min<std::size_t>(v.batch, indices.size() - off));
    bd.clear();
    base.clear();
    count.clear();
    map.clear();
    // The kernels index their per-brick metadata by position within the batch,
    // so the batch's rows are gathered densely here rather than indexed sparsely
    // on the device. It is a few kilobytes per call.
    for (std::uint32_t i = 0; i < n; ++i) {
      const std::uint32_t bi = indices[off + i];
      if (bi >= v.descs.size()) return Status::invalid_argument;
      bd.push_back(v.descs[bi]);
      base.push_back(v.tab_base_all[bi]);
      count.push_back(v.tab_count_all[bi]);
      map.insert(map.end(), v.tab_map.begin() + static_cast<std::ptrdiff_t>(bi) *
                                                    detail::ModelIndex::total,
                 v.tab_map.begin() +
                     static_cast<std::ptrdiff_t>(bi + 1) * detail::ModelIndex::total);
    }

    if (cudaMemcpy(v.d_descs, bd.data(), n * sizeof(BrickDesc), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(v.d_tab_base, base.data(), n * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(v.d_tab_count, count.data(), n * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(v.d_map, map.data(), map.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(v.d_error, 0, 4) != cudaSuccess ||
        cudaMemset(v.d_counts, 0, static_cast<std::size_t>(n) * kChunksPerBrick * 4) !=
            cudaSuccess)
      return Status::io_error;

    DeviceBrickTables bt{v.d_tab_fc, v.d_tab_slot, v.d_map, v.d_tab_base};
    const std::size_t shared_bytes = static_cast<std::size_t>(v.max_tab_count) * kProbScale +
                                     static_cast<std::size_t>(v.P) * kRansStates * 4 + 4;
    if (n <= kSharedLaunchMaxBricks && v.P >= 32 && shared_bytes <= v.shared_limit) {
      k1_entropy_decode_shared<<<n, v.P, shared_bytes>>>(
          v.d_descs, v.dm, bt, v.d_tab_count, v.max_level, v.d_levels, v.d_indices, v.d_counts,
          v.max_nz, n, v.d_error);
    } else {
      const std::uint32_t threads = n * v.P;
      const std::uint32_t blocks = (threads + kK1Threads - 1) / kK1Threads;
      k1_entropy_decode<<<blocks, kK1Threads>>>(v.d_descs, v.dm, bt, v.max_level, v.d_levels,
                                                v.d_indices, v.d_counts, v.max_nz, v.P, n,
                                                v.d_error);
    }

    std::uint8_t* dst = static_cast<std::uint8_t*>(d_out) + off * brick_vox * esz;
    const dim3 blocks2(n * kChunksPerBrick);
#define GPUDCT_DV_K2(T)                                                                       \
  k2_inverse_transform<T><<<blocks2, 256>>>(v.d_levels, v.d_indices, v.d_counts, v.max_nz,    \
                                            v.d_quant, v.h.data_scale, v.h.data_offset, v.lo, \
                                            v.hi, reinterpret_cast<T*>(dst))
    switch (v.h.dtype) {
      case DType::u8:  GPUDCT_DV_K2(std::uint8_t); break;
      case DType::s8:  GPUDCT_DV_K2(std::int8_t); break;
      case DType::u16: GPUDCT_DV_K2(std::uint16_t); break;
      case DType::s16: GPUDCT_DV_K2(std::int16_t); break;
      case DType::u32: GPUDCT_DV_K2(std::uint32_t); break;
      case DType::s32: GPUDCT_DV_K2(std::int32_t); break;
      case DType::f32: GPUDCT_DV_K2(float); break;
    }
#undef GPUDCT_DV_K2

    if (cudaDeviceSynchronize() != cudaSuccess) return Status::io_error;
    std::uint32_t err = 0;
    cudaMemcpy(&err, v.d_error, 4, cudaMemcpyDeviceToHost);
    if (err == kErrOverflow) {
      // A denser chunk than the scratch was sized for. Grow once to the format's
      // worst case and redo this batch; the buffers persist, so a volume that
      // needs the headroom pays for it once rather than per call.
      if (v.max_nz >= static_cast<std::uint32_t>(kChunkVox)) return Status::corrupt_bitstream;
      const std::uint32_t grown = kChunkVox;
      std::int16_t* nl = nullptr;
      std::uint16_t* ni = nullptr;
      if (cudaMalloc(&nl, static_cast<std::size_t>(v.batch) * kChunksPerBrick * grown * 2) !=
              cudaSuccess ||
          cudaMalloc(&ni, static_cast<std::size_t>(v.batch) * kChunksPerBrick * grown * 2) !=
              cudaSuccess) {
        cudaFree(nl);
        cudaFree(ni);
        return Status::out_of_memory;
      }
      cudaFree(v.d_levels);
      cudaFree(v.d_indices);
      v.d_levels = nl;
      v.d_indices = ni;
      v.device_bytes += static_cast<std::size_t>(v.batch) * kChunksPerBrick *
                        (grown - v.max_nz) * 4;
      v.max_nz = grown;
      off -= v.batch;  // redo this batch with the larger scratch
      continue;
    }
    if (err != 0) return Status::corrupt_bitstream;
  }
  return Status::ok;
}

Status device_volume_open(std::span<const std::uint8_t> archive, DeviceVolumeImpl** out,
                          DeviceVolumeInfo& summary) {
  std::unique_ptr<DeviceVolumeImpl> v;
  const Status s = device_volume_open_impl(archive, v);
  if (s != Status::ok) return s;
  summary.info = v->info;
  summary.grid = v->grid;
  summary.bricks = v->h.brick_count;
  summary.batch = v->batch;
  summary.device_bytes = v->device_bytes;
  *out = v.release();
  return Status::ok;
}

void device_volume_close(DeviceVolumeImpl* v) { delete v; }

Status device_malloc(std::size_t bytes, void** out) {
  if (out == nullptr) return Status::invalid_argument;
  *out = nullptr;
  return cudaMalloc(out, bytes) == cudaSuccess ? Status::ok : Status::out_of_memory;
}

void device_free(void* p) { cudaFree(p); }

Status device_to_host(void* dst, const void* src, std::size_t bytes) {
  return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? Status::ok
                                                                            : Status::io_error;
}

Status device_volume_decode(DeviceVolumeImpl* v, std::span<const std::uint32_t> indices,
                            void* d_out) {
  if (v == nullptr) return Status::invalid_argument;
  return device_volume_decode_impl(*v, indices, d_out);
}

#undef CUDA_OK

}  // namespace gpudct::cuda
