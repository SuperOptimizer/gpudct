// CUDA encode kernels (docs/ROADMAP.md M7).
//
// Included into backend.cu; see the note there about keeping device structs and
// the host code that fills them in one translation unit.
//
//   KE1  one block per chunk. Gathers voxels, forward DCT, quantizes, writes a
//        dense i16 level array.
//   KE2  one thread per rANS stream. Walks that stream's chunks in *reverse*,
//        expanding each into symbols and encoding them, writing bytes backwards
//        into a per-stream slab.
//
// The reverse walk is what makes KE2 possible at all. rANS is LIFO, so the
// encoder must consume the whole stream's symbol sequence back to front -- which
// naively means buffering every symbol in the stream (hundreds of thousands, per
// thread). But reversing a concatenation is the same as concatenating the
// reversals in reverse order, so processing chunks last-to-first with each
// chunk's own symbols reversed produces the identical byte stream while
// buffering only one chunk's worth at a time.
//
// That also makes the output structurally identical to the CPU encoder's: same
// symbol sequence, same states, same bytes, given the same quantized levels.

namespace gpudct::cuda {

// Upper bound on symbols from one chunk. A chunk emits at most: 1 flag, 1
// exponent, 8 L1 mask bytes, 8 L2 bytes per significant sub-block (64), and per
// nonzero coefficient a magnitude symbol plus up to 25 bypass bits plus a sign.
// A stream whose chunk exceeds this is rejected and the host falls back, rather
// than the kernel writing past the end.
static constexpr int kMaxChunkSymbols = 2 + 8 + 64 * 8 + kChunkVox * 28;

// Threads per symbol-building block. Bounded by the shared-memory scratch each
// thread needs for its neighbour context (64 ints).
static constexpr int kKE2AThreads = 128;

struct EncSym {
  std::uint16_t model;
  std::uint16_t value;
};

// Division-free encoder parameters, mirroring Model::EncSymbol on the host.
struct DeviceEncSymbol {
  std::uint32_t rcp_freq;
  std::uint32_t freq;
  std::uint32_t bias;
  std::uint32_t cmpl_freq;
  std::uint32_t rcp_shift;
};

struct DeviceEncModels {
  const DeviceEncSymbol* sym;        // [offset[model] + value]
  const std::uint32_t* offset;       // per-model base
  // Per-brick tables, mirroring the decode side. Null map => global tables only.
  const DeviceEncSymbol* brick_sym;  // [(base[brick] + table) * 256 + value]
  const std::uint8_t* brick_map;     // [brick][model] -> table, or 0xff
  const std::uint32_t* brick_base;   // [brick] -> first table index
};

static constexpr std::uint8_t kUseGlobalTableEnc = 0xff;

__device__ __forceinline__ DeviceEncSymbol enc_symbol(const DeviceEncModels& m,
                                                      std::uint32_t brick,
                                                      std::uint32_t model,
                                                      std::uint32_t value) {
  if (m.brick_map != nullptr) {
    const std::uint8_t t = m.brick_map[static_cast<std::size_t>(brick) * kModelTotalDev + model];
    if (t != kUseGlobalTableEnc)
      return m.brick_sym[(static_cast<std::size_t>(m.brick_base[brick]) + t) * 256 + value];
  }
  return m.sym[m.offset[model] + value];
}

// Summed magnitude of a coefficient's causal 3D neighbours within its
// sub-block; mirrors neighbour_sum in core/chunk_codec.cpp.
__device__ __forceinline__ int nsum_of(const int* nb, int bit) {
  const int bu = bit & 3, bv = (bit >> 2) & 3, bw = (bit >> 4) & 3;
  int s = 0;
  if (bu > 0) s += nb[bit - 1];
  if (bv > 0) s += nb[bit - 4];
  if (bw > 0) s += nb[bit - 16];
  return s;
}

// --------------------------------------------------------------------------
// KE1: forward transform and quantize.
// --------------------------------------------------------------------------
__constant__ float c_alpha[16];

template <int N>
struct LeeFwd {
  __device__ __forceinline__ static void run(float* v) {
    constexpr int H = N / 2;
    float g[H], h[H];
    const float* s = lee_s<N>();
    for (int k = 0; k < H; ++k) {
      const float a = v[k], b = v[N - 1 - k];
      g[k] = a + b;
      h[k] = (a - b) * s[k];
    }
    LeeFwd<H>::run(g);
    LeeFwd<H>::run(h);
    for (int k = 0; k < H; ++k) v[2 * k] = g[k];
    for (int k = 0; k < H - 1; ++k) v[2 * k + 1] = h[k] + h[k + 1];
    v[N - 1] = h[H - 1];
  }
};

template <>
struct LeeFwd<1> {
  __device__ __forceinline__ static void run(float*) {}
};

__device__ __forceinline__ void dct16_pencil(float* v) {
  LeeFwd<16>::run(v);
  for (int u = 0; u < 16; ++u) v[u] *= c_alpha[u];
}

// Rate estimates for the device RDO, precomputed on the host.
//
// The CPU version calls Model::bit_cost, which walks the model tables. Doing
// that per candidate on the device would mean several dependent global loads
// inside the innermost decision; a small table of "bits to code magnitude m in
// band b" costs 5 KB of constant memory and answers it with one lookup.
static constexpr int kRdoMagTable = 256;
__constant__ float c_rdo_bits[kNumBandsDev][kLevelCtxDev][kRdoMagTable];
__constant__ float c_rdo_dc_bits[kRdoMagTable];
__constant__ float c_rdo_sub_overhead[kNumBandsDev];
__constant__ float c_rdo_lambda_scale[1];

__device__ __forceinline__ float rdo_bits(int band, int ctx, int mag, bool is_dc) {
  if (mag <= 0) return 0.0f;
  if (mag < kRdoMagTable) return is_dc ? c_rdo_dc_bits[mag] : c_rdo_bits[band][ctx][mag];
  // Past the table the cost grows with the magnitude's bit length; the escape
  // mantissa is bypass, so each doubling costs one more bit.
  const float base = is_dc ? c_rdo_dc_bits[kRdoMagTable - 1]
                           : c_rdo_bits[band][ctx][kRdoMagTable - 1];
  return base + 2.0f * log2f(static_cast<float>(mag) / (kRdoMagTable - 1));
}

// One thread per sub-block. Mirrors rdo_optimize in core/rdo.hpp.
__device__ void rdo_subblock(const float* __restrict coeffs, const float* __restrict quant,
                             int sb, std::int16_t* __restrict levels) {
  const int band = band_of_subblock_index(sb);
  const float lambda_band = c_rdo_lambda_scale[0];

  float keep_cost = 0.0f, zero_cost = 0.0f;
  bool any_nonzero = false;
  int nb[kSubVox] = {};  // neighbour magnitudes, mirroring the host

  for (int bit = 0; bit < kSubVox; ++bit) {
    const int idx = coeff_of_subblock_bit(sb, bit);
    const float c = coeffs[idx];
    const float q = quant[idx];
    const float lambda = lambda_band * q * q;
    const int base = levels[idx];
    const int mag = base < 0 ? -base : base;
    const float ac = fabsf(c);

    const float d_zero = ac * ac;
    zero_cost += d_zero;
    if (mag == 0) {
      keep_cost += d_zero;
      continue;
    }

    const int ctx = level_ctx_of_prev(nsum_of(nb, bit));
    int best_mag = 0;
    float best_j = d_zero;
    for (int mtry = mag; mtry >= mag - 1 && mtry >= 1; --mtry) {
      const float err = ac - static_cast<float>(mtry) * q;
      const float j = err * err + lambda * rdo_bits(band, ctx, mtry, idx == 0);
      if (j < best_j) {
        best_j = j;
        best_mag = mtry;
      }
    }
    levels[idx] = static_cast<std::int16_t>(c < 0.0f ? -best_mag : best_mag);
    keep_cost += best_j;
    nb[bit] = best_mag;
    if (best_mag != 0) any_nonzero = true;
  }

  if (!any_nonzero) return;
  const float q0 = quant[coeff_of_subblock_bit(sb, 0)];
  const float overhead = lambda_band * q0 * q0 * c_rdo_sub_overhead[band];
  if (zero_cost < keep_cost + overhead)
    for (int bit = 0; bit < kSubVox; ++bit) levels[coeff_of_subblock_bit(sb, bit)] = 0;
}

template <typename T>
__global__ void ke1_forward(const T* __restrict volume, std::uint32_t dim_x, std::uint32_t dim_y,
                            std::uint32_t dim_z, const std::uint32_t* __restrict brick_origin,
                            float scale, float offset, const float* __restrict rquant,
                            float deadzone, const float* __restrict quant, bool rdo,
                            std::int16_t* __restrict out_levels) {
  __shared__ float sh[kChunkVox];

  const std::uint32_t chunk = blockIdx.x;
  const int tid = static_cast<int>(threadIdx.x);

  const std::uint32_t brick = chunk / kChunksPerBrick;
  const std::uint32_t ci = chunk % kChunksPerBrick;
  const std::uint32_t ox =
      brick_origin[brick * 3 + 0] + static_cast<std::uint32_t>((ci % kBrickChunks) * 16);
  const std::uint32_t oy =
      brick_origin[brick * 3 + 1] +
      static_cast<std::uint32_t>(((ci / kBrickChunks) % kBrickChunks) * 16);
  const std::uint32_t oz =
      brick_origin[brick * 3 + 2] +
      static_cast<std::uint32_t>((ci / (kBrickChunks * kBrickChunks)) * 16);

  // Gather with edge replication, exactly as gather_chunk does on the host.
  for (int i = tid; i < kChunkVox; i += blockDim.x) {
    const std::uint32_t lx = static_cast<std::uint32_t>(i % 16);
    const std::uint32_t ly = static_cast<std::uint32_t>((i / 16) % 16);
    const std::uint32_t lz = static_cast<std::uint32_t>(i / 256);
    const std::uint32_t gx = min(ox + lx, dim_x - 1);
    const std::uint32_t gy = min(oy + ly, dim_y - 1);
    const std::uint32_t gz = min(oz + lz, dim_z - 1);
    const std::size_t o = (static_cast<std::size_t>(gz) * dim_y + gy) * dim_x + gx;
    sh[i] = static_cast<float>(volume[o]) * scale + offset;
  }
  __syncthreads();

  float v[16];
  // x pass, then y, then z -- the host order in forward_chunk.
  {
    const int zy = tid;
    for (int x = 0; x < 16; ++x) v[x] = sh[zy * 16 + x];
    dct16_pencil(v);
    __syncthreads();
    for (int x = 0; x < 16; ++x) sh[zy * 16 + x] = v[x];
  }
  __syncthreads();
  {
    const int z = tid / 16, x = tid % 16;
    for (int y = 0; y < 16; ++y) v[y] = sh[z * 256 + y * 16 + x];
    dct16_pencil(v);
    __syncthreads();
    for (int y = 0; y < 16; ++y) sh[z * 256 + y * 16 + x] = v[y];
  }
  __syncthreads();
  {
    const int yx = tid;
    for (int z = 0; z < 16; ++z) v[z] = sh[z * 256 + yx];
    dct16_pencil(v);
    __syncthreads();
    for (int z = 0; z < 16; ++z) sh[z * 256 + yx] = v[z];
  }
  __syncthreads();

  std::int16_t* dst = out_levels + static_cast<std::size_t>(chunk) * kChunkVox;
  for (int i = tid; i < kChunkVox; i += blockDim.x) {
    const float a = fabsf(sh[i]) * rquant[i] + deadzone;
    int mag = (a < 1.0f) ? 0 : static_cast<int>(a);
    if (mag > 32767) mag = 32767;  // clamped by the host's plausible-level bound
    dst[i] = static_cast<std::int16_t>(sh[i] < 0.0f ? -mag : mag);
  }

  if (rdo) {
    // Sub-blocks are independent, so one thread each. The coefficients are still
    // in shared memory from the transform above, which is why this runs here
    // rather than as its own kernel.
    __syncthreads();
    for (int sb = tid; sb < kSubCount; sb += blockDim.x) rdo_subblock(sh, quant, sb, dst);
  }
}

// --------------------------------------------------------------------------
// KE2: symbol generation and rANS encoding, one thread per stream.
// --------------------------------------------------------------------------

// Emits one symbol and, if `hist` is set, counts it.
//
// Counting here rather than in a second pass is what makes the histogram nearly
// free. A separate kernel had to re-read every symbol, and since each thread
// walks its own chunk's buffer the warp's 32 reads land ~6 KB apart -- fully
// uncoalesced, and worth ~150 ms per gigabyte. The symbols are already in
// registers at this point.
__device__ __forceinline__ void emit(EncSym* buf, int& n, int cap, std::uint16_t model,
                                     std::uint16_t value, bool& overflow,
                                     std::uint32_t* __restrict hist) {
  if (n >= cap) {
    overflow = true;
    return;
  }
  buf[n].model = model;
  buf[n].value = value;
  ++n;
  if (hist != nullptr)
    atomicAdd(&hist[static_cast<std::size_t>(model) * 256 + value], 1u);
}

__device__ __forceinline__ void emit_len_mag(EncSym* buf, int& n, int cap, std::uint32_t mag,
                                             std::uint16_t len_model, bool& overflow,
                                             std::uint32_t* __restrict hist) {
  const int nb = 32 - __clz(static_cast<int>(mag));
  emit(buf, n, cap, len_model, static_cast<std::uint16_t>(nb), overflow, hist);
  for (int i = nb - 2; i >= 0; --i)
    emit(buf, n, cap, MI::bypass, static_cast<std::uint16_t>((mag >> i) & 1), overflow, hist);
}

// Builds one chunk's symbol sequence in coding (forward) order.
__device__ int build_chunk_symbols(const std::int16_t* __restrict levels, EncSym* buf, int cap,
                                   bool& overflow, int* __restrict nb,
                                   std::uint32_t* __restrict hist) {
  int n = 0;

  unsigned long long sub_mask[kSubCount];
  for (int i = 0; i < kSubCount; ++i) sub_mask[i] = 0ull;
  unsigned long long l1 = 0;
  for (int idx = 0; idx < kChunkVox; ++idx) {
    if (levels[idx] == 0) continue;
    const int u = idx & 15, v = (idx >> 4) & 15, w = (idx >> 8) & 15;
    const int sb = ((w >> 2) * 4 + (v >> 2)) * 4 + (u >> 2);
    const int bit = ((w & 3) * 4 + (v & 3)) * 4 + (u & 3);
    sub_mask[sb] |= (1ull << bit);
    l1 |= (1ull << sb);
  }

  emit(buf, n, cap, MI::chunk_nonzero, static_cast<std::uint16_t>(l1 != 0 ? 1 : 0), overflow, hist);
  if (l1 == 0) return n;
  emit(buf, n, cap, MI::exponent, 0, overflow, hist);

  int mctx = 0;
  for (int j = 0; j < 8; ++j) {
    const std::uint32_t byte = static_cast<std::uint32_t>((l1 >> (8 * j)) & 0xff);
    emit(buf, n, cap, static_cast<std::uint16_t>(MI::l1(j, mctx)),
         static_cast<std::uint16_t>(byte), overflow, hist);
    mctx = mask_ctx_of_prev(byte);
  }

  for (int sb = 0; sb < kSubCount; ++sb) {
    if (!((l1 >> sb) & 1ull)) continue;
    const int band = band_of_subblock_index(sb);

    mctx = 0;
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t byte = static_cast<std::uint32_t>((sub_mask[sb] >> (8 * j)) & 0xff);
      emit(buf, n, cap, static_cast<std::uint16_t>(MI::l2(band, mctx)),
           static_cast<std::uint16_t>(byte), overflow, hist);
      mctx = mask_ctx_of_prev(byte);
    }

    for (int i = 0; i < kSubVox; ++i) nb[i] = 0;
    for (int bit = 0; bit < kSubVox; ++bit) {
      if (!((sub_mask[sb] >> bit) & 1ull)) continue;
      const int idx = coeff_of_subblock_bit(sb, bit);
      const int lv = levels[idx];
      const int mag = lv < 0 ? -lv : lv;

      if (idx == 0) {
        emit_len_mag(buf, n, cap, static_cast<std::uint32_t>(mag), MI::dc_len, overflow, hist);
      } else {
        const std::uint16_t model =
            static_cast<std::uint16_t>(MI::level(band, level_ctx_of_prev(nsum_of(nb, bit))));
        if (mag <= 15) {
          emit(buf, n, cap, model, static_cast<std::uint16_t>(mag - 1), overflow, hist);
        } else {
          emit(buf, n, cap, model, static_cast<std::uint16_t>(kLevelEscape), overflow, hist);
          emit_len_mag(buf, n, cap, static_cast<std::uint32_t>(mag) - 15, MI::esc_len, overflow, hist);
        }
      }
      emit(buf, n, cap, MI::bypass, static_cast<std::uint16_t>(lv < 0 ? 1 : 0), overflow, hist);
      nb[bit] = mag;
    }
  }
  return n;
}

// KE2a: symbol generation, one thread per chunk.
//
// Generation was originally folded into the per-stream encoder, which meant
// 2048 threads each scanning 4096 coefficients per chunk -- and twice over, once
// to count and once to emit. Chunks are independent, so this runs at one thread
// per chunk instead: 16x the parallelism on the part that does the scanning,
// and the count falls out of the same pass that writes the symbols.
__global__ void ke2a_build_symbols(const std::int16_t* __restrict levels,
                                   std::uint32_t chunk_count, EncSym* __restrict syms,
                                   int sym_capacity, std::uint32_t* __restrict counts,
                                   std::uint32_t* __restrict hist,
                                   std::uint32_t* __restrict error_flag) {
  // The neighbour-magnitude scratch lives in shared memory. As a thread-local
  // array it is indexed by a runtime value, so it spills to local memory and
  // every context lookup becomes a global-memory round trip -- which took this
  // kernel from 25 ms to 247 ms per gigabyte when the neighbour context landed.
  __shared__ int sh_nb[kKE2AThreads * kSubVox];

  const std::uint32_t chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= chunk_count) return;

  bool overflow = false;
  EncSym* buf = syms + static_cast<std::size_t>(chunk) * sym_capacity;
  const int n = build_chunk_symbols(levels + static_cast<std::size_t>(chunk) * kChunkVox, buf,
                                    sym_capacity, overflow, sh_nb + threadIdx.x * kSubVox,
                                    hist == nullptr
                                        ? nullptr
                                        : hist + static_cast<std::size_t>(chunk /
                                                                          kChunksPerBrick) *
                                                     kModelTotalDev * 256);
  if (overflow) {
    atomicExch(error_flag, 1u);
    counts[chunk] = 0;
    return;
  }
  counts[chunk] = static_cast<std::uint32_t>(n);
}

// KE2b: rANS arithmetic, one thread per stream.
//
// All that remains serial: walk this stream's chunks last-to-first, each chunk's
// pre-built symbols last-to-first, and fold them into the interleaved states.
__global__ void ke2b_range_encode(const EncSym* __restrict syms, int sym_capacity,
                                  const std::uint32_t* __restrict counts,
                                  DeviceEncModels models, std::uint32_t streams_per_brick,
                                  std::uint32_t brick_count, std::uint8_t* __restrict out_bytes,
                                  std::uint32_t stream_capacity,
                                  std::uint32_t* __restrict out_sizes,
                                  std::uint32_t* __restrict error_flag) {
  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t P = streams_per_brick;
  const std::uint32_t b = tid / P;
  const std::uint32_t s = tid % P;
  if (b >= brick_count) return;

  std::uint8_t* slab = out_bytes + static_cast<std::size_t>(tid) * stream_capacity;
  std::uint32_t cursor = stream_capacity;

  std::uint32_t state[kRansStates];
  for (std::uint32_t i = 0; i < kRansStates; ++i) state[i] = kRansL;

  // A symbol's interleaved state is fixed by its index within the stream, so the
  // reverse walk has to start from the stream's total symbol count. The counts
  // are already computed, so this is a handful of loads rather than a full
  // second pass over the coefficients.
  std::uint32_t total = 0;
  for (std::uint32_t ci = s; ci < kChunksPerBrick; ci += P)
    total += counts[b * kChunksPerBrick + ci];

  std::int32_t first = static_cast<std::int32_t>(s);
  std::int32_t last = first;
  while (last + static_cast<std::int32_t>(P) < kChunksPerBrick)
    last += static_cast<std::int32_t>(P);

  std::uint32_t k = total;
  for (std::int32_t ci = last; ci >= first; ci -= static_cast<std::int32_t>(P)) {
    const std::size_t cidx = static_cast<std::size_t>(b) * kChunksPerBrick +
                             static_cast<std::uint32_t>(ci);
    const EncSym* buf = syms + cidx * sym_capacity;
    const int n = static_cast<int>(counts[cidx]);
    for (int i = n - 1; i >= 0; --i) {
      --k;
      const DeviceEncSymbol e = enc_symbol(models, b, buf[i].model, buf[i].value);
      std::uint32_t x = state[k % kRansStates];
      const std::uint32_t x_max = ((kRansL >> kProbBits) << 8) * e.freq;
      while (x >= x_max) {
        if (cursor == 0) {
          atomicExch(error_flag, 1u);
          return;
        }
        slab[--cursor] = static_cast<std::uint8_t>(x & 0xff);
        x >>= 8;
      }
      const std::uint32_t q =
          static_cast<std::uint32_t>((static_cast<std::uint64_t>(x) * e.rcp_freq) >> 32) >>
          e.rcp_shift;
      state[k % kRansStates] = x + e.bias + q * e.cmpl_freq;
    }
  }

  // Flush the interleaved states. Bytes go back-to-front here, which is the same
  // as the host's "emit then reverse", so the order must match the host exactly:
  // state n-1 down to state 0, most-significant byte first. That leaves state 0
  // little-endian at the lowest addresses, which is what the decoder reads.
  for (std::int32_t i = static_cast<std::int32_t>(kRansStates) - 1; i >= 0; --i) {
    if (cursor < 4) {
      atomicExch(error_flag, 1u);
      return;
    }
    const std::uint32_t x = state[i];
    slab[--cursor] = static_cast<std::uint8_t>(x >> 24);
    slab[--cursor] = static_cast<std::uint8_t>(x >> 16);
    slab[--cursor] = static_cast<std::uint8_t>(x >> 8);
    slab[--cursor] = static_cast<std::uint8_t>(x);
  }

  out_sizes[tid] = stream_capacity - cursor;
}

// Packs each stream's live bytes together so the readback moves only the output
// rather than the whole scratch.
//
// Each stream writes its bytes at the *end* of a fixed-size slab, so the live
// region is the tail. Copying the slabs verbatim meant transferring the full
// per-stream capacity -- hundreds of megabytes to retrieve a few -- which cost
// more than the encoding did.
__global__ void ke3_compact(const std::uint8_t* __restrict slabs,
                            std::uint32_t stream_capacity,
                            const std::uint32_t* __restrict sizes,
                            const std::uint32_t* __restrict offsets, std::uint32_t nstreams,
                            std::uint8_t* __restrict packed) {
  const std::uint32_t sIdx = blockIdx.x;
  if (sIdx >= nstreams) return;
  const std::uint32_t sz = sizes[sIdx];
  const std::uint8_t* src =
      slabs + static_cast<std::size_t>(sIdx) * stream_capacity + (stream_capacity - sz);
  std::uint8_t* dst = packed + offsets[sIdx];
  for (std::uint32_t i = threadIdx.x; i < sz; i += blockDim.x) dst[i] = src[i];
}

}  // namespace gpudct::cuda
