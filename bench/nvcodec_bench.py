#!/usr/bin/env python3
"""Compare gpudct against the NVENC/NVDEC hardware video codecs on volumetric u8.

The question this answers: a GPU already contains fixed-function silicon that
compresses 8-bit images at tens of gigabytes per second. If we treat a volume as
a video -- z becomes time -- how much ratio, quality and speed does that buy,
and what does a codec built for 3D data actually add?

Brick edge length is swept, and the sweep *is* the experiment. A brick is the
random-access unit: at edge `b`, seeing one voxel costs decoding b^3. The premise
going in was that video codecs earn their ratio from long GOPs, so their numbers
would improve steadily as `b` grows and random access is given away. Measured,
that is mostly false: 16x the frame area is worth +1.9% at qp24 and 32x the clip
length +5.6%, and length saturates past ~128 frames. At matched granularity a
video codec and gpudct are even on ratio -- see docs/QUALITY.md section 2.1.

Grayscale handling: NVENC has no useful monochrome mode. `-pix_fmt gray` is
accepted and decodes, but --sweep measured it at +85% bitrate for equal PSNR, so
input is promoted to yuv420p with flat chroma -- two constant quarter-planes,
which cost almost nothing. All metrics are computed on luma against the original.

PSNR uses peak 255, matching `gpudct eval`'s psnr_range on 0..255 data, so the
two tools' numbers can be read on the same axis.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------- ffmpeg glue


def find_ffmpeg():
    for name in ("ffmpeg", "ffmpeg.exe"):
        p = shutil.which(name)
        if p:
            return p
    root = os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"
        r"\BtbN.FFmpeg.GPL_Microsoft.Winget.Source_8wekyb3d8bbwe"
    )
    for dirpath, _, files in os.walk(root):
        if "ffmpeg.exe" in files:
            return os.path.join(dirpath, "ffmpeg.exe")
    sys.exit("ffmpeg not found; winget install BtbN.FFmpeg.GPL")


FFMPEG = None


def run(cmd, stdin_bytes=None, allow_fail=False):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, input=stdin_bytes,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        if allow_fail:
            return None, r.stderr.decode("utf-8", "replace")
        sys.stderr.write(r.stderr.decode("utf-8", "replace")[-3000:])
        raise RuntimeError(f"ffmpeg failed ({r.returncode})")
    return dt, r.stdout


# mp4 rather than a raw elementary stream: -stream_loop, which the timing method
# depends on, needs a seekable input.
CODECS = {
    "h264": ("h264_nvenc", "h264_cuvid", "mp4"),
    "hevc": ("hevc_nvenc", "hevc_cuvid", "mp4"),
    "av1": ("av1_nvenc", "av1_cuvid", "mp4"),
}

# Both halves of the fixed-function path refuse frames below a hardware
# minimum, measured on this device by bisection. NVENC stops at 129 wide for
# HEVC/AV1 and 145 for H.264; NVDEC is stricter and will not open a stream
# smaller than 144x144. So the binding constraint is the decoder, and a 128^3
# brick -- the granularity this codec is built around -- cannot go through the
# hardware video path at all without being padded out by 1.27x in pixels.
MIN_DIM = {"h264": 146, "hevc": 144, "av1": 144}  # even, for 4:2:0 chroma

# ------------------------------------------------------------------ encoding

TUNED = {
    "preset": "p7",
    # -tune uhq is rejected by the driver whenever B-frames are enabled, at
    # every resolution tried up to 512x512, so hq is the only tune that can
    # also have B-frames -- which are worth much more here.
    "tune": "hq",
    # Monochrome (-pix_fmt gray, HEVC Rext) is accepted by NVENC and decoded by
    # NVDEC, but costs +85% bitrate at equal PSNR -- its mono path is far worse
    # than 4:2:0, whose two flat chroma planes are nearly free. yuv444p ties
    # 4:2:0 to within 0.01%, for the same reason.
    "pix_fmt": "yuv420p",
    # Hierarchical B-frames are the single biggest lever: turning B-frames off
    # costs +13%, and referencing only half of them ("middle") rather than all
    # ("each") is worth -12.8%. -rc-lookahead buys nothing on top of it, and at
    # bf>3 the adaptive B decision overrides the setting anyway.
    "bf": 3,
    "b_ref_mode": "middle",
    "extra": [],
}


def cfg_with(**kw):
    c = dict(TUNED)
    c.update(kw)
    return c


def encode_args(codec, qp, gop, cfg, lossless=False):
    enc, _, _ = CODECS[codec]
    a = ["-c:v", enc, "-preset", cfg["preset"]]
    if lossless:
        a += ["-tune", "lossless"]
    else:
        a += ["-tune", cfg["tune"], "-rc", "constqp", "-qp", str(qp)]
    a += [
        # Adaptive quantization moves bits to where a human eye looks. The
        # metric here is PSNR and there is no viewer, so AQ is a pure loss.
        "-spatial-aq", "0",
        "-temporal-aq", "0",
        "-g", str(gop),
        "-pix_fmt", cfg["pix_fmt"],
        "-color_range", "pc",
        "-an",
    ]
    if cfg["bf"] > 0:
        a += ["-bf", str(cfg["bf"]), "-b_ref_mode", cfg["b_ref_mode"]]
    else:
        a += ["-bf", "0"]
    return a + list(cfg["extra"])


def raw_in(w, h):
    return ["-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "30"]


def encode(buf, w, h, codec, qp, gop, cfg, out_path, lossless=False,
           loops=0, raw_path=None, allow_fail=False):
    src = ["-stream_loop", str(loops), "-i", raw_path] if raw_path else ["-i", "-"]
    dst = ["-f", "null", "-"] if out_path is None else \
          ["-f", CODECS[codec][2], out_path]
    cmd = ([FFMPEG, "-hide_banner", "-loglevel", "error", "-y"] + raw_in(w, h)
           + src + encode_args(codec, qp, gop, cfg, lossless) + dst)
    dt, err = run(cmd, None if raw_path else buf, allow_fail)
    if dt is None:
        return None, err
    return dt, (os.path.getsize(out_path) if out_path else 0)


def decode_to_host(path, codec):
    # The explicit full-range filter is load-bearing for AV1: av1_nvenc does not
    # preserve the full-range tag that HEVC does, so the decoder assumes limited
    # range and expands 16..235 to 0..255. That is an affine distortion, not a
    # coding loss -- it pins AV1's measured PSNR near 29 dB no matter what the
    # quantizer does, and hides ~18 dB. Verified a no-op for HEVC by the
    # --verify lossless round-trip, which stays bit-exact with it applied.
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error",
           "-c:v", CODECS[codec][1], "-i", path,
           "-vf", "scale=in_range=full:out_range=full",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    return run(cmd)


def decode_on_device(path, codec, loops):
    """Frames never leave VRAM -- the comparison that matches DeviceVolume,
    which also does not copy back."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error",
           "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
           "-stream_loop", str(loops), "-c:v", CODECS[codec][1], "-i", path,
           "-f", "null", "-"]
    return run(cmd)[0]


# ffmpeg spends on the order of 100 ms starting up, opening a CUDA context and
# initialising the codec. For a 16 MiB brick that dwarfs the work, so a single
# wall-clock measurement reports the process, not the ASIC. Everything below is
# timed by slope: run the clip once and 1+R times in one process and difference,
# which cancels the fixed cost exactly.


def timed_by_slope(fn, loops):
    t1 = min(fn(0) for _ in range(2))
    t2 = min(fn(loops) for _ in range(2))
    return max(t2 - t1, 1e-9) / loops


# ------------------------------------------------------------------- metrics


def metrics(o, d):
    o = o.astype(np.int32)
    d = d.astype(np.int32)
    e = np.abs(o - d)
    mse = float(np.mean((o - d) ** 2))
    return {
        "psnr": float("inf") if mse == 0 else 10.0 * np.log10(65025.0 / mse),
        "mae": float(e.mean()),
        "p99": float(np.percentile(e, 99)),
        "max": int(e.max()),
    }


def bd_rate(ref, test):
    """Bjontegaard delta-rate: average bitrate change of `test` vs `ref` over
    the PSNR range they share. Negative is better. Piecewise-linear in
    (psnr, log bpv), matching how the C++ side computes it."""
    ref = sorted(ref)
    test = sorted(test)
    lo = max(ref[0][0], test[0][0])
    hi = min(ref[-1][0], test[-1][0])
    if hi - lo < 1e-6:
        return float("nan")

    def interp(pts, p):
        xs = [a for a, _ in pts]
        ys = [np.log(b) for _, b in pts]
        return float(np.interp(p, xs, ys))

    ps = np.linspace(lo, hi, 64)
    d = np.mean([interp(test, p) - interp(ref, p) for p in ps])
    return float((np.exp(d) - 1.0) * 100.0)


# -------------------------------------------------------------- geometries


def pad_to(sub, n):
    """Edge-replicate rows/columns out to the hardware's minimum frame."""
    if sub.shape[2] < n:
        k = n - sub.shape[2]
        sub = np.concatenate([sub, np.repeat(sub[:, :, -1:], k, axis=2)], axis=2)
    if sub.shape[1] < n:
        k = n - sub.shape[1]
        sub = np.concatenate([sub, np.repeat(sub[:, -1:, :], k, axis=1)], axis=1)
    return sub


def as_bricks(vol, codec, b):
    nz, ny, nx = vol.shape
    n = max(b, MIN_DIM[codec])
    for bz in range(0, nz, b):
        for by in range(0, ny, b):
            for bx in range(0, nx, b):
                sl = (slice(bz, bz + b), slice(by, by + b), slice(bx, bx + b))
                sub = np.ascontiguousarray(vol[sl])
                pad = np.ascontiguousarray(pad_to(sub, n))
                yield (f"b{bz}_{by}_{bx}", pad.shape[2], pad.shape[1],
                       pad.shape[0], pad.tobytes(), sl, sub.shape)


# ------------------------------------------------------------------ the runs


def run_bricks(vol, codec, qp, cfg, tmpdir, reps, b, limit=None, timed=True):
    """Per-brick clips at edge length `b`. Ratio and PSNR are over every brick
    encoded; the timing is per brick, because that is the unit a viewer asks
    for -- and `b` *is* the random-access granularity."""
    total_bytes = 0
    orig_parts, dec_parts = [], []
    enc_one = dec_one = float("nan")
    raw_path = os.path.join(tmpdir, "_slope.raw")
    n = 0
    for name, w, h, f, buf, sl, shape in as_bricks(vol, codec, b):
        if limit is not None and n >= limit:
            break
        path = os.path.join(tmpdir, f"{name}_{codec}_{qp}.{CODECS[codec][2]}")
        _, nbytes = encode(buf, w, h, codec, qp, f, cfg, path)
        total_bytes += nbytes
        _, out = decode_to_host(path, codec)
        got = np.frombuffer(out, dtype=np.uint8)[: f * h * w].reshape(f, h, w)
        dec_parts.append(got[:, : shape[1], : shape[2]].copy())  # drop the pad
        orig_parts.append(np.ascontiguousarray(vol[sl]))

        # Time the first brick only; the rest are identical work and timing
        # every one multiplies the run for no information.
        if n == 0 and timed:
            with open(raw_path, "wb") as fh:
                fh.write(buf)
            enc_one = timed_by_slope(
                lambda L: encode(buf, w, h, codec, qp, f, cfg, None,
                                 loops=L, raw_path=raw_path)[0], reps)
            dec_one = timed_by_slope(
                lambda L: decode_on_device(path, codec, L), reps)
        os.remove(path)
        n += 1

    o = np.concatenate([p.ravel() for p in orig_parts])
    d = np.concatenate([p.ravel() for p in dec_parts])
    m = metrics(o, d)
    vox = o.size
    m.update(geometry=b, codec=codec, qp=qp, preset=cfg["preset"],
             bf=cfg["bf"], pix=cfg["pix_fmt"], bytes=total_bytes, bricks=n,
             ratio=vox / total_bytes, bpv=8.0 * total_bytes / vox,
             enc_ms=enc_one * 1e3, dec_ms=dec_one * 1e3,
             enc_MBps=(b ** 3) / enc_one / 1e6,
             dec_MBps=(b ** 3) / dec_one / 1e6)
    return m


# ------------------------------------------------------------------ reporting

HEADER = (f"{'brick':>5} {'codec':<5} {'pre':<3} {'bf':>2} {'pix':<7} {'qp':>3} "
          f"{'ratio':>8}  {'bpv':>7} {'psnr':>6} {'mae':>6} {'p99':>4} {'max':>4} "
          f"{'enc ms':>8} {'dec ms':>8} {'enc MB/s':>9} {'dec MB/s':>9}")


def fmt_row(m):
    return (f"{m['geometry']:>5} {m['codec']:<5} {m['preset']:<3} {m['bf']:>2} "
            f"{m['pix']:<7} {m['qp']:>3} {m['ratio']:>7.2f}x {m['bpv']:>7.4f} "
            f"{m['psnr']:>6.2f} {m['mae']:>6.3f} {m['p99']:>4.0f} {m['max']:>4.0f} "
            f"{m['enc_ms']:>8.1f} {m['dec_ms']:>8.2f} "
            f"{m['enc_MBps']:>9.1f} {m['dec_MBps']:>9.1f}")


# --------------------------------------------------------------- sweep mode

# Candidate settings, scored by BD-rate against the baseline over a QP set.
# Anything the driver rejects is reported as such rather than silently skipped.
def candidates(rnd):
    if rnd == 2:
        # Round 2: combinations of what round 1 liked. Round 1's verdicts:
        #   b_ref middle       -12.8%   the single biggest win
        #   no-scenecut + la32  -5.4%
        #   bf=5 + la32         -5.0%
        #   monochrome         +84.8%   NVENC's Rext mono path costs ~2x the
        #                               bitrate at equal PSNR, while the flat
        #                               chroma planes of 4:2:0 are nearly free.
        #                               Counterintuitive, and the reason to
        #                               sweep rather than assume.
        la = ["-rc-lookahead", "32"]
        ns = ["-no-scenecut", "1"]
        return [
            ("baseline (bf3, ref each)", cfg_with()),
            ("mid", cfg_with(b_ref_mode="middle")),
            ("mid + la32", cfg_with(b_ref_mode="middle", extra=la)),
            ("mid + la32 + noscene", cfg_with(b_ref_mode="middle", extra=la + ns)),
            ("mid bf5", cfg_with(b_ref_mode="middle", bf=5)),
            ("mid bf5 + la32", cfg_with(b_ref_mode="middle", bf=5, extra=la)),
            ("mid bf5 + la32 + noscene",
             cfg_with(b_ref_mode="middle", bf=5, extra=la + ns)),
            ("mid bf2", cfg_with(b_ref_mode="middle", bf=2)),
            ("mid bf4", cfg_with(b_ref_mode="middle", bf=4)),
            ("b_ref disabled", cfg_with(b_ref_mode="disabled")),
            ("b_ref disabled + la32", cfg_with(b_ref_mode="disabled", extra=la)),
        ]
    return [
        ("baseline p7 420 bf3", cfg_with()),
        ("monochrome (gray)", cfg_with(pix_fmt="gray")),
        ("yuv444p", cfg_with(pix_fmt="yuv444p")),
        ("no B-frames", cfg_with(bf=0)),
        ("bf=5", cfg_with(bf=5)),
        ("bf=5 + lookahead 32", cfg_with(bf=5, extra=["-rc-lookahead", "32"])),
        ("b_ref middle", cfg_with(b_ref_mode="middle")),
        ("no scenecut + la32", cfg_with(extra=["-rc-lookahead", "32",
                                               "-no-scenecut", "1"])),
        ("10-bit encode of 8-bit", cfg_with(extra=["-highbitdepth", "1"])),
        ("preset p4", cfg_with(preset="p4")),
        ("preset p1", cfg_with(preset="p1")),
        ("tune uhq (needs bf=0)", cfg_with(tune="uhq", bf=0)),
        ("spatial-aq on", cfg_with(extra=["-spatial-aq", "1"])),
        ("mono + bf5 + la32", cfg_with(pix_fmt="gray", bf=5,
                                       extra=["-rc-lookahead", "32"])),
    ]


def sweep(vol, codec, qps, tmpdir, b, bricks, rnd):
    print(f"\nsettings sweep: {codec}, brick {b}, qp {qps}, "
          f"{bricks} brick(s), BD-rate vs baseline (negative = better)\n")
    print(f"{'setting':<26} {'BD-rate':>9}   " +
          "  ".join(f"qp{q}: ratio/psnr" for q in qps))
    base_curve = None
    results = []
    for name, cfg in candidates(rnd):
        curve, cells, failed = [], [], None
        for qp in qps:
            try:
                m = run_bricks(vol, codec, qp, cfg, tmpdir, 0, b, bricks,
                               timed=False)
            except RuntimeError:
                failed = "driver rejected"
                break
            curve.append((m["psnr"], m["bpv"]))
            cells.append(f"{m['ratio']:6.2f}x/{m['psnr']:5.2f}")
        if failed:
            print(f"{name:<26} {failed:>9}")
            continue
        if base_curve is None:
            base_curve = curve
            bd = 0.0
        else:
            bd = bd_rate(base_curve, curve)
        results.append((bd, name))
        print(f"{name:<26} {bd:>+8.2f}%   " + "  ".join(cells), flush=True)

    print("\nbest first:")
    for bd, name in sorted(results):
        print(f"  {bd:>+8.2f}%  {name}")


# ------------------------------------------------------------------- driver


def main():
    global FFMPEG
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--dims", default="512,512,512", help="nx,ny,nz")
    ap.add_argument("--codecs", default="hevc,av1")
    ap.add_argument("--qps", default="12,18,24,30,36,42,48")
    ap.add_argument("--geoms", default="256,512",
                    help="comma list of brick edge lengths")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--presets", default=None,
                    help="comma list; sweep encoder preset instead of TUNED")
    ap.add_argument("--bricks", type=int, default=None,
                    help="encode only the first N bricks")
    ap.add_argument("--tmpdir", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--sweep", type=int, default=0, metavar="ROUND",
                    help="score encoder settings by BD-rate (round 1 or 2)")
    ap.add_argument("--verify", action="store_true",
                    help="lossless round-trip check, then exit")
    a = ap.parse_args()

    FFMPEG = find_ffmpeg()
    nx, ny, nz = (int(v) for v in a.dims.split(","))
    vol = np.fromfile(a.raw, dtype=np.uint8)
    if vol.size != nx * ny * nz:
        sys.exit(f"{a.raw} has {vol.size} bytes, {nx}x{ny}x{nz} needs {nx*ny*nz}")
    vol = vol.reshape(nz, ny, nx)

    tmpdir = a.tmpdir or os.path.join(os.path.dirname(os.path.abspath(a.raw)), "nvc")
    os.makedirs(tmpdir, exist_ok=True)
    qps = [int(q) for q in a.qps.split(",")]
    geoms = [int(g) for g in a.geoms.split(",")]

    if a.verify:
        for codec in a.codecs.split(","):
            path = os.path.join(tmpdir, f"verify_{codec}.mp4")
            r = encode(vol.tobytes(), nx, ny, codec, 0, nz, cfg_with(bf=0),
                       path, lossless=True, allow_fail=True)
            if r[0] is None:
                print(f"{codec:<5} lossless: unsupported by this encoder")
                continue
            _, out = decode_to_host(path, codec)
            dec = np.frombuffer(out, dtype=np.uint8)[: vol.size].reshape(vol.shape)
            print(f"{codec:<5} lossless: "
                  f"{'EXACT' if np.array_equal(vol, dec) else 'MISMATCH'}  "
                  f"ratio {vol.size/os.path.getsize(path):.3f}x")
        return

    print(f"{a.raw}  {nx}x{ny}x{nz} u8  ({vol.size/2**20:.1f} MiB)")

    if a.sweep:
        for codec in a.codecs.split(","):
            sweep(vol, codec, qps, tmpdir, geoms[0], a.bricks, a.sweep)
        return

    rows = []
    print(HEADER)
    for b in geoms:
        for codec in a.codecs.split(","):
            for qp in qps:
                for pre in (a.presets.split(",") if a.presets else [None]):
                    cfg = cfg_with(preset=pre) if pre else TUNED
                    m = run_bricks(vol, codec, qp, cfg, tmpdir, a.reps, b,
                                   a.bricks)
                    rows.append(m)
                    print(fmt_row(m), flush=True)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
