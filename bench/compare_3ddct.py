#!/usr/bin/env python3
"""Rate-distortion comparison against 3ddct (docs/QUALITY.md section 2).

Two codecs cannot be compared by putting their quality knobs side by side: the
knobs are not the same scale and neither is the resulting quality. The honest
comparison is BD-rate -- the average difference in bitrate at *equal quality*,
integrated over the quality range where both codecs actually operate.

This computes BD-rate by piecewise-linear interpolation of each codec's
rate-quality curve, which needs no polynomial fitting and cannot produce the
overshoot artifacts a cubic fit does at the ends of the range.

Usage:
    python compare_3ddct.py <gpudct.exe> <cmp3ddct.exe> <brick.raw> X Y Z
"""

import os
import subprocess
import sys
import tempfile

# Scratch paths are built here rather than hardcoded as /tmp/...: this script has
# to run on Windows too, where that is not a path at all and every intermediate
# would silently fail to appear.
_TMP = tempfile.mkdtemp(prefix="gpudct_cmp_")


def _tmp(name):
    return os.path.join(_TMP, name)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def metrics(gpudct, orig, dec, dims, dtype="u8"):
    """PSNR / SSIM / MAE of `dec` against `orig`, via the gpudct metrics tool."""
    out = run([gpudct, "metrics", orig, dec, "--dims", dims, "--dtype", dtype])
    got = {}
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("psnr (data range)"):
            got["psnr"] = float(s.split()[-2])
        elif s.startswith("ssim 3d"):
            got["ssim"] = float(s.split()[2])
        elif s.startswith("mae ") and "by intensity" not in s:
            # The report also has a "mae by intensity" row; only the bare one is
            # a single number.
            try:
                got["mae"] = float(s.split()[1])
            except ValueError:
                pass
    return got


def curve_gpudct(gpudct, src, dims, qualities, profile="balanced"):
    rows = []
    for q in qualities:
        out = run([gpudct, "compress", src, _tmp("g.gdct"), "--dims", dims,
                   "--dtype", "u8", "--quality", str(q), "--profile", profile])
        size = None
        for line in out.splitlines():
            if line.startswith("output"):
                size = int(line.split()[1])
        if size is None:
            continue
        run([gpudct, "decompress", _tmp("g.gdct"), _tmp("g.raw")])
        m = metrics(gpudct, src, _tmp("g.raw"), dims)
        if "psnr" in m:
            rows.append((size, m["psnr"], m["ssim"], q))
    return rows


def curve_3ddct(gpudct, cmp3, src, dims, qualities):
    x, y, z = dims.split(",")
    rows = []
    for q in qualities:
        out = run([cmp3, src, _tmp("3.raw"), x, y, z, str(q)])
        if not out.strip():
            continue
        size = int(out.split()[0])
        m = metrics(gpudct, src, _tmp("3.raw"), dims)
        if "psnr" in m:
            rows.append((size, m["psnr"], m["ssim"], q))
    return rows


def interp_rate(curve, target, key):
    """Bits-per-voxel at a given quality, linearly interpolated.

    Returns None outside the curve's range rather than extrapolating: a BD-rate
    computed off the end of a measured curve is fiction.
    """
    pts = sorted(((r[key], r[0]) for r in curve))
    for i in range(len(pts) - 1):
        (q0, r0), (q1, r1) = pts[i], pts[i + 1]
        if q0 <= target <= q1:
            if q1 == q0:
                return r0
            t = (target - q0) / (q1 - q0)
            return r0 + t * (r1 - r0)
    return None


def bd_rate(ours, theirs, key=1, label="PSNR"):
    """Mean bitrate difference at equal quality. Negative means we use fewer bits."""
    lo = max(min(r[key] for r in ours), min(r[key] for r in theirs))
    hi = min(max(r[key] for r in ours), max(r[key] for r in theirs))
    if hi <= lo:
        return None, 0, (lo, hi)

    steps = 50
    diffs = []
    for i in range(steps + 1):
        t = lo + (hi - lo) * i / steps
        a = interp_rate(ours, t, key)
        b = interp_rate(theirs, t, key)
        if a and b:
            diffs.append((a - b) / b * 100.0)
    if not diffs:
        return None, 0, (lo, hi)
    return sum(diffs) / len(diffs), len(diffs), (lo, hi)


def main():
    if len(sys.argv) < 7:
        print(__doc__)
        return 2
    gpudct, cmp3, src = sys.argv[1], sys.argv[2], sys.argv[3]
    dims = ",".join(sys.argv[4:7])

    gq = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    tq = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    print(f"corpus: {src}  dims {dims}\n")
    ours = curve_gpudct(gpudct, src, dims, gq)
    theirs = curve_3ddct(gpudct, cmp3, src, dims, tq)

    print(f"{'':<8} {'gpudct':^34}   {'3ddct':^34}")
    print(f"{'point':<8} {'bytes':>9} {'bpv':>7} {'psnr':>7} {'ssim':>8}   "
          f"{'bytes':>9} {'bpv':>7} {'psnr':>7} {'ssim':>8}")
    nvox = 1
    for d in dims.split(","):
        nvox *= int(d)
    for i in range(max(len(ours), len(theirs))):
        a = ours[i] if i < len(ours) else None
        b = theirs[i] if i < len(theirs) else None
        fa = (f"{a[0]:>9} {8*a[0]/nvox:>7.4f} {a[1]:>7.2f} {a[2]:>8.5f}"
              if a else " " * 34)
        fb = (f"{b[0]:>9} {8*b[0]/nvox:>7.4f} {b[1]:>7.2f} {b[2]:>8.5f}"
              if b else " " * 34)
        print(f"{i:<8} {fa}   {fb}")

    for key, label in ((1, "PSNR"), (2, "3D-SSIM")):
        bd, n, rng = bd_rate(ours, theirs, key, label)
        if bd is None:
            print(f"\nBD-rate vs {label}: no overlapping quality range")
            continue
        verdict = "gpudct smaller" if bd < 0 else "3ddct smaller"
        print(f"\nBD-rate vs {label:8}: {bd:+.2f}%  ({verdict}, "
              f"over {label} {rng[0]:.2f}..{rng[1]:.4f}, {n} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
