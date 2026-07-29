"""Do independently-decoded units seam at their shared faces?

The apron test earlier asked whether the codec degrades *near* a unit border, by
comparing edge PSNR against core PSNR inside one unit. It does not, and that was
the wrong question. A seam is a discontinuity *across* the join between two units
that were coded without knowledge of each other -- a derivative artifact, not a
magnitude one -- and per-unit PSNR cannot see it.

HEVC deblocks and applies SAO inside a frame, but nothing filters across clips,
so tiled independent clips should seam like tiled independent JPEGs.

Metric is QUALITY.md 1.3's blockiness index: mean |delta| across the interior
faces of the tiling, over mean |delta| one voxel inside. Computed on the original
too, since real data has its own gradient there; the number that matters is how
much the codec *raises* it.
"""
import os, subprocess, numpy as np

FF = open(r"C:\Users\mcdof\AppData\Local\Temp\claude\D--gpudct\6905deb5-8764-42d8-acd6-66ff3a151bf3\scratchpad\ffpath").read().strip()
SRC = "scrollL0.raw"
N = 512
vol = np.fromfile(SRC, np.uint8).reshape(N, N, N)


def hevc(sub, qp, w):
    """Encode one unit of edge `w` (already padded/aproned), return decoded."""
    p = "nvc/seam.hevc"
    subprocess.run(
        [FF, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{w}", "-r", "30",
         "-i", "-", "-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq",
         "-rc", "constqp", "-qp", str(qp),
         "-init_qpI", str(max(0, qp-4)), "-init_qpP", str(qp),
         "-init_qpB", str(min(51, qp+4)),
         "-bf", "3", "-b_ref_mode", "middle", "-spatial-aq", "0",
         "-temporal-aq", "0", "-g", str(sub.shape[0]), "-pix_fmt", "yuv420p",
         "-color_range", "pc", "-an", "-f", "hevc", p],
        input=sub.tobytes(), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=True)
    out = subprocess.run(
        [FF, "-hide_banner", "-loglevel", "error", "-c:v", "hevc_cuvid", "-i", p,
         "-vf", "scale=in_range=full:out_range=full",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout
    d = np.frombuffer(out, np.uint8)[:sub.size].reshape(sub.shape)
    return d, os.path.getsize(p)


def tile_hevc(P, qp, apron):
    """Tile the volume into P^3 units. apron=0 -> edge-replicated pad to 144;
    apron>0 -> carry that many real neighbouring voxels on each side."""
    MIN = 144
    out = np.zeros_like(vol)
    total = 0
    for z in range(0, N, P):
        for y in range(0, N, P):
            for x in range(0, N, P):
                if apron:
                    a = apron
                    zs, ys, xs = z-a, y-a, x-a
                    ze, ye, xe = z+P+a, y+P+a, x+P+a
                    sub = np.zeros((P+2*a,)*3, np.uint8)
                    # clamp-sample the source so border units still get content
                    zi = np.clip(np.arange(zs, ze), 0, N-1)
                    yi = np.clip(np.arange(ys, ye), 0, N-1)
                    xi = np.clip(np.arange(xs, xe), 0, N-1)
                    sub = vol[np.ix_(zi, yi, xi)]
                    w = sub.shape[1]
                    if w < MIN:   # still under the decoder floor
                        pad = MIN - w
                        sub = np.pad(sub, ((0, 0), (0, pad), (0, pad)), mode="edge")
                    sub = np.ascontiguousarray(sub)
                    d, nb = hevc(sub, qp, sub.shape[2])
                    out[z:z+P, y:y+P, x:x+P] = d[a:a+P, a:a+P, a:a+P]
                else:
                    sub = np.ascontiguousarray(vol[z:z+P, y:y+P, x:x+P])
                    w = max(P, MIN)
                    if w > P:
                        sub = np.ascontiguousarray(
                            np.pad(sub, ((0, 0), (0, w-P), (0, w-P)), mode="edge"))
                    d, nb = hevc(sub, qp, w)
                    out[z:z+P, y:y+P, x:x+P] = d[:P, :P, :P]
                total += nb
    return out, total


def blockiness(a, P):
    """mean|delta| across the tiling's interior faces / mean|delta| just inside."""
    face, inter = [], []
    for ax in (0, 1, 2):
        d = np.abs(np.diff(a.astype(np.int32), axis=ax))
        # index j in d is the step between plane j and j+1
        idx = np.arange(d.shape[ax])
        on = (idx % P) == (P - 1)          # the joins between units
        off = (idx % P) == (P // 2)        # a reference plane mid-unit
        face.append(np.take(d, np.where(on)[0], axis=ax).mean())
        inter.append(np.take(d, np.where(off)[0], axis=ax).mean())
    return float(np.mean(face) / np.mean(inter)), float(np.mean(face))


def psnr(a, b):
    return 10*np.log10(65025/((a.astype(np.int32)-b.astype(np.int32))**2).mean())


P = 128
print(f"tiling {N}^3 into {P}^3 units, seams measured at the shared faces")
b0, f0 = blockiness(vol, P)
print(f"original volume: blockiness {b0:.4f}  (face |d| {f0:.3f})\n")
print(f"{'config':<26} {'ratio':>8} {'psnr':>7} {'blockiness':>11} {'excess':>8}")
for qp in (24, 30):
    for name, apron in (("replicated pad", 0), ("apron 8", 8), ("apron 16", 16)):
        dec, nb = tile_hevc(P, qp, apron)
        b, f = blockiness(dec, P)
        print(f"qp{qp} {name:<21} {vol.size/nb:>7.2f}x {psnr(vol,dec):>7.2f} "
              f"{b:>11.4f} {b/b0-1:>+7.1%}")
