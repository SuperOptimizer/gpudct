"""Does an apron make HEVC viable at 128^3 effective granularity?

NVDEC will not decode below 144x144, so a 128^3 unit has to be padded to 144^3
somehow. Three ways, all measured on the same interior brick, at the same QP,
scored on the *inner 128^3 only* -- the voxels a caller actually receives:

  A  144^3 of real data, all of it useful      the reference: no waste at all,
                                               but the access unit is 144^3
  B  128^3 payload + 8-voxel real apron        apron carries true neighbouring
                                               content, trimmed after decode
  C  128^3 payload + edge-replicated pad       what bench/nvcodec_bench.py did

B and C carry the identical 1.42x pixel overhead, so B-vs-C isolates what a real
apron is worth over cheap padding. A-vs-B is the price of the finer granularity.

Effective ratio is always (useful voxels) / (stored bytes): the apron is
overhead, and it is duplicated into all six neighbours, so it must be charged.
"""
import os, subprocess, numpy as np

FF = open(r"C:\Users\mcdof\AppData\Local\Temp\claude\D--gpudct\6905deb5-8764-42d8-acd6-66ff3a151bf3\scratchpad\ffpath").read().strip()
vol = np.memmap("big512.raw", dtype=np.uint8, mode="r").reshape(512, 512, 512)

P, A = 128, 144            # payload edge, apron edge
o = (A - P) // 2           # 8-voxel apron on every side
z0 = y0 = x0 = 192         # an interior brick, so the apron is real data


def code(sub, qp):
    n = sub.shape[1]
    p = "nvc/ap.mp4"
    subprocess.run(
        [FF, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{n}x{n}", "-r", "30",
         "-i", "-", "-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq",
         "-rc", "constqp", "-qp", str(qp), "-bf", "3", "-b_ref_mode", "middle",
         "-spatial-aq", "0", "-temporal-aq", "0", "-g", str(sub.shape[0]),
         "-pix_fmt", "yuv420p", "-color_range", "pc", "-an", "-f", "mp4", p],
        input=sub.tobytes(), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=True)
    out = subprocess.run(
        [FF, "-hide_banner", "-loglevel", "error", "-c:v", "hevc_cuvid", "-i", p,
         "-vf", "scale=in_range=full:out_range=full",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout
    d = np.frombuffer(out, np.uint8)[:sub.size].reshape(sub.shape)
    return d, os.path.getsize(p)


def psnr(a, b):
    mse = ((a.astype(np.int32) - b.astype(np.int32)) ** 2).mean()
    return 10 * np.log10(65025 / mse)


payload = np.ascontiguousarray(vol[z0:z0+P, y0:y0+P, x0:x0+P])
apron = np.ascontiguousarray(vol[z0-o:z0-o+A, y0-o:y0-o+A, x0-o:x0-o+A])

pad = np.pad(payload, ((0, A-P), (0, A-P), (0, A-P)), mode="edge")
pad = np.ascontiguousarray(pad)

print(f"interior brick at ({x0},{y0},{z0}); apron {o} voxels/side; "
      f"pixel overhead {A**3/P**3:.3f}x")
print(f"{'qp':>3}  {'config':<28} {'eff ratio':>10} {'psnr(inner)':>12} "
      f"{'edge psnr':>10} {'core psnr':>10}")
for qp in (18, 24, 30, 36):
    for name, src, inner in (
        ("A 144^3 all useful", apron, None),
        ("B 128^3 + real apron", apron, (o, o+P)),
        ("C 128^3 + replicated pad", pad, (0, P)),
    ):
        d, nbytes = code(src, qp)
        if inner is None:
            got, ref, useful = d, apron, apron.size
        else:
            a, b = inner
            got = d[a:b, a:b, a:b]
            ref, useful = payload, payload.size
        # edge = outermost 8 voxels of the delivered cube, core = the rest
        e = 8
        m = np.ones(ref.shape, bool)
        m[e:-e, e:-e, e:-e] = False
        print(f"{qp:>3}  {name:<28} {useful/nbytes:>9.2f}x "
              f"{psnr(ref, got):>11.2f} "
              f"{psnr(ref[m], got[m]):>10.2f} {psnr(ref[~m], got[~m]):>10.2f}")
    print()
