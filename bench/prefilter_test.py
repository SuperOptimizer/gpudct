"""Does encoder-side pre-filtering buy ratio? (docs/ROADMAP.md M4 --denoise)

Scroll CT is noisy and a transform codec spends real bits coding that noise.
3ddct ships a `predeblock` pre-filter on the argument that removing it improves
both ratio and apparent quality. That is testable without implementing anything
in the codec: filter the volume, compress the filtered version, and score the
result against the ORIGINAL.

Scoring against the original is the whole point. Measuring the filtered volume
against itself would credit the filter for the distortion it introduced, which
is how a pre-filter can be made to look arbitrarily good. Here the filter's own
error is charged to the codec, so a win is a real win: fewer bits AND closer to
the ground truth than spending those bits on noise would have been.

SSIM is reported alongside PSNR because PSNR is the metric most hostile to
denoising -- it scores every removed noise voxel as error regardless of whether
the structure improved.
"""
import os, subprocess, numpy as np

EXE = r"D:\gpudct\build-werror\tools\gpudct.exe"
SRC = "scrollL0.raw"
N = 512
vol = np.fromfile(SRC, np.uint8).reshape(N, N, N)


def gauss1d(sigma):
    r = max(1, int(round(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def blur3(a, sigma):
    """Separable Gaussian, edge-replicated."""
    k = gauss1d(sigma)
    r = len(k) // 2
    out = a.astype(np.float32)
    for ax in (0, 1, 2):
        pad = [(0, 0)] * 3
        pad[ax] = (r, r)
        p = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None)] * 3
            sl[ax] = slice(i, i + out.shape[ax])
            acc += w * p[tuple(sl)]
        out = acc
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def psnr(a, b):
    return 10 * np.log10(65025 / ((a.astype(np.int32) - b.astype(np.int32)) ** 2).mean())


def run(src_path, q):
    subprocess.run([EXE, "compress", src_path, "pf.gdct", "--dims", f"{N},{N},{N}",
                    "--dtype", "u8", "--quality", str(q)],
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run([EXE, "decompress", "pf.gdct", "pf.raw"],
                   stdout=subprocess.DEVNULL, check=True)
    d = np.fromfile("pf.raw", np.uint8).reshape(N, N, N)
    return vol.size / os.path.getsize("pf.gdct"), d


def ssim_via_cli(dec_path):
    out = subprocess.run([EXE, "metrics", SRC, dec_path, "--dims", f"{N},{N},{N}",
                          "--dtype", "u8"], stdout=subprocess.PIPE, check=True).stdout.decode()
    for line in out.splitlines():
        if "ssim 3d" in line:
            return float(line.split()[2])
    return float("nan")


QUALS = (0.125, 0.25, 0.5, 1.0)
variants = [("none", None), ("gauss 0.4", 0.4), ("gauss 0.6", 0.6), ("gauss 0.8", 0.8)]

print(f"{'prefilter':<12} {'quality':>7} {'ratio':>8} {'psnr':>7} {'ssim':>8}   (vs ORIGINAL)")
for name, sigma in variants:
    if sigma is None:
        path = SRC
    else:
        f = blur3(vol, sigma)
        path = f"pf_src_{sigma}.raw"
        f.tofile(path)
        print(f"{'  ' + name:<12} {'filter only':>7} {'':>8} {psnr(vol, f):>7.2f}")
    for q in QUALS:
        r, d = run(path, q)
        d.tofile("pf_dec.raw")
        print(f"{name:<12} {q:>7} {r:>7.2f}x {psnr(vol, d):>7.2f} {ssim_via_cli('pf_dec.raw'):>8.5f}")
    print()
