"""Precompute all per-frame quantities, then assemble the 3-panel movie.

Panel 1: the rendered g-band image of the binary (to scale, mas axes).
Panel 2: g- and i-band lightcurves with a moving phase cursor.
Panel 3: g2(B) at 400 and 800 nm, sampled every 10 m from 10 to 150 m,
         with a smooth underlying curve; baseline oriented along the
         instantaneous projected separation axis (or a fixed PA).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from .hbt import baseline_vectors_along_pa, check_frequency, dft_points, split_frequency
from .orbit import SkyPositions, sky_positions
from .params import BinarySystem, GridConfig, MovieConfig, planck
from .photometry import anchored_mags, apparent_ab_mag
from .render import check_extent, render_kernel, spectral_weights
from .spectral import _auto_chunk

DISPLAY_HALF_PX = 128   # display crop half-width after 2x downsampling
DISPLAY_BIN = 2
# The sky panel is a false-color RGB composite: renders at these
# wavelengths weighted by the absolute Planck surface brightness and
# white-balanced to a reference temperature, so a hot star looks blue
# and a cool one orange (a 4900 K blackbody is nearly flat in raw
# B_lambda across the optical -- without the balance it reads gray).
# The gamma stretch is applied to luminance only, preserving the color
# saturation while lifting the much fainter cool star above black.
RGB_DISPLAY_NM = (700.0, 550.0, 440.0)  # R, G, B channels
WHITE_REF_TEFF = 7500.0                 # appears white/neutral
DISPLAY_GAMMA = 0.43


@dataclass
class FrameData:
    system: BinarySystem
    grid: GridConfig
    cfg: MovieConfig
    phase: np.ndarray          # orbital phase in [0, 1)
    disp_imgs: np.ndarray      # (nf, m, m) display images (g band)
    disp_extent_mas: float     # display half-width in mas
    mags: dict                 # band name -> (nf,) relative magnitudes
    g2_fine: np.ndarray        # (nf, n_lambda, n_fine)
    g2_pts: np.ndarray         # (nf, n_lambda, n_baselines)
    pa: np.ndarray             # (nf,) baseline position angle used [rad]


def _display_crop(img: np.ndarray, n: int) -> np.ndarray:
    half = DISPLAY_HALF_PX * DISPLAY_BIN
    c = n // 2
    crop = img[c - half:c + half, c - half:c + half]
    m = 2 * DISPLAY_HALF_PX
    return crop.reshape(m, DISPLAY_BIN, m, DISPLAY_BIN).mean(axis=(1, 3))


def render_display_rgb(pos, system, grid) -> np.ndarray:
    """(m, m, 3) true-temperature-color sky image at one epoch.

    Each channel is the rendered image at RGB_DISPLAY_NM multiplied by
    the secondary's Planck surface brightness at that wavelength (the
    render is in units of B_lambda(T2), so this restores the absolute
    inter-channel scaling) and divided by the white-reference Planck
    spectrum: a pixel of star s carries B_lambda(T_s)/B_lambda(T_ref)."""
    from .render import render_image

    m = 2 * DISPLAY_HALF_PX
    rgb = np.empty((m, m, 3), np.float32)
    for c, lam_nm in enumerate(RGB_DISPLAY_NM):
        img = np.asarray(render_image(pos, system, lam_nm, grid))
        lam_m = lam_nm * 1e-9
        rgb[..., c] = (_display_crop(img, grid.n)
                       * planck(lam_m, system.secondary.teff)
                       / planck(lam_m, WHITE_REF_TEFF))
    return rgb


def stretch_rgb(disp: np.ndarray) -> np.ndarray:
    """Normalize a (..., 3) RGB stack to [0, 1], gamma-stretching the
    luminance only so per-pixel color ratios (saturation) are preserved."""
    lum = disp.max(axis=-1)
    lum_stretched = np.clip(lum / lum.max(), 0.0, 1.0) ** DISPLAY_GAMMA
    scale = np.where(lum > 0.0, lum_stretched / np.maximum(lum, 1e-30), 0.0)
    return np.clip(disp * scale[..., None], 0.0, 1.0)


@partial(jax.jit, static_argnames=("n", "n_band", "n_disp", "chunk"))
def _frames_jit(x1, y1, x2, y2, front2,            # (nf,)
                r1, r2,                            # scalars
                w1, u1, u2,                        # (n_wl,) all wavelengths
                fx_hi, fx_lo, fy_hi, fy_lo,        # (nf, n_g2, K)
                n: int, n_band: int, n_disp: int, chunk: int):
    """Everything the movie needs, for all frames, in one jitted scan:
    per frame the binary is rendered at every wavelength (bands, display
    RGB, g2 channels) by one vmapped kernel call, the band images are
    summed, the display images cropped and binned, and the g2 channels
    sampled at the K baseline points by the exact DFT."""
    half = DISPLAY_HALF_PX * DISPLAY_BIN
    c = n // 2
    m = 2 * DISPLAY_HALF_PX
    render_all = jax.vmap(render_kernel,
                          in_axes=(None, None, None, None, None, None, None,
                                   0, None, 0, 0, None))

    def one_frame(fr):
        fx1, fy1, fx2, fy2, ffront, fxh, fxl, fyh, fyl = fr
        imgs = render_all(fx1, fy1, fx2, fy2, ffront, r1, r2,
                          w1, jnp.float32(1.0), u1, u2, n)   # (n_wl, n, n)
        flux = jnp.sum(imgs[:n_band], axis=(1, 2))
        crop = imgs[n_band:n_band + n_disp, c - half:c + half, c - half:c + half]
        disp = crop.reshape(n_disp, m, DISPLAY_BIN, m, DISPLAY_BIN).mean(axis=(2, 4))
        vis = jax.vmap(dft_points)(imgs[n_band + n_disp:], fxh, fxl, fyh, fyl)
        return flux, disp, jnp.abs(vis) ** 2

    return jax.lax.map(one_frame,
                       (x1, y1, x2, y2, front2, fx_hi, fx_lo, fy_hi, fy_lo),
                       batch_size=chunk)


def precompute_frames(system: BinarySystem, grid: GridConfig, cfg: MovieConfig,
                      verbose: bool = True,
                      chunk_size: int | None = None) -> FrameData:
    nf = cfg.n_frames
    psi = 2.0 * np.pi * np.arange(nf) / nf  # mean anomaly from periastron
    pos_all = sky_positions(psi, system)
    check_extent(pos_all, system, grid)

    pa_used = (np.asarray(pos_all.pa, dtype=float) if cfg.baseline_pa == "follow"
               else np.full(nf, np.radians(float(cfg.baseline_pa))))
    band_nm = [lam for _, lam in cfg.bands]
    all_nm = [*band_nm, *RGB_DISPLAY_NM, *cfg.wavelengths_nm]
    u1, u2, w1 = spectral_weights(np.asarray(all_nm), system)

    # baseline points: the fine curve followed by the marked baselines
    fine_b = cfg.fine_baselines_m
    pts_b = np.asarray(cfg.baselines_m, dtype=float)
    b_all = np.concatenate([fine_b, pts_b])
    bvec = np.stack([baseline_vectors_along_pa(b_all, pa) for pa in pa_used])  # (nf, K, 2)
    lam_m = np.asarray(cfg.wavelengths_nm, dtype=float) * 1e-9
    f = bvec[:, None, :, :] / lam_m[None, :, None, None] * grid.pixel_scale_rad
    check_frequency(f, grid.n)
    fx_hi, fx_lo = split_frequency(f[..., 0])
    fy_hi, fy_lo = split_frequency(f[..., 1])
    f32 = lambda a: jnp.asarray(a, dtype=jnp.float32)

    chunk = max(1, (_auto_chunk() if chunk_size is None else chunk_size) // 4)
    chunk = min(chunk, nf)
    s = grid.pixel_scale_mas
    if verbose:
        print(f"  rendering {nf} frames x {len(all_nm)} wavelengths "
              f"(chunks of {chunk}) ...", flush=True)
    flux, disp_raw, vis2 = _frames_jit(
        f32(pos_all.x1 / s), f32(pos_all.y1 / s),
        f32(pos_all.x2 / s), f32(pos_all.y2 / s), jnp.asarray(pos_all.front2),
        jnp.float32(system.angular_radius_mas(system.primary) / s),
        jnp.float32(system.angular_radius_mas(system.secondary) / s),
        w1, u1, u2, f32(fx_hi), f32(fx_lo), f32(fy_hi), f32(fy_lo),
        grid.n, len(cfg.bands), len(RGB_DISPLAY_NM), chunk)
    flux = np.asarray(flux, dtype=float)                 # (nf, n_band)
    disp_raw = np.asarray(disp_raw, dtype=np.float32)    # (nf, 3, m, m)
    g2 = 1.0 + np.asarray(vis2, dtype=np.float32)        # (nf, n_g2, K)
    if verbose:
        print(f"  frame {nf}/{nf}", flush=True)

    # display RGB: restore the absolute inter-channel Planck scaling and
    # white-balance (see render_display_rgb)
    disp = np.transpose(disp_raw, (0, 2, 3, 1)).copy()
    for ch, lam_nm in enumerate(RGB_DISPLAY_NM):
        lam = lam_nm * 1e-9
        disp[..., ch] *= (planck(lam, system.secondary.teff)
                          / planck(lam, WHITE_REF_TEFF))

    mags = {band: anchored_mags(apparent_ab_mag(flux[:, j], lam_nm, system, grid),
                                band, system)
            for j, (band, lam_nm) in enumerate(cfg.bands)}
    extent = DISPLAY_HALF_PX * DISPLAY_BIN * grid.pixel_scale_mas
    return FrameData(system, grid, cfg, psi / (2 * np.pi), disp, extent,
                     mags, g2[:, :, :fine_b.size], g2[:, :, fine_b.size:],
                     pa_used)


def make_movie(fd: FrameData, path: str, verbose: bool = True) -> None:
    cfg = fd.cfg
    nf = len(fd.phase)
    colors = {"g": "tab:blue", "i": "tab:red"}
    lam_colors = ["tab:blue", "tab:red"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.2))
    fig.suptitle(f"HBT intensity interferometry: {fd.system.name}", fontsize=14)

    # --- panel 1: sky image (true-temperature-color RGB) ---
    e = fd.disp_extent_mas
    disp_rgb = stretch_rgb(fd.disp_imgs)
    im = ax1.imshow(disp_rgb[0], origin="lower", extent=[-e, e, -e, e])
    ax1.set_xlabel("x [mas]")
    ax1.set_ylabel("y [mas]")
    ax1.set_title("Sky image (temperature color)")
    phase_txt = ax1.text(0.03, 0.95, "", transform=ax1.transAxes, color="w", fontsize=10)

    # --- panel 2: lightcurves ---
    for band, _ in cfg.bands:
        ax2.plot(fd.phase, fd.mags[band], color=colors.get(band, None),
                 label=f"{band} band")
    cursor = ax2.axvline(fd.phase[0], color="k", lw=1, alpha=0.7)
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    ax2.set_xlabel("Orbital phase")
    ax2.set_ylabel("Apparent magnitude")
    ax2.set_title("Lightcurve")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    # --- panel 3: g2(B) ---
    fine_b = cfg.fine_baselines_m
    lines, marks = [], []
    for j, lam_nm in enumerate(cfg.wavelengths_nm):
        (ln,) = ax3.plot(fine_b, fd.g2_fine[0, j], color=lam_colors[j],
                         label=f"{lam_nm:.0f} nm")
        (mk,) = ax3.plot(cfg.baselines_m, fd.g2_pts[0, j], "o",
                         color=lam_colors[j], ms=5)
        lines.append(ln)
        marks.append(mk)
    ax3.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax3.axhline(2.0, color="gray", lw=0.8, ls="--")
    ax3.set_xlim(0, cfg.fine_baseline_max_m)
    ax3.set_ylim(0.95, 2.05)
    ax3.set_xlabel("Baseline B [m]")
    ax3.set_ylabel(r"$g^{(2)}(B)$")
    pa_note = ("baseline along projected separation" if cfg.baseline_pa == "follow"
               else f"baseline PA = {cfg.baseline_pa}\N{DEGREE SIGN}")
    ax3.set_title(f"$g^{{(2)}}(B)$ ({pa_note})")
    ax3.legend(loc="upper right")
    ax3.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(k):
        im.set_data(disp_rgb[k])
        phase_txt.set_text(f"phase = {fd.phase[k]:.3f}")
        cursor.set_xdata([fd.phase[k], fd.phase[k]])
        for j in range(len(cfg.wavelengths_nm)):
            lines[j].set_ydata(fd.g2_fine[k, j])
            marks[j].set_ydata(fd.g2_pts[k, j])
        return [im, phase_txt, cursor, *lines, *marks]

    anim = FuncAnimation(fig, update, frames=nf, blit=False)
    writer = FFMpegWriter(fps=cfg.fps, bitrate=4000)
    progress = (lambda k, n: print(f"  writing frame {k + 1}/{n}", flush=True)
                if (k % 40 == 0 or k == n - 1) else None) if verbose else None
    anim.save(path, writer=writer, dpi=130, progress_callback=progress)
    plt.close(fig)
