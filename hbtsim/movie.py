"""Precompute all per-frame quantities, then assemble the 3-panel movie.

Panel 1: the rendered g-band image of the binary (to scale, mas axes).
Panel 2: g- and i-band lightcurves with a moving phase cursor.
Panel 3: g2(B) at 400 and 800 nm, sampled every 10 m from 10 to 150 m,
         with a smooth underlying curve; baseline oriented along the
         instantaneous projected separation axis (or a fixed PA).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from . import hbt
from .orbit import SkyPositions, sky_positions
from .params import BinarySystem, GridConfig, MovieConfig
from .photometry import apparent_ab_mag, band_flux
from .render import render_image

DISPLAY_HALF_PX = 128   # display crop half-width after 2x downsampling
DISPLAY_BIN = 2


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


def precompute_frames(system: BinarySystem, grid: GridConfig, cfg: MovieConfig,
                      verbose: bool = True) -> FrameData:
    nf = cfg.n_frames
    psi = 2.0 * np.pi * np.arange(nf) / nf  # psi=0: greatest separation
    pos_all = sky_positions(psi, system)

    fine_b = cfg.fine_baselines_m
    disp = np.empty((nf, 2 * DISPLAY_HALF_PX, 2 * DISPLAY_HALF_PX), np.float32)
    fluxes = {band: np.empty(nf) for band, _ in cfg.bands}
    g2_fine = np.empty((nf, len(cfg.wavelengths_nm), fine_b.size), np.float32)
    g2_pts = np.empty((nf, len(cfg.wavelengths_nm), len(cfg.baselines_m)), np.float32)
    pa_used = np.empty(nf)

    for k in range(nf):
        pos = SkyPositions(*(np.asarray(v)[k] for v in pos_all))
        pa = float(pos.pa) if cfg.baseline_pa == "follow" else np.radians(float(cfg.baseline_pa))
        pa_used[k] = pa

        for band, lam_nm in cfg.bands:
            img = np.asarray(render_image(pos, system, lam_nm, grid))
            fluxes[band][k] = band_flux(img)
            if band == "g":
                disp[k] = _display_crop(img, grid.n)

        for j, lam_nm in enumerate(cfg.wavelengths_nm):
            img = render_image(pos, system, lam_nm, grid)
            v2map = hbt.vis2_map(img, grid.pad)
            lam_m = lam_nm * 1e-9
            g2_fine[k, j] = np.asarray(hbt.g2_of_baseline(v2map, fine_b, lam_m, pa, grid))
            g2_pts[k, j] = np.asarray(hbt.g2_of_baseline(v2map, np.asarray(cfg.baselines_m), lam_m, pa, grid))

        if verbose and (k % 20 == 0 or k == nf - 1):
            print(f"  frame {k + 1}/{nf}", flush=True)

    mags = {band: apparent_ab_mag(fluxes[band], lam_nm, system, grid)
            for band, lam_nm in cfg.bands}
    extent = DISPLAY_HALF_PX * DISPLAY_BIN * grid.pixel_scale_mas
    return FrameData(system, grid, cfg, psi / (2 * np.pi), disp, extent,
                     mags, g2_fine, g2_pts, pa_used)


def make_movie(fd: FrameData, path: str, verbose: bool = True) -> None:
    cfg = fd.cfg
    nf = len(fd.phase)
    colors = {"g": "tab:blue", "i": "tab:red"}
    lam_colors = ["tab:blue", "tab:red"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.2))
    fig.suptitle(f"HBT intensity interferometry: {fd.system.name}", fontsize=14)

    # --- panel 1: sky image ---
    e = fd.disp_extent_mas
    vmax = fd.disp_imgs.max()
    im = ax1.imshow(fd.disp_imgs[0], origin="lower", extent=[-e, e, -e, e],
                    cmap="inferno", vmin=0.0, vmax=vmax)
    ax1.set_xlabel("x [mas]")
    ax1.set_ylabel("y [mas]")
    ax1.set_title("Sky image (g band)")
    phase_txt = ax1.text(0.03, 0.95, "", transform=ax1.transAxes, color="w", fontsize=10)

    # --- panel 2: lightcurves ---
    for band, _ in cfg.bands:
        ax2.plot(fd.phase, fd.mags[band], color=colors.get(band, None),
                 label=f"{band} band")
    cursor = ax2.axvline(fd.phase[0], color="k", lw=1, alpha=0.7)
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    ax2.set_xlabel("Orbital phase")
    ax2.set_ylabel("Apparent magnitude (AB)")
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
        im.set_data(fd.disp_imgs[k])
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
