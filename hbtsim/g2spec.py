"""Movie of the measurable g2(lambda) spectrum, one frame per hour.

Each frame shows what the telescope pair would measure in a one-hour
integration with the source dispersed over the detector array: per
spectral channel a simulated data point g2(lambda) = 1 + |V|^2 + noise
with its 1-sigma error bar (the noise-equivalent |V|^2 from the photon
budget, snr.vis2_noise), on top of the true curve.  As the binary moves
through its 3.96-day orbit the fringe pattern sweeps across the band
and washes out through the eclipses, where the analytic visibility is
invalid -- |V|^2(B, lambda) therefore comes from the batched FFT
pipeline (hbtsim.spectral), which handles the overlapping disks.

g2 is shown in the ideal Siegert normalization 1 + |V|^2, consistent
with the orbit movie; the error bars carry the instrument response
(jitter, dead time, dark counts, unpolarized-light factor 1/2).

Workflow (compute is GPU-friendly, rendering needs ffmpeg):

    python -m hbtsim.g2spec --compute-only        # writes the .npz
    python -m hbtsim.g2spec --render-only         # .npz -> .mp4
    python -m hbtsim.g2spec                       # both
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

import numpy as np

from .movie import (DISPLAY_BIN, DISPLAY_HALF_PX, render_display_rgb,
                    stretch_rgb)
from .orbit import SkyPositions, sky_positions
from .params import BETA_AUR, BinarySystem, GridConfig
from .snr import (C2PU, SPAD_LAMBDA, Detector, Observation, Spectrograph,
                  Telescope, system_ab_mag, vis2_noise)
from .spectral import spectral_vis2


def precompute(system: BinarySystem = BETA_AUR, baseline_m: float = 50.0,
               spectrograph: Spectrograph = Spectrograph(),
               t_int_s: float = 3600.0,
               telescope: Telescope = C2PU,
               detector: Detector = SPAD_LAMBDA,
               grid: GridConfig = GridConfig(),
               chunk_size: int | None = None,
               seed: int = 42, verbose: bool = True) -> dict:
    """Per-hour |V|^2(lambda), 1-sigma errors and one noisy realization
    over one orbital period.  Returns a dict of arrays (np.savez-able)."""
    period_h = system.period_days * 24.0
    hours = np.arange(0.0, np.floor(period_h) + 0.5)  # 0..95 for Beta Aur
    phases = hours / period_h
    nm = spectrograph.channel_centers_nm
    det1 = replace(detector, n_pixels=1)

    n_e, n_c = hours.size, nm.size
    m_disp = 2 * DISPLAY_HALF_PX
    vis2 = np.empty((n_e, n_c), np.float32)
    flux = np.empty((n_e, n_c), np.float32)
    disp = np.empty((n_e, m_disp, m_disp, 3), np.float32)
    for k, ph in enumerate(phases):
        pos = SkyPositions(*(np.asarray(v) for v in
                             sky_positions(2 * np.pi * ph, system)))
        v2, fl = spectral_vis2(pos, [baseline_m], nm, system, grid,
                               chunk_size=chunk_size, return_flux=True)
        vis2[k] = np.asarray(v2)[:, 0]
        flux[k] = np.asarray(fl)
        disp[k] = render_display_rgb(pos, system, grid)
        if verbose and (k % 12 == 0 or k == n_e - 1):
            print(f"  epoch {k + 1}/{n_e} (phase {ph:.3f})", flush=True)

    # per-epoch, per-channel magnitude: out-of-eclipse anchored value plus
    # the eclipse dimming from the rendered flux
    mag0 = np.array([system_ab_mag(system, float(l)) for l in nm])
    dmag = -2.5 * np.log10(flux / flux.max(axis=0, keepdims=True))
    mags = mag0[None, :] + dmag

    sigma = np.empty((n_e, n_c), np.float32)
    for k in range(n_e):
        for j, lam_nm in enumerate(nm):
            obs = Observation(wavelength_nm=float(lam_nm),
                              filter_width_nm=spectrograph.channel_width_nm,
                              t_int_s=t_int_s)
            sigma[k, j] = vis2_noise(float(mags[k, j]), obs,
                                     telescope1=telescope, detector1=det1)

    rng = np.random.default_rng(seed)
    noisy = vis2 + sigma * rng.standard_normal(vis2.shape).astype(np.float32)

    return dict(hours=hours, phases=phases, channel_nm=nm,
                vis2=vis2, sigma=sigma, noisy=noisy, mags=mags,
                disp=disp,
                disp_extent_mas=np.float64(DISPLAY_HALF_PX * DISPLAY_BIN
                                           * grid.pixel_scale_mas),
                baseline_m=np.float64(baseline_m), t_int_s=np.float64(t_int_s),
                tel_diameter_m=np.float64(telescope.diameter_m),
                period_h=np.float64(period_h),
                system_name=np.str_(system.name))


def _bin_channels(noisy: np.ndarray, sigma: np.ndarray, nm: np.ndarray,
                  nbin: int):
    """Inverse-variance binning of nbin adjacent channels (display only:
    per-channel sigma varies strongly across the band, so a weighted mean
    is the natural combination)."""
    n_e, n_c = noisy.shape
    n_b = n_c // nbin
    w = (1.0 / sigma**2)[:, :n_b * nbin].reshape(n_e, n_b, nbin)
    y = noisy[:, :n_b * nbin].reshape(n_e, n_b, nbin)
    wsum = w.sum(axis=2)
    return (nm[:n_b * nbin].reshape(n_b, nbin).mean(axis=1),
            (w * y).sum(axis=2) / wsum,
            1.0 / np.sqrt(wsum))


def make_movie(data: dict, path: str, fps: int = 8, nbin: int = 8,
               verbose: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    nm = data["channel_nm"]
    hours, phases = data["hours"], data["phases"]
    g2_true = 1.0 + data["vis2"]
    g2_meas = 1.0 + data["noisy"]
    sigma = data["sigma"]
    n_e = hours.size
    nm_b, noisy_b, sigma_b = _bin_channels(data["noisy"], sigma, nm, nbin)
    g2_b = 1.0 + noisy_b
    has_sky = "disp" in data

    if has_sky:
        fig, (ax_sky, ax) = plt.subplots(
            1, 2, figsize=(13.5, 5.5), width_ratios=[1.0, 1.7])
        e = float(data["disp_extent_mas"])
        if data["disp"].ndim == 4:  # RGB temperature-color composite
            disp = stretch_rgb(data["disp"])
            im = ax_sky.imshow(disp[0], origin="lower", extent=[-e, e, -e, e])
            ax_sky.set_title("Sky image (temperature color)")
        else:  # legacy single-band npz
            disp = data["disp"]
            im = ax_sky.imshow(disp[0], origin="lower",
                               extent=[-e, e, -e, e], cmap="inferno",
                               vmin=0.0, vmax=float(disp.max()))
            ax_sky.set_title("Sky image (g band)")
        ax_sky.set_xlabel("x [mas]")
        ax_sky.set_ylabel("y [mas]")
    else:
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        im = None

    (true_ln,) = ax.plot(nm, g2_true[0], color="tab:orange", lw=1.5,
                         label="model", zorder=4)
    (chan_ln,) = ax.plot(nm, g2_meas[0], ".", color="tab:blue", ms=2,
                         alpha=0.25, zorder=2,
                         label="per channel (1 h)")
    container = ax.errorbar(nm_b, g2_b[0], yerr=sigma_b[0], fmt="o",
                            color="tab:blue", ms=4, elinewidth=1.2,
                            capsize=0, zorder=3,
                            label=f"{nbin}-channel bins (1 h)")
    meas_ln, _, (bars,) = container
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.axhline(2.0, color="gray", lw=0.8, ls="--")
    ax.set_xlim(nm[0] - 5, nm[-1] + 5)
    ax.set_ylim(0.9, 2.05)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(r"$g^{(2)}(\lambda)$")
    tel_txt = (f"2 × {float(data['tel_diameter_m']):.0f} m, "
               if "tel_diameter_m" in data else "")
    ax.set_title(f"{data['system_name']} — {tel_txt}"
                 f"B = {float(data['baseline_m']):.0f} m, "
                 f"{nm.size} channels, "
                 f"{float(data['t_int_s']) / 3600:.0f} h per point")
    label = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    def segments(k):
        return [np.column_stack([np.full(2, x),
                                 [y - e, y + e]])
                for x, y, e in zip(nm_b, g2_b[k], sigma_b[k])]

    def update(k):
        true_ln.set_ydata(g2_true[k])
        chan_ln.set_ydata(g2_meas[k])
        meas_ln.set_ydata(g2_b[k])
        bars.set_segments(segments(k))
        label.set_text(f"t = {hours[k]:.0f} h   "
                       f"orbital phase = {phases[k]:.3f}")
        out = [true_ln, chan_ln, meas_ln, bars, label]
        if im is not None:
            im.set_data(disp[k])
            out.append(im)
        return out

    anim = FuncAnimation(fig, update, frames=n_e, blit=False)
    progress = (lambda k, n: print(f"  writing frame {k + 1}/{n}", flush=True)
                if (k % 24 == 0 or k == n - 1) else None) if verbose else None
    anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=3000), dpi=140,
              progress_callback=progress)
    plt.close(fig)


def main(argv=None) -> None:
    from .params import SYSTEMS

    p = argparse.ArgumentParser(description="g2(lambda) movie with error bars")
    p.add_argument("--system", choices=sorted(SYSTEMS), default="betaaur")
    p.add_argument("--npz", default=None,
                   help="data file (default output/g2spec_<system>.npz)")
    p.add_argument("--out", default=None,
                   help="movie path (default output/g2spec_<system>.mp4)")
    p.add_argument("--compute-only", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--baseline", type=float, default=50.0,
                   help="baseline in m (C2PU: 15, Keck pair: 85)")
    p.add_argument("--diameter", type=float, default=C2PU.diameter_m,
                   help="telescope diameter in m (C2PU: 1, Keck: 10)")
    p.add_argument("--throughput", type=float, default=C2PU.throughput)
    p.add_argument("--time", type=float, default=3600.0,
                   help="integration time per frame in s")
    p.add_argument("--channels", type=int, default=320)
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    system = SYSTEMS[args.system]
    npz = args.npz or f"output/g2spec_{args.system}.npz"
    out = args.out or f"output/g2spec_{args.system}.mp4"
    os.makedirs(os.path.dirname(npz) or ".", exist_ok=True)
    if not args.render_only:
        spec = Spectrograph(n_channels=args.channels)
        tel = Telescope(diameter_m=args.diameter, throughput=args.throughput)
        print(f"Computing g2(lambda) every hour over one period of "
              f"{system.name} (2 x {tel.diameter_m:.0f} m, "
              f"B = {args.baseline:.0f} m, {args.channels} channels) ...")
        data = precompute(system=system, baseline_m=args.baseline,
                          spectrograph=spec, t_int_s=args.time, telescope=tel,
                          chunk_size=args.chunk, seed=args.seed)
        np.savez_compressed(npz, **data)
        print(f"Wrote {npz}")
    if not args.compute_only:
        data = dict(np.load(npz, allow_pickle=False))
        print(f"Rendering {out} ...")
        make_movie(data, out, fps=args.fps)
        print("Done.")


if __name__ == "__main__":
    main()
