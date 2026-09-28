"""Physics hooks (defaults off): orbital radial velocities and the Doppler
shift of model tables, rotational broadening, interstellar extinction,
air/vacuum channel frames."""

from dataclasses import replace

import numpy as np
import pytest

from hbtsim.orbit import positions_at, sky_positions
from hbtsim.params import ALGOL, BETA_AUR, DELTA_VEL, FluxTable, LDProfile, planck
from hbtsim.sed import (air_to_vacuum, cardelli_extinction, doppler_shift,
                        planck_flux_table, prepare_system, rotational_broaden,
                        vacuum_to_air, with_tables)
from hbtsim.snr import Spectrograph, model_ab_mag, system_ab_mag


def test_beta_aur_semi_amplitudes():
    """Southworth et al. 2007: K_1 = 107.5, K_2 = 111.5 km/s."""
    psi = np.linspace(0, 2 * np.pi, 3601)
    pos = sky_positions(psi, BETA_AUR)
    assert pos.vr1_kms.max() == pytest.approx(107.5, rel=0.02)
    assert pos.vr2_kms.max() == pytest.approx(111.5, rel=0.02)
    assert np.allclose(pos.vr1_kms * BETA_AUR.primary.mass_msun
                       + pos.vr2_kms * BETA_AUR.secondary.mass_msun, 0.0, atol=1e-9)
    # circular orbit: the RV is extreme at quadrature (psi = 0) and zero
    # at conjunction (psi = 90 deg)
    assert abs(float(positions_at(BETA_AUR, 0.0).vr2_kms)) == pytest.approx(111.5, rel=0.02)
    assert abs(float(positions_at(BETA_AUR, 0.25).vr2_kms)) < 0.1


def test_secondary_recedes_at_ascending_node():
    """The ascending node lies along -p (u = 180 deg): dz goes from
    positive (in front) to negative there, i.e. the secondary moves away
    from the observer."""
    for sysm in (BETA_AUR, DELTA_VEL):
        psi = np.linspace(0, 2 * np.pi, 20001)
        pos = sky_positions(psi, sysm)
        front = np.asarray(pos.front2)
        # indices where front2 switches True -> False: the ascending node
        k = np.where(front[:-1] & ~front[1:])[0]
        assert k.size >= 1
        assert np.all(pos.vr2_kms[k] > 0) and np.all(pos.vr1_kms[k] < 0)


def test_doppler_shift_moves_a_line_by_the_orbital_velocity():
    lam = np.linspace(470.0, 500.0, 3001)
    depth = np.exp(-0.5 * ((lam - 486.0) / 0.05) ** 2)
    ft = FluxTable(lam, np.pi * planck(lam * 1e-9, 9350.0) * (1 - 0.7 * depth))
    ld = LDProfile(mu=np.linspace(0, 1, 5), wavelength_nm=lam,
                   intensity=np.tile(np.linspace(0.4, 1.0, 5), (lam.size, 1)))
    star = with_tables(BETA_AUR.primary, ft, ld)
    shifted = doppler_shift(star, 107.0)
    lam_min = shifted.flux_table.wavelength_nm[np.argmin(shifted.flux_table.flux)]
    assert lam_min == pytest.approx(486.0 * (1 + 107e3 / 2.99792458e8), abs=0.02)
    assert doppler_shift(star, 0.0) is star
    # through prepare_system: only when the system opts in
    sysm = replace(BETA_AUR, primary=star, secondary=with_tables(
        BETA_AUR.secondary, planck_flux_table(9200.0, lam), ld))
    spec = Spectrograph.from_resolving_power(5000.0, 480.0, 492.0)
    pos = positions_at(sysm, 0.0)                 # quadrature: |v| ~ 107 km/s
    off = prepare_system(sysm, spec, pos)
    on = prepare_system(replace(sysm, doppler=True), spec, pos)
    k_off = np.argmin(off.primary.flux_table.flux)
    k_on = np.argmin(on.primary.flux_table.flux)
    assert abs(k_on - k_off) in (1, 2)            # ~1.8 channels at R = 5000
    assert np.array_equal(off.primary.flux_table.wavelength_nm, spec.channel_centers_nm)


def test_rotational_broadening_conserves_equivalent_width():
    lam = np.linspace(480.0, 492.0, 6001)
    cont = np.pi * planck(lam * 1e-9, 9350.0)
    line = 1 - 0.8 * np.exp(-0.5 * ((lam - 486.0) / 0.03) ** 2)
    ft = FluxTable(lam, cont * line)
    ld = LDProfile(mu=np.linspace(0, 1, 4), wavelength_nm=lam,
                   intensity=np.tile(np.linspace(0.5, 1.0, 4), (lam.size, 1)))
    bf, bl = rotational_broaden(ft, ld, 120.0)
    ew = lambda f: np.trapezoid(1 - f / cont, lam)
    assert ew(bf.flux) == pytest.approx(ew(ft.flux), rel=2e-3)
    # FWHM of a narrow line after broadening ~ 1.7 v sin i lambda / c
    prof = 1 - bf.flux / cont
    half = prof > 0.5 * prof.max()
    fwhm = lam[half].max() - lam[half].min()
    expect = 1.7 * 120e3 / 2.99792458e8 * 486.0
    assert fwhm == pytest.approx(expect, rel=0.15)
    assert bl.intensity.shape == ld.intensity.shape
    assert rotational_broaden(ft, ld, 0.0) == (ft, ld)


def test_cardelli_extinction_values():
    assert cardelli_extinction(550.0, 1.0) == pytest.approx(1.0, abs=0.01)
    assert cardelli_extinction(440.0, 1.0) / cardelli_extinction(550.0, 1.0) == pytest.approx(1.32, abs=0.02)
    assert cardelli_extinction(2200.0, 1.0) == pytest.approx(0.11, abs=0.02)   # K band ~0.11 A_V
    with pytest.raises(ValueError):
        cardelli_extinction(250.0, 1.0)
    # reddening dims the model magnitude by A_lambda, nothing else changes
    m0 = model_ab_mag(BETA_AUR, np.array([440.0, 800.0]))
    m1 = model_ab_mag(replace(BETA_AUR, a_v=0.5), np.array([440.0, 800.0]))
    assert np.allclose(m1 - m0, cardelli_extinction(np.array([440.0, 800.0]), 0.5))


def test_air_vacuum_round_trip():
    lam = np.array([400.0, 500.0, 800.0])
    vac = air_to_vacuum(lam)
    assert vac[1] - lam[1] == pytest.approx(0.14, abs=0.01)
    assert np.allclose(vacuum_to_air(vac), lam, atol=1e-6)
    # an "air" spectrograph rebins on the vacuum-converted edges
    lam_t = np.linspace(470.0, 500.0, 3001)
    star = with_tables(BETA_AUR.primary, planck_flux_table(9350.0, lam_t),
                       LDProfile(mu=np.linspace(0, 1, 4), wavelength_nm=lam_t,
                                 intensity=np.ones((lam_t.size, 4))))
    sysm = replace(BETA_AUR, primary=star, secondary=with_tables(
        BETA_AUR.secondary, planck_flux_table(9200.0, lam_t),
        LDProfile(mu=np.linspace(0, 1, 4), wavelength_nm=lam_t, intensity=np.ones((lam_t.size, 4)))))
    spec_v = Spectrograph(lambda_min_nm=480.0, lambda_max_nm=490.0, n_channels=10)
    spec_a = replace(spec_v, frame="air")
    pv = prepare_system(sysm, spec_v).primary.flux_table
    pa = prepare_system(sysm, spec_a).primary.flux_table
    assert pa.wavelength_nm[0] - pv.wavelength_nm[0] == pytest.approx(0.14, abs=0.01)
