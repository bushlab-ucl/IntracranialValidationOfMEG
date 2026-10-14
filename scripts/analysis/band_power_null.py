"""Spatial specificity of the band power correlation (random-location null).

The SEEG band power is kept fixed, and the MEG band power of each
(session, region, band) is replaced by that of the same number of virtual
electrodes (VEs) drawn at random from the session's random-location null
VEs (beamformed for that band). The pooled correlation is recomputed for
N_NULL draws; the Monte-Carlo p-values are BH-FDR corrected over bands.

Needs ``band_power.py`` to have been run.

Output: ``<out_dir>/analysis/band_power/null.npz``

Usage: python band_power_null.py
"""

import mne
import numpy as np
import pandas as pd

from ivmeg import config, data, paths, spectral

N_NULL = 200
SEED = 0


def null_band_power(sid, band):
    """Band power of each random-location null VE of a session."""
    raw_s = mne.io.read_raw_fif(paths.seeg_bipolar(sid))
    raw_n = mne.io.read_raw_fif(paths.null_ve(sid, "random", band),
                                preload=True)
    n = min(raw_s.n_times, raw_n.n_times)
    good = data.good_mask(raw_s, n) & data.good_mask(raw_n, n)
    x = data.zscore(raw_n.get_data()[:, :n])[:, good]
    freqs, psd = spectral.welch(x, raw_n.info["sfreq"])
    return spectral.band_power(freqs, psd, config.BANDS[band])


out_dir = paths.out_dir("analysis", "band_power")
regions = pd.read_csv(out_dir / "regions.csv")
rng = np.random.default_rng(SEED)

null = np.full((len(regions), N_NULL), np.nan)
for sid in config.SESSIONS:
    for band in config.BANDS:
        power = null_band_power(sid, band)
        rows = np.flatnonzero((regions["session"] == sid)
                              & (regions["band"] == band))
        for i in rows:
            k = int(regions["n_contacts"][i])
            if len(power) <= k:
                continue
            for j in range(N_NULL):
                sel = rng.choice(len(power), size=k, replace=False)
                null[i, j] = spectral.log10_median(power[sel])
    print(f"{sid}: done")

bands = list(config.BANDS)
obs_r = np.full(len(bands), np.nan)
null_r = np.full((len(bands), N_NULL), np.nan)
p, z = np.full(len(bands), np.nan), np.full(len(bands), np.nan)
for b, band in enumerate(bands):
    sel = (regions["band"] == band).to_numpy()
    x = regions["seeg"].to_numpy()[sel]
    obs_r[b] = spectral.pearson(x, regions["meg"].to_numpy()[sel])[0]
    for j in range(N_NULL):
        null_r[b, j] = spectral.pearson(x, null[sel, j])[0]
    p[b], z[b] = spectral.null_test(obs_r[b], null_r[b])
p_fdr = spectral.bh_fdr(p)

np.savez(out_dir / "null.npz", bands=bands, obs_r=obs_r, null_r=null_r,
         p=p, p_fdr=p_fdr, z=z, null_meg=null)
print(f"\n{'band':>10s}  {'r_obs':>6s}  {'p_fdr':>7s}  {'z':>6s}")
for b, band in enumerate(bands):
    print(f"{band:>10s}  {obs_r[b]:+.2f}  {p_fdr[b]:7.3g}  {z[b]:+6.2f}")
