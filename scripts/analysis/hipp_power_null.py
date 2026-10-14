"""Spatial specificity of the hippocampal power correlation (matched-depth
null).

The SEEG power is kept fixed, and the MEG power of each (session, band) is
replaced by the mean log10 power of the same number of virtual electrodes
(VEs) drawn at random from the session's matched-depth null VEs (at the
depth of its hippocampal contacts, outside the hippocampus). The correlation
across sessions is recomputed for N_NULL draws; the Monte-Carlo p-values are
BH-FDR corrected over bands.

Needs ``hipp_power.py`` to have been run.

Output: ``<out_dir>/analysis/hipp_power/null.npz``

Usage: python hipp_power_null.py
"""

import mne
import numpy as np
import pandas as pd

from ivmeg import config, data, paths, spectral

N_NULL = 200
SEED = 0


def null_band_power(sid, band):
    """Band power of each matched-depth null VE of a session."""
    raw_s = mne.io.read_raw_fif(paths.seeg_bipolar(sid))
    raw_n = mne.io.read_raw_fif(paths.null_ve(sid, "matched_depth", band),
                                preload=True)
    n = min(raw_s.n_times, raw_n.n_times)
    good = data.good_mask(raw_s, n) & data.good_mask(raw_n, n)
    x = data.zscore(raw_n.get_data()[:, :n][:, good])
    freqs, psd = spectral.welch(x, raw_n.info["sfreq"])
    power = spectral.band_power(freqs, psd, config.BANDS[band])
    return power[np.isfinite(power) & (power > 0)]


out_dir = paths.out_dir("analysis", "hipp_power")
df = pd.read_csv(out_dir / "sessions.csv")
sessions = list(dict.fromkeys(df["session"]))
bands = list(config.BANDS)
rng = np.random.default_rng(SEED)

obs = {key: df.pivot(index="session", columns="band", values=key)
       .loc[sessions, bands].to_numpy() for key in
       ("seeg", "meg", "n_contacts")}
null = np.full((len(sessions), len(bands), N_NULL), np.nan)
for i, sid in enumerate(sessions):
    for b, band in enumerate(bands):
        k = int(obs["n_contacts"][i, b])
        if not paths.null_ve(sid, "matched_depth", band).exists():
            print(f"{sid}: no matched-depth null VEs")
            break
        power = null_band_power(sid, band)
        if power.size <= k:
            continue
        for j in range(N_NULL):
            sel = rng.choice(power.size, size=k, replace=False)
            null[i, b, j] = np.mean(np.log10(power[sel]))

obs_r = np.array([spectral.pearson(obs["seeg"][:, b], obs["meg"][:, b])[0]
                  for b in range(len(bands))])
null_r = np.array([[spectral.pearson(obs["seeg"][:, b], null[:, b, j])[0]
                    for j in range(N_NULL)] for b in range(len(bands))])
p, z = np.array([spectral.null_test(obs_r[b], null_r[b])
                 for b in range(len(bands))]).T
p_fdr = spectral.bh_fdr(p)

np.savez(out_dir / "null.npz", sessions=sessions, bands=bands,
         obs_r=obs_r, null_r=null_r, p=p, p_fdr=p_fdr, z=z)
print(f"{len(sessions)} sessions")
print(f"{'band':>10s}  {'r_obs':>6s}  {'p_fdr':>7s}  {'z':>6s}")
for b, band in enumerate(bands):
    print(f"{band:>10s}  {obs_r[b]:+.2f}  {p_fdr[b]:7.3g}  {z[b]:+6.2f}")
