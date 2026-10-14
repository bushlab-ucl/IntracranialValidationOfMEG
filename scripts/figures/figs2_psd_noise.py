"""Figure S2: per-region SEEG PSDs with and without added white noise.

Per session and region, the median log10 PSD over the bipolar SEEG
contacts; per region, the mean +/- SEM over sessions.

Usage: python figs2_psd_noise.py
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from ivmeg import config, data, paths
from ivmeg.spectral import EPS, MEG_COLOR, SEEG_COLOR, plot_region_psds

NOISE_SD = 1.0  # as in analysis/psd.py


files = {s: np.load(paths.out_dir("analysis", "psd") / f"{s}.npz")
         for s in config.SESSIONS}
df = pd.concat([pd.DataFrame({"session": s, "region_aal24": d["region"]})
                for s, d in files.items()])
clean, noisy, n_contacts, xyz = {}, {}, {}, {}
for sid, d in files.items():
    for r in data.analysed_regions(df):
        idx = np.flatnonzero(d["region"] == r)
        if not idx.size:
            continue
        clean.setdefault(r, []).append(
            np.median(np.log10(d["seeg_clean"][idx] + EPS), axis=0))
        noisy.setdefault(r, []).append(
            np.median(np.log10(d["seeg_noisy"][idx] + EPS), axis=0))
        n_contacts[r] = n_contacts.get(r, 0) + idx.size
        xyz.setdefault(r, []).append(d["mni"][idx])
centroids = {r: tuple(np.median(np.concatenate(v), axis=0))
             for r, v in xyz.items()}
traces = {r: [(SEEG_COLOR, clean[r]), (MEG_COLOR, noisy[r])]
          for r in clean}
freqs = next(iter(files.values()))["noise_freqs"]
plot_region_psds(
    freqs, traces, n_contacts, centroids,
    [(SEEG_COLOR, "SEEG"),
     (MEG_COLOR, rf"SEEG + noise ($\sigma$ = {NOISE_SD:g})")],
    paths.figures() / "figs2_psd_noise.png",
)
