"""Figure 2A: per-region PSDs of the SEEG and MEG virtual electrodes (VEs).

Per session and region, the median log10 PSD over contacts (SEEG and the VE
at the same contacts); per region, the mean +/- SEM over sessions.

Usage: python fig2a_psd.py
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from ivmeg import config, data, paths
from ivmeg.spectral import EPS, MEG_COLOR, SEEG_COLOR, plot_region_psds


def load_region_psds(seeg_key, other_key, other_names_key):
    """Per region: lists of per-session median log10 PSDs (SEEG, other)
    over the contacts present in both, the total number of contacts and
    the median MNI coordinate."""
    regions = data.analysed_regions()
    seeg, other, n_contacts, xyz = {}, {}, {}, {}
    for sid in config.SESSIONS:
        d = np.load(paths.out_dir("analysis", "psd") / f"{sid}.npz")
        names = list(d["seeg_names"])
        other_idx = {n: i for i, n in enumerate(d[other_names_key])}
        for r in regions:
            idx = [i for i, (n, reg) in enumerate(zip(names, d["region"]))
                   if reg == r and n in other_idx]
            if not idx:
                continue
            seeg.setdefault(r, []).append(
                np.median(np.log10(d[seeg_key][idx] + EPS), axis=0))
            other.setdefault(r, []).append(np.median(np.log10(
                d[other_key][[other_idx[names[i]] for i in idx]] + EPS),
                axis=0))
            n_contacts[r] = n_contacts.get(r, 0) + len(idx)
            xyz.setdefault(r, []).append(d["mni"][idx])
    centroids = {r: tuple(np.median(np.concatenate(v), axis=0))
                 for r, v in xyz.items()}
    return seeg, other, n_contacts, centroids


seeg, ve, n_contacts, centroids = load_region_psds("seeg", "ve",
                                                   "ve_names")
freqs = np.load(paths.out_dir("analysis", "psd")
                / f"{next(iter(config.SESSIONS))}.npz")["freqs"]
traces = {r: [(SEEG_COLOR, seeg[r]), (MEG_COLOR, ve[r])] for r in seeg}
plot_region_psds(freqs, traces, n_contacts, centroids,
                 [(SEEG_COLOR, "SEEG"), (MEG_COLOR, "MEG-VE")],
                 paths.figures() / "fig2a_psd.png")
