"""Figure S3: band power correlation vs the random-location null.

Usage: python figs3_band_power_null.py
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from ivmeg import paths
from ivmeg.spectral import plot_null_hists


d = np.load(paths.out_dir("analysis", "band_power") / "null.npz")
plot_null_hists(d["obs_r"], d["null_r"], d["p_fdr"], "Null distribution",
                paths.figures() / "figs3_band_power_null.png")
