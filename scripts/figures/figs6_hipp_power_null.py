"""Figure S6: hippocampal power correlation vs the matched-depth null.

Usage: python figs6_hipp_power_null.py
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from ivmeg import paths
from ivmeg.spectral import plot_null_hists


d = np.load(paths.out_dir("analysis", "hipp_power") / "null.npz")
plot_null_hists(d["obs_r"], d["null_r"], d["p_fdr"], "Depth-matched null",
                paths.figures() / "figs6_hipp_power_null.png",
                text_y=0.97)
