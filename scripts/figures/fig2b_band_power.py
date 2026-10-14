"""Figure 2B: SEEG vs MEG virtual electrode (VE) band power per band.

Each point is a region in one session (log10 of the median power over its
contacts); the annotated correlation pools all points and is descriptive
(spatial specificity is tested against the random-location null, Fig S3).

Usage: python fig2b_band_power.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ivmeg import atlas, config, paths
from ivmeg.plotting import BAND_LABELS


df = pd.read_csv(paths.out_dir("analysis", "band_power") / "regions.csv")
colors = atlas.region_colors()
bands = list(config.BANDS)

fig, axes = plt.subplots(1, len(bands), figsize=(3.9 * len(bands), 4.5),
                         constrained_layout=True)
for ax, band in zip(axes, bands):
    sel = df[df["band"] == band]
    x, y = sel["seeg"].to_numpy(), sel["meg"].to_numpy()
    ax.scatter(x, y, c=[colors[r] for r in sel["region"]], s=44,
               edgecolor="k", linewidth=0.5, alpha=0.9)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color="black",
            linewidth=1.0, linestyle="--", alpha=0.7)
    r = stats.pearsonr(x, y)[0]
    ax.text(0.96, 0.04, f"corr. = {r:+.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=15,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none",
                      pad=2.0))

    # Equal axes so the y = x line is at 45 degrees
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.08 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.plot(lim, lim, color="0.5", linewidth=0.8, linestyle=":",
            alpha=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel(r"SEEG log$_{10}$ power", fontsize=16)
    if band == bands[0]:
        ax.set_ylabel(r"MEG-VE log$_{10}$ power", fontsize=16)
    lo_hz, hi_hz = config.BANDS[band]
    ax.set_title(f"{BAND_LABELS[band]}  ({lo_hz:g}–{hi_hz:g} Hz)",
                 fontsize=18)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

regions = sorted(df["region"].unique())
handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                      markerfacecolor=colors[r], markeredgecolor="k",
                      markersize=10, label=r) for r in regions]
fig.legend(handles=handles, loc="lower center",
           bbox_to_anchor=(0.5, -0.20),
           ncol=int(np.ceil(len(regions) / 2)), frameon=False,
           fontsize=15, handletextpad=0.6, columnspacing=2.6,
           labelspacing=0.6)
out = paths.figures() / "fig2b_band_power.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
