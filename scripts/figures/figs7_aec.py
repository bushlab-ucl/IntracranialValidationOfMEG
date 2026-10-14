"""Figure S7: AEC of the SEEG and MEG-VEs.

Panels (following the caption):
  A ``figs7a_edge_coverage.png``  sessions in which each region pair could
                                  be estimated (SEEG)
  B ``figs7b_meg_ve_aec.png``     group-average MEG-VE AEC per band
  C ``figs7c_seeg_aec.png``       group-average SEEG AEC per band
  D ``figs7d_aec_scatter.png``    MEG-VE vs SEEG AEC per (session, edge)

Run ``scripts/analysis/aec.py`` first.
"""

import matplotlib
import numpy as np

from ivmeg import config, paths
from ivmeg.plotting import BAND_LABELS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IN = paths.out_dir("analysis", "aec")
OUT = paths.figures()


def title(band):
    lo, hi = config.BANDS[band]
    return f"{BAND_LABELS[band]}  ({lo:g}–{hi:g} Hz)"


def region_ticks(ax, regions):
    ax.set_xticks(range(len(regions)))
    ax.set_yticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=90, fontsize=8)
    ax.set_yticklabels(regions, fontsize=8)


def plot_connectome(modality, label, filename):
    d = np.load(IN / f"connectome_{modality}.npz")
    regions, M = d["regions"], d["aec"]
    vmax = float(np.nanpercentile(np.abs(M[np.isfinite(M)]), 99))
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11), constrained_layout=True)
    for ax, band, m in zip(axes.flat, config.BANDS, M):
        im = ax.imshow(np.ma.masked_invalid(m), vmin=-vmax, vmax=vmax,
                       cmap="RdBu_r", interpolation="nearest")
        region_ticks(ax, regions)
        ax.set_title(title(band), fontsize=14)
        fig.colorbar(im, ax=ax, shrink=0.85).set_label(label, fontsize=11)
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / filename}")
    return d


# A: edge coverage (the same for every band)
d = np.load(IN / "connectome_seeg.npz")
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(d["n_sess"][0], cmap="viridis")
region_ticks(ax, d["regions"])
ax.set_title("Sessions per edge", fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.85, label="n sessions")
fig.tight_layout()
fig.savefig(OUT / "figs7a_edge_coverage.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'figs7a_edge_coverage.png'}")

# B, C: group connectomes
plot_connectome("meg", "AEC (orthogonalised)", "figs7b_meg_ve_aec.png")
plot_connectome("seeg", "AEC", "figs7c_seeg_aec.png")

# D: cross-modal scatter
d = np.load(IN / "connectome_corr.npz")
sessions = sorted({s for b in config.BANDS for s in d[f"session_{b}"]})
cmap = plt.get_cmap("tab20", max(len(sessions), 3))
colour = {s: cmap(i) for i, s in enumerate(sessions)}
fig, axes = plt.subplots(1, len(config.BANDS),
                         figsize=(3.9 * len(config.BANDS), 4.5),
                         constrained_layout=True)
for i, (ax, band) in enumerate(zip(axes, config.BANDS)):
    x, y = d[f"seeg_{band}"], d[f"meg_{band}"]
    ax.scatter(x, y, c=[colour[s] for s in d[f"session_{band}"]], s=16,
               alpha=0.6, linewidth=0)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color="black",
            linewidth=1.0, linestyle="--", alpha=0.7)
    ax.text(0.04, 0.96, f"corr. = {float(d[f'r_{band}']):+.2f}\n"
            f"p = {float(d[f'p_{band}']):.1g}", transform=ax.transAxes,
            va="top", ha="left", fontsize=14,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2.0))
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.08 * (hi - lo) if hi > lo else 0.05
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.5",
            linewidth=0.8, linestyle=":", alpha=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel("SEEG AEC", fontsize=16)
    if i == 0:
        ax.set_ylabel("MEG-VE AEC", fontsize=16)
    ax.set_title(title(band), fontsize=18)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "figs7d_aec_scatter.png", dpi=300, bbox_inches="tight")
print(f"Saved {OUT / 'figs7d_aec_scatter.png'}")
