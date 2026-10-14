"""Figure 4A: hippocampal PSDs (top) and power correlation (bottom).

PSD: mean +/- SEM of log10 PSD over all (session, contact) pairs.
Power: one point per session (mean log10 power over its hippocampal
contacts), with the Pearson correlation across sessions (descriptive; spatial
specificity is tested against the matched-depth null, Fig S6).

Usage: python fig4a_hipp_power.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ivmeg import config, paths
from ivmeg.plotting import BAND_LABELS
from ivmeg.spectral import MEG_COLOR, SEEG_COLOR

F_MIN, F_MAX = 1.5, 95.0
POINT_COLOR = "#4C72B0"


def plot_psd(d, filename):
    f = d["freqs"]
    fmask = (f >= F_MIN) & (f <= F_MAX)
    f = f[fmask]

    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    for i, (lo, hi) in enumerate(config.BANDS.values()):
        ax.axvspan(lo, hi, color=["#f0f4fa", "#e3ecf6"][i % 2], alpha=0.6,
                   zorder=0)
    for key, color, label in [("seeg", SEEG_COLOR, "SEEG"),
                              ("meg", MEG_COLOR, "MEG-VE")]:
        log_psd = np.log10(np.maximum(d[key][:, fmask], 1e-30))
        mean = log_psd.mean(axis=0)
        sem = log_psd.std(axis=0) / np.sqrt(log_psd.shape[0])
        ax.fill_between(f, mean - sem, mean + sem, color=color, alpha=0.18,
                        linewidth=0)
        ax.plot(f, mean, color=color, linewidth=2.0, label=label)
    ax.set_xlim(F_MIN, F_MAX)
    ax.set_xlabel("Frequency (Hz)", fontsize=15)
    ax.set_ylabel(r"$\log_{10}\left[\mathrm{PSD}\,/\,(1/\mathrm{Hz})\right]$",
                  fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", alpha=0.2, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", fontsize=13, frameon=False)

    top = ax.secondary_xaxis("top")
    top.set_xticks([(lo + hi) / 2 for lo, hi in config.BANDS.values()])
    top.set_xticklabels([BAND_LABELS[b] for b in config.BANDS], fontsize=13)
    top.tick_params(axis="x", which="minor", bottom=False, top=False)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


def plot_power_corr(df, filename):
    bands = list(config.BANDS)
    fig, axes = plt.subplots(1, len(bands), figsize=(4.0 * len(bands), 4.6),
                             constrained_layout=True)
    for ax, band in zip(axes, bands):
        sel = df[df["band"] == band]
        x, y = sel["seeg"].to_numpy(), sel["meg"].to_numpy()
        ax.scatter(x, y, color=POINT_COLOR, s=60, edgecolor="k",
                   linewidth=0.4, alpha=0.9)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color="black",
                linewidth=1.6, linestyle="--", alpha=0.85)
        r = stats.pearsonr(x, y)[0]
        ax.text(0.04, 0.96, f"corr. = {r:+.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=16,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.85, edgecolor="0.6", linewidth=0.8))
        lo, hi = config.BANDS[band]
        ax.set_title(f"{BAND_LABELS[band]}  ({lo:g}–{hi:g} Hz)", fontsize=20)
        ax.set_xlabel(r"SEEG $\log_{10}$ power", fontsize=18)
        if band == bands[0]:
            ax.set_ylabel(r"MEG-VE $\log_{10}$ power", fontsize=18)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


in_dir = paths.out_dir("analysis", "hipp_power")
plot_psd(np.load(in_dir / "psd.npz"), paths.figures() / "fig4a_hipp_psd.png")
plot_power_corr(pd.read_csv(in_dir / "sessions.csv"),
                paths.figures() / "fig4a_hipp_power_corr.png")
