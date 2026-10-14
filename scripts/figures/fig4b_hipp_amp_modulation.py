"""Figure 4B: hippocampal SEEG amplitude modulation, true vs matched-depth.

Points are sessions. Above each band:
- temporal: one-sided Wilcoxon signed-rank test of the true modulation > 0,
- spatial: paired one-sided Wilcoxon test of true > matched-depth null.

Reads ``<out_dir>/analysis/hipp_amp_modulation/sessions.csv``.
Output: ``figures/fig4b_hipp_amp_modulation.png``
"""

from math import floor, log10

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from ivmeg import config, paths
from ivmeg.plotting import BAND_LABELS

matplotlib.use("Agg")

TRUE_C = "#C0392B"
NULL_C = "#B0B4BA"


def wilcoxon(x):
    x = x[np.isfinite(x)]
    if x.size < 2 or np.allclose(x, 0):
        return np.nan
    try:
        return stats.wilcoxon(x, alternative="greater").pvalue
    except ValueError:
        return np.nan


def annotate(p):
    if not np.isfinite(p):
        return "n.s."
    if p <= 1e-4:
        return "p<0.0001 ***"
    stars = next((s for t, s in [(0.001, "***"), (0.01, "**"), (0.05, "*"),
                                 (0.10, "†")] if p < t), "")
    s = f"{round(p, max(1, -int(floor(log10(p))))):g}"
    return f"p={'1.0' if s == '1' else s} {stars}".rstrip()


plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.9})
df = pd.read_csv(paths.out_dir("analysis", "hipp_amp_modulation")
                 / "sessions.csv")
bands = [b for b in config.BANDS if b in set(df["band"])]
true = {b: df[df["band"] == b]["true_mod"].to_numpy() for b in bands}
null = {b: df[df["band"] == b]["null_mod"].to_numpy() for b in bands}
p_temporal = {b: wilcoxon(true[b]) for b in bands}
p_spatial = {b: wilcoxon(true[b] - null[b]) for b in bands}

allv = np.concatenate([np.concatenate([true[b], null[b]]) for b in bands])
vmax, vmin = np.nanmax(allv), min(0.0, np.nanmin(allv))
span = vmax - vmin
y_spatial, y_temporal = vmax + 0.10 * span, vmax + 0.22 * span

fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
rng = np.random.default_rng(0)

def draw(y, x, color):
    y = y[np.isfinite(y)]
    ax.boxplot(y, positions=[x], widths=0.36, showfliers=False,
               patch_artist=True,
               medianprops=dict(color="#1a1a1a", lw=1.5),
               whiskerprops=dict(color=color, lw=1.1),
               capprops=dict(color=color, lw=1.1),
               boxprops=dict(facecolor=color, alpha=0.28, edgecolor=color,
                             lw=1.1))
    ax.scatter(x + (rng.random(y.size) - 0.5) * 0.17, y, s=24,
               color=color, edgecolor="white", linewidth=0.4, zorder=3,
               alpha=0.95)

for i, b in enumerate(bands):
    draw(true[b], i - 0.20, TRUE_C)
    draw(null[b], i + 0.20, NULL_C)
    ax.text(i, y_temporal, annotate(p_temporal[b]), ha="center",
            va="center", fontsize=9.5, color=TRUE_C, fontweight="bold")
    ax.text(i, y_spatial, annotate(p_spatial[b]), ha="center",
            va="center", fontsize=9.5, color="#404040")
nb = len(bands)
ax.text(nb - 0.32, y_temporal, "temporal", ha="left", va="center",
        fontsize=10.5, color=TRUE_C, fontweight="bold")
ax.text(nb - 0.32, y_spatial, "spatial", ha="left", va="center",
        fontsize=10.5, color="#404040")
ax.axhline(0.0, color="#404040", lw=1.0, ls=(0, (4, 3)), zorder=1)
ax.text(nb + 0.78, 0.01 * span, "chance", ha="right", va="bottom",
        fontsize=8.5, color="#404040", style="italic")

ax.set_xlim(-0.85, nb + 0.85)
ax.set_ylim(vmin - 0.04 * span, y_temporal + 0.12 * span)
ax.set_xticks(range(nb))
ax.set_xticklabels([BAND_LABELS[b] for b in bands], fontsize=13)
ax.set_ylabel("SEEG amplitude modulation by MEG-VE\n"
              r"hippocampal bursts  $(A_{\rm burst}-A_{\rm non\text{-}burst})/"
              r"(A_{\rm burst}+A_{\rm non\text{-}burst})$", fontsize=11)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(handles=[
    Patch(facecolor=TRUE_C, alpha=0.28, edgecolor=TRUE_C,
          label="true hippocampal VE"),
    Patch(facecolor=NULL_C, alpha=0.28, edgecolor=NULL_C,
          label="matched-depth null")],
    loc="upper right", bbox_to_anchor=(0.86, 0.72), fontsize=9.5,
    frameon=False, handlelength=1.2, handleheight=1.2)

out = paths.figures() / "fig4b_hipp_amp_modulation.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
for b in bands:
    print(f"{b:>11} (n={true[b].size}): temporal p={p_temporal[b]:.4g}  "
          f"spatial p={p_spatial[b]:.4g}")
