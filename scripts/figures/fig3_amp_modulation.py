"""Figure 3: SEEG amplitude modulation index during MEG-VE bursts.

One point per (session, region): A_burst and A_non-burst are averaged over
the region's contacts and the index formed once. Filled points have a
significantly positive Stouffer-combined block-shuffle test.

Reads ``<out_dir>/analysis/amp_modulation/contacts.csv``.
Output: ``figures/fig3_amp_modulation.png``
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ivmeg import bursts, config, paths

matplotlib.use("Agg")

def aggregate(df, regions):
    """One row per (session, region) with mod_index and sig."""
    rows = []
    for (sid, region), g in df.groupby(["session", "region"], sort=False):
        g = g[np.isfinite(g["a_in"]) & np.isfinite(g["a_out"])]
        if region not in regions or len(g) < bursts.MIN_CONTACTS_PER_SESSION:
            continue
        a_in, a_out = g["a_in"].mean(), g["a_out"].mean()
        if not (a_in > 0 and a_out > 0):
            continue
        rows.append(dict(session=sid, region=region,
                         mod_index=bursts.mod_index(a_in, a_out),
                         sig=bursts.stouffer_significant(g["block_z"])))
    return pd.DataFrame(rows)


df = pd.read_csv(paths.out_dir("analysis", "amp_modulation")
                 / "contacts.csv")
regions = bursts.kept_regions(df)
agg = pd.concat([aggregate(df[df["band"] == b], regions).assign(band=b)
                 for b in config.BANDS])

fig, ax = plt.subplots(figsize=(11.0, 5.8), constrained_layout=True)
print(f"{'band':>11} {'n':>4} {'median':>8} {'%>0':>6} {'%sig':>6}")
shown = bursts.box_strip(ax, agg, "mod_index",
                  "{band:>11} {n:4d} {med:8.4f} {pos:6.1f} {sig:6.1f}",
                  np.random.default_rng(0))
ax.set_ylabel("SEEG Amplitude Modulation Index\n"
              r"$(A_{\rm burst} - A_{\rm non\text{-}burst})\,/\,"
              r"(A_{\rm burst} + A_{\rm non\text{-}burst})$", fontsize=15)

# Percentage of significant points under each box
y0, y1 = ax.get_ylim()
pad = (y1 - y0) * 0.09
ax.set_ylim(y0 - pad, y1)
for pos, band in enumerate(config.BANDS, start=1):
    frac = 100 * agg[agg["band"] == band]["sig"].mean()
    ax.text(pos, y0 - pad * 0.55, f"{frac:.0f}% sig.", ha="center",
            va="center", fontsize=12, color="0.25")
bursts.legends(fig, shown)

out = paths.figures() / "fig3_amp_modulation.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
