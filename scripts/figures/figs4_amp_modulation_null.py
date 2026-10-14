"""Figure S4: spatial specificity of the SEEG amplitude modulation.

One point per (session, region): the matched-minus-random modulation index
averaged over the region's contacts. Filled points have a significantly
positive Stouffer-combined random-VE test.

Reads ``<out_dir>/analysis/amp_modulation_null/contacts.csv``.
Output: ``figures/figs4_amp_modulation_null.png``
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from ivmeg import bursts, config, paths

matplotlib.use("Agg")


def aggregate(df, regions):
    rows = []
    for (sid, region), g in df.groupby(["session", "region"], sort=False):
        z = g["z"].to_numpy()
        if region not in regions or len(g) < bursts.MIN_CONTACTS_PER_SESSION \
                or not np.isfinite(g["diff"]).any() \
                or not np.isfinite(z).any():
            continue
        rows.append(dict(session=sid, region=region, diff=g["diff"].mean(),
                         sig=bursts.stouffer_significant(z)))
    return pd.DataFrame(rows)


df = pd.read_csv(paths.out_dir("analysis", "amp_modulation_null")
                 / "contacts.csv")
regions = bursts.kept_regions(df)
agg = pd.concat([aggregate(df[df["band"] == b], regions).assign(band=b)
                 for b in config.BANDS])

print(f"{'band':>11} {'contacts':>8} {'obs':>7} {'random':>7} "
      f"{'%>rand':>7} {'%p<.05':>7}")
for band in config.BANDS:
    b = df[df["band"] == band]
    print(f"{band:>11} {len(b):8d} {b['obs_mod'].mean():7.4f} "
          f"{b['null_mean'].mean():7.4f} "
          f"{100 * np.mean(b['obs_mod'] > b['null_mean']):7.1f} "
          f"{100 * np.mean(b['p'] < 0.05):7.1f}")

fig, ax = plt.subplots(figsize=(11.0, 5.8), constrained_layout=True)
print(f"\n{'band':>11} {'n':>4} {'median':>8} {'%>0':>6} {'%sig':>6}")
shown = bursts.box_strip(ax, agg, "diff",
                  "{band:>11} {n:4d} {med:8.4f} {pos:6.1f} {sig:6.1f}",
                  np.random.default_rng(0))
ax.set_ylabel("Spatial specificity of MEG-burst → SEEG\n"
              "modulation  (matched − random location)", fontsize=15)
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
for pos, band in enumerate(config.BANDS, start=1):
    frac = 100 * agg[agg["band"] == band]["sig"].mean()
    ax.text(pos, 0.015, f"{frac:.0f}% sig.", transform=trans, ha="center",
            va="bottom", fontsize=11, color="0.25")
bursts.legends(fig, shown)

out = paths.figures() / "figs4_amp_modulation_null.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
