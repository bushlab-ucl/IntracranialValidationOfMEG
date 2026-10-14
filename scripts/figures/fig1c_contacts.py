"""Figure 1C: bipolar SEEG contacts per region.

Contacts are pooled over sessions. Only regions that enter the region-level
analyses are shown (contacts in at least 3 sessions and 10 contacts pooled).

Output: ``figures/fig1c_contact_counts.png`` (bar chart) and
``fig1c_glass_brain.png`` (one marker per region at the median contact
location, sized by the number of contacts).
"""

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from nilearn import plotting

from ivmeg import atlas, config, data, electrodes, paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

df = pd.concat([electrodes.bipolar_midpoints(s).assign(session=s)
                for s in config.SESSIONS], ignore_index=True)
regions = data.analysed_regions(df)
dropped = sorted(set(df["region_aal24"]) - set(regions) - {"Undefined"})
print(f"Dropped regions below the analysis threshold: {dropped}")
df = df[df["region_aal24"].isin(regions)]
counts = df["region_aal24"].value_counts()  # sorted, largest first
print(f"{len(counts)} regions, {counts.sum()} contacts:\n{counts.to_string()}")
colours = atlas.region_colors()

# Bar chart
fig, ax = plt.subplots(figsize=(10, max(7, 0.45 * len(counts))))
ax.barh(range(len(counts)), counts.values,
        color=[colours[r] for r in counts.index])
ax.set_yticks(range(len(counts)))
ax.set_yticklabels(counts.index, fontsize=16)
ax.invert_yaxis()
ax.set_xlabel("Number of bipolar contacts", fontsize=20)
ax.tick_params(axis="x", labelsize=18)
fig.tight_layout()
out = paths.figures() / "fig1c_contact_counts.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

# Glass brain
regions = sorted(regions)
centroids = df.groupby("region_aal24")[["mni_x", "mni_y", "mni_z"]].median()
display = plotting.plot_glass_brain(None, display_mode="lzry", colorbar=False)
for r in regions:
    display.add_markers([centroids.loc[r].to_numpy()],
                        marker_color=[colours[r]],
                        marker_size=int(20 + 5 * counts[r]), alpha=0.85)
handles = [Line2D([0], [0], marker="o", linestyle="", color=colours[r],
                  markersize=14, label=f"{r} (n={counts[r]})")
           for r in regions]
fig = list(display.axes.values())[0].ax.figure
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=15,
           frameon=False, bbox_to_anchor=(0.5, 0.10), handletextpad=0.6,
           columnspacing=1.8, labelspacing=0.6)
out = paths.figures() / "fig1c_glass_brain.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
