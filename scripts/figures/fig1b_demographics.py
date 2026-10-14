"""Figure 1B: age and sex of the participants.

Output: ``figures/fig1b_demographics.png``
"""

import matplotlib
import numpy as np

from ivmeg import paths, release

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOUR = "#4C72B0"

df = release.participants()
age = df["age"].to_numpy(dtype=float)
sex = df["sex"].value_counts().reindex(["F", "M"]).fillna(0).astype(int)
print(f"{len(df)} participants, {df['n_sessions'].sum()} sessions")
print(f"Age: median {np.median(age):.0f}, range {age.min():.0f}-{age.max():.0f}")
print(f"Sex: F={sex['F']}, M={sex['M']}")
print(f"Years with epilepsy: median {df['years_with_epilepsy'].median():.0f}, "
      f"range {df['years_with_epilepsy'].min():.0f}-"
      f"{df['years_with_epilepsy'].max():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.6, 1.0]})

ax = axes[0]
bins = np.arange(np.floor(age.min() / 5) * 5, np.ceil(age.max() / 5) * 5 + 5, 5)
ax.hist(age, bins=bins, color=COLOUR, edgecolor="k", linewidth=0.6, alpha=0.95)
ax.set_xlabel("Age (years)", fontsize=14)
ymax = max(2, int(np.histogram(age, bins=bins)[0].max()) + 1)
ax.set_ylim(0, ymax)
ax.set_yticks(np.arange(0, ymax + 1, 1))

ax = axes[1]
ax.bar(["Female", "Male"], sex.values, color=COLOUR, edgecolor="k",
       linewidth=0.6, width=0.55, alpha=0.95)
ax.set_ylim(0, sex.max() + 1.5)
ax.set_yticks(np.arange(0, sex.max() + 2, 1))

for ax in axes:
    ax.set_ylabel("Number of patients", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)

out = paths.figures() / "fig1b_demographics.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
