"""Band power of the SEEG and the MEG virtual electrodes (VEs) per region.

Each contact is z-scored within session, its PSD is estimated with Welch's
method (4 s segments) over the samples that are good in both modalities, and
band power is the integral of the PSD over the band. Per (session, region,
band), the log10 of the median power over contacts.

Cross-modal agreement per band is the Pearson correlation pooled over all
(session, region) points, and per (region, band) across sessions (BH-FDR
over all cells).

Output: ``<out_dir>/analysis/band_power/``
- ``contacts.csv``: band power per (session, contact, band)
- ``regions.csv``: log10 median power per (session, region, band)
- ``stats.npz``: pooled and per-region correlations

Usage: python band_power.py
"""

import numpy as np
import pandas as pd

from ivmeg import config, data, paths, spectral


def contact_band_power(x, sfreq):
    """(n_contacts, n_bands) band power of z-scored data ``x``."""
    freqs, psd = spectral.welch(x, sfreq)
    return np.stack([spectral.band_power(freqs, psd, b)
                     for b in config.BANDS.values()], axis=1)


def region_table(contacts, value_cols):
    """log10 of the median over contacts per (session, region, band)."""
    df = contacts[contacts["region"].isin(data.analysed_regions())]
    g = df.groupby(["session", "region", "band"], sort=False)
    out = g[value_cols].agg(spectral.log10_median)
    out["n_contacts"] = g.size()
    return out.reset_index()


out_dir = paths.out_dir("analysis", "band_power")
rows = []
for sid in config.SESSIONS:
    s = data.load(sid)
    seeg = contact_band_power(data.zscore(s.seeg)[:, s.good], s.sfreq)
    meg = contact_band_power(data.zscore(s.meg)[:, s.good], s.sfreq)
    for k, row in s.contacts.iterrows():
        for b, band in enumerate(config.BANDS):
            rows.append({"session": sid, "contact": row["bipolar_name"],
                         "region": row["region_aal24"], "band": band,
                         "seeg": seeg[k, b], "meg": meg[k, b]})
    print(f"{sid}: {len(s.contacts)} contacts")
contacts = pd.DataFrame(rows)
contacts.to_csv(out_dir / "contacts.csv", index=False)

regions = region_table(contacts, ["seeg", "meg"])
regions.to_csv(out_dir / "regions.csv", index=False)

bands = list(config.BANDS)
region_names = sorted(regions["region"].unique())
pooled = np.array([spectral.pearson(*regions[regions["band"] == b][
    ["seeg", "meg"]].to_numpy().T) for b in bands])
cell_r = np.full((len(region_names), len(bands)), np.nan)
cell_p = np.full_like(cell_r, np.nan)
for i, r in enumerate(region_names):
    for j, b in enumerate(bands):
        sel = regions[(regions["region"] == r) & (regions["band"] == b)]
        cell_r[i, j], cell_p[i, j] = spectral.pearson(
            sel["seeg"].to_numpy(), sel["meg"].to_numpy())
np.savez(out_dir / "stats.npz", bands=bands, regions=region_names,
         pooled_r=pooled[:, 0], pooled_p=pooled[:, 1], region_r=cell_r,
         region_p=cell_p, region_p_fdr=spectral.bh_fdr(cell_p))

print("\nPooled correlation (median over contacts):")
for b, (r, p) in zip(bands, pooled):
    print(f"  {b:>10s}: r = {r:+.2f}, p = {p:.2g}")
