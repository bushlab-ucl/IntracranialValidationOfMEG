"""Band power and PSDs at the hippocampal contacts (SEEG and MEG virtual
electrodes, VEs).

Hippocampal contacts are the bipolar channels whose midpoint is within 3 mm
of the subject's own hippocampus (FastSurfer; electrodes.is_hippocampal). Samples that are bad in either
modality are dropped, each contact is z-scored and its PSD estimated with
Welch's method (4 s segments). Per (session, band), the mean log10 band
power over contacts (geometric mean); the cross-modal agreement per band is the Pearson
correlation across sessions.

Output: ``<out_dir>/analysis/hipp_power/``
- ``contacts.csv``: band power per (session, contact, band)
- ``sessions.csv``: mean log10 power per (session, band)
- ``psd.npz``: PSD per (session, contact)

Usage: python hipp_power.py
"""

import numpy as np
import pandas as pd

from ivmeg import config, data, paths, spectral


def session_table(contacts):
    """Mean log10 power over contacts per (session, band)."""
    ok = (contacts["seeg"] > 0) & (contacts["meg"] > 0)
    df = contacts[ok].assign(seeg=lambda d: np.log10(d["seeg"]),
                             meg=lambda d: np.log10(d["meg"]))
    g = df.groupby(["session", "band"], sort=False)
    out = g[["seeg", "meg"]].mean()
    out["n_contacts"] = g.size()
    return out.reset_index()


out_dir = paths.out_dir("analysis", "hipp_power")
rows, psd_seeg, psd_meg, psd_session = [], [], [], []
for sid in config.SESSIONS:
    s = data.load(sid)
    hipp = s.contacts.index[s.contacts["hippocampus"]].to_numpy()
    if not hipp.size:
        print(f"{sid}: no hippocampal contacts")
        continue
    seeg = data.zscore(s.seeg[hipp][:, s.good])
    meg = data.zscore(s.meg[hipp][:, s.good])
    freqs, p_seeg = spectral.welch(seeg, s.sfreq)
    _, p_meg = spectral.welch(meg, s.sfreq)
    for k, i in enumerate(hipp):
        for band, limits in config.BANDS.items():
            rows.append({
                "session": sid,
                "contact": s.contacts["bipolar_name"][i],
                "band": band,
                "seeg": spectral.band_power(freqs, p_seeg[k], limits),
                "meg": spectral.band_power(freqs, p_meg[k], limits),
            })
    psd_seeg.append(p_seeg)
    psd_meg.append(p_meg)
    psd_session += [sid] * len(hipp)
    print(f"{sid}: {len(hipp)} hippocampal contacts")

contacts = pd.DataFrame(rows)
contacts.to_csv(out_dir / "contacts.csv", index=False)
sessions = session_table(contacts)
sessions.to_csv(out_dir / "sessions.csv", index=False)
np.savez(out_dir / "psd.npz", freqs=freqs,
         seeg=np.concatenate(psd_seeg), meg=np.concatenate(psd_meg),
         session=np.array(psd_session))

print(f"\n{sessions['session'].nunique()} sessions, "
      f"{len(contacts) // len(config.BANDS)} contacts")
for band in config.BANDS:
    sel = sessions[sessions["band"] == band]
    r, p = spectral.pearson(sel["seeg"].to_numpy(), sel["meg"].to_numpy())
    print(f"  {band:>10s}: r = {r:+.2f}, p = {p:.2g}, n = {len(sel)}")
