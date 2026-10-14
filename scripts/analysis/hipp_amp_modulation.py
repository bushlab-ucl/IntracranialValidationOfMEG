"""Hippocampal SEEG amplitude modulation during MEG-VE bursts (Figure 4B).

The Figure 3 modulation index restricted to hippocampal contacts, with a
matched-depth spatial control:

- true: each hippocampal SEEG envelope gated by the bursts of the MEG-VE at
  its own location.
- null: the same SEEG envelopes gated by the bursts of each matched-depth VE
  (same depth below the scalp, outside the hippocampus).

Both are aggregated to one value per session: A_burst and A_non-burst are
averaged over hippocampal contacts and the index formed once (for the null,
then averaged over the matched-depth VEs).

Output: ``<out_dir>/analysis/hipp_amp_modulation/sessions.csv``
"""

import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from ivmeg import bursts, config, data, paths

MIN_GOOD_S = 60.0


def process(s, band):
    hipp = s.contacts["hippocampus"].to_numpy()
    raw_md = mne.io.read_raw_fif(paths.null_ve(s.id, "matched_depth", band),
                                 preload=True)
    n = min(s.good.size, raw_md.n_times)
    good = s.good[:n] & data.good_mask(raw_md, n)
    n_good = int(good.sum())
    if n_good < s.sfreq * MIN_GOOD_S:
        return None

    ie_env = bursts.envelope(data.zscore(s.seeg[hipp, :n][:, good]),
                             s.sfreq, band)
    me_env = bursts.envelope(data.zscore(s.meg[hipp, :n][:, good]),
                             s.sfreq, band)
    md_env = np.abs(hilbert(data.zscore(raw_md.get_data()[:, :n][:, good]),
                            axis=-1))

    # True: contact-paired MEG-VE bursts
    a = [bursts.readout(e, bursts.burst_train(m, s.sfreq, band))
         for e, m in zip(ie_env, me_env)]
    a = np.array([x for x in a if np.isfinite(x[0])])
    if a.size == 0:
        return None
    true_mod = bursts.mod_index(*a.mean(axis=0))

    # Null: every hippocampal contact gated by each matched-depth VE
    trains = np.stack([bursts.burst_train(e, s.sfreq, band) for e in md_env],
                      axis=1).astype(float)
    n_in = trains.sum(axis=0)
    sums = ie_env @ trains
    with np.errstate(invalid="ignore", divide="ignore"):
        a_in = np.nanmean(sums / n_in, axis=0)
        a_out = np.nanmean((ie_env.sum(axis=1)[:, None] - sums)
                           / (n_good - n_in), axis=0)
    mod = bursts.mod_index(a_in, a_out)
    mod = mod[(n_in > 0) & (n_in < n_good) & np.isfinite(mod)]

    return dict(session=s.id, band=band, true_mod=true_mod,
                null_mod=mod.mean() if mod.size else np.nan,
                n_hipp=len(a), n_matched_depth=mod.size)


rows = []
for sid in config.SESSIONS:
    s = data.load(sid)
    if not s.contacts["hippocampus"].any():
        continue
    for band in config.BANDS:
        r = process(s, band)
        if r is not None and np.isfinite(r["true_mod"]):
            rows.append(r)
    print(f"{sid}: done", flush=True)
df = pd.DataFrame(rows)
out = paths.out_dir("analysis", "hipp_amp_modulation") / "sessions.csv"
df.to_csv(out, index=False)

print(f"\n{'band':>11} {'n':>3} {'true':>8} {'null':>8} {'true>null':>9}")
for band, g in df.groupby("band", sort=False):
    print(f"{band:>11} {len(g):3d} {g['true_mod'].mean():8.4f} "
          f"{g['null_mod'].mean():8.4f} "
          f"{(g['true_mod'] > g['null_mod']).sum():4d}/{len(g)}")
print(f"Saved {out}")
