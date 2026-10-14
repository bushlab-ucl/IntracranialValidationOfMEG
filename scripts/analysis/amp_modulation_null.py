"""Random-location null for the SEEG amplitude modulation (Figure S4).

Is a contact's SEEG amplitude modulated more by bursts on the MEG-VE at its
own location than by bursts on random-location VEs in a different region?
The SEEG envelope is held fixed and only the burst train is swapped. The
random VEs come from a beamformer for each band, so they are only
Hilbert-enveloped.

Per contact: the observed modulation index against the pool of
random-location indices gives a z-score, a one-sided p-value and the
location-specific excess (observed - mean of the pool).

Output: ``<out_dir>/analysis/amp_modulation_null/contacts.csv``
"""

import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from ivmeg import bursts, config, data, paths

MIN_POOL = 10


def process(sid):
    s = data.load(sid)
    df = s.contacts
    keep = (df["region_aal24"] != "Undefined").to_numpy()
    names = df["bipolar_name"][keep].tolist()
    regions = df["region_aal24"][keep].to_numpy()
    sources = pd.read_csv(paths.null_sources(sid, "random"))
    null_region = dict(zip(sources["channel"], sources["region_aal24"]))

    rows = []
    for band in config.BANDS:
        raw_r = mne.io.read_raw_fif(paths.null_ve(sid, "random", band),
                                    preload=True)
        n = min(s.good.size, raw_r.n_times)
        good = s.good[:n] & data.good_mask(raw_r, n)
        if good.sum() < 100:
            continue
        ie_env = bursts.envelope(data.zscore(s.seeg[keep, :n][:, good]),
                                 s.sfreq, band)
        me_env = bursts.envelope(data.zscore(s.meg[keep, :n][:, good]),
                                 s.sfreq, band)
        rd_env = np.abs(hilbert(data.zscore(raw_r.get_data()[:, :n][:, good]),
                                axis=-1))
        strong = bursts.strong_contacts(ie_env)
        if not strong.any():
            continue

        trains = np.stack([bursts.burst_train(e, s.sfreq, band)
                           for e in rd_env], axis=1)
        has_bursts = trains.sum(axis=0) > 0
        rd_region = np.array([null_region[c] for c in raw_r.ch_names])
        null_mod = bursts.mod_index_matrix(ie_env, trains)

        for i in np.flatnonzero(strong):
            obs = bursts.mod_index(*bursts.readout(
                ie_env[i], bursts.burst_train(me_env[i], s.sfreq, band)))
            if not np.isfinite(obs):
                continue
            pool = null_mod[i, has_bursts & (rd_region != regions[i])]
            pool = pool[np.isfinite(pool)]
            if pool.size < MIN_POOL:
                continue
            sd = pool.std(ddof=1)
            rows.append(dict(
                session=sid, contact=names[i], region=regions[i], band=band,
                obs_mod=obs, null_mean=pool.mean(), diff=obs - pool.mean(),
                z=(obs - pool.mean()) / sd if sd > 1e-12 else np.nan,
                p=(np.sum(pool >= obs) + 1) / (pool.size + 1),
                n_pool=pool.size,
            ))
    return rows


rows = []
for sid in config.SESSIONS:
    rows += process(sid)
    print(f"{sid}: done", flush=True)
out = paths.out_dir("analysis", "amp_modulation_null") / "contacts.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Saved {out}")
