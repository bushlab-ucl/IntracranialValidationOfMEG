"""SEEG amplitude modulation during MEG-VE bursts (Figure 3).

For each session, band and contact:
  1. Drop the samples that are bad in either modality, z-score, band-pass
     and take the Hilbert envelope of the SEEG and the MEG-VE (broadband
     beamformer).
  2. Exclude contacts with a weak SEEG band envelope.
  3. Detect bursts on the MEG-VE envelope.
  4. Mean SEEG envelope inside / outside the MEG bursts, and a
     block-shuffle permutation test of their log-ratio.

Output: ``<out_dir>/analysis/amp_modulation/contacts.csv``
"""

import numpy as np
import pandas as pd

from ivmeg import bursts, config, data, paths


def process(sid):
    s = data.load(sid)
    df = s.contacts
    keep = (df["region_aal24"] != "Undefined").to_numpy()
    ie = data.zscore(s.seeg[keep][:, s.good])
    me = data.zscore(s.meg[keep][:, s.good])
    names = df["bipolar_name"][keep].tolist()
    regions = df["region_aal24"][keep].tolist()

    rows = []
    for band in config.BANDS:
        ie_env = bursts.envelope(ie, s.sfreq, band)
        me_env = bursts.envelope(me, s.sfreq, band)
        for i in np.flatnonzero(bursts.strong_contacts(ie_env)):
            train = bursts.burst_train(me_env[i], s.sfreq, band)
            a_in, a_out = bursts.readout(ie_env[i], train)
            z, p = bursts.block_shuffle_null(ie_env[i], train, s.sfreq)
            rows.append(dict(
                session=sid, contact=names[i], region=regions[i], band=band,
                a_in=a_in, a_out=a_out, mod_index=bursts.mod_index(a_in, a_out),
                block_z=z, block_p=p,
            ))
    return rows


rows = []
for sid in config.SESSIONS:
    rows += process(sid)
    print(f"{sid}: done", flush=True)
out = paths.out_dir("analysis", "amp_modulation") / "contacts.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Saved {out}")
