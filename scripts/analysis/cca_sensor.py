"""CCA between the SEEG contacts of each region and the MEG sensors (Fig 1D).

For each session, AAL24 region with at least 3 contacts and frequency band:
the z-scored, band-passed SEEG of the region's contacts against the
z-scored, band-passed MEG sensors. The sensors enter as an orthonormal
basis of their column space (the SSS rank, ~64-68), so the null directions
of the 306 channels cannot inflate the correlation. Significance by a
block-shuffle permutation null (100 permutations, 1 s blocks).

Also saves the leading canonical pair: weights (filters) and patterns
(topographies) on the MEG sensors and the SEEG contacts.

Output: ``<out_dir>/analysis/cca/sensor.npz``
"""

import mne
import numpy as np

from ivmeg import cca, config, data, paths


def scatter(v, cols, n):
    out = np.full(n, np.nan)
    out[cols] = v
    return out


rng = np.random.default_rng(cca.SEED)
rows, meg_ch_names = [], None
for sid in config.SESSIONS:
    s = data.load(sid, meg="sensor")
    groups = cca.region_groups(s.contacts)
    if not groups:
        continue

    # Sensor patterns are saved on the full 306-channel layout (NaN at
    # this session's bad channels) so they line up across sessions
    info = mne.io.read_info(paths.meg_preproc(sid))
    all_meg = [info["ch_names"][i]
               for i in mne.pick_types(info, meg=True, exclude=[])]
    if meg_ch_names is None:
        meg_ch_names = all_meg
    elif all_meg != meg_ch_names:
        raise RuntimeError(f"{sid}: MEG channel layout differs")
    cols = np.array([meg_ch_names.index(c) for c in s.meg_names])

    seeg, meg = data.zscore(s.seeg), data.zscore(s.meg)
    block_n = int(cca.BLOCK_LEN_S * s.sfreq)
    common_len = (int(s.good.sum()) // block_n) * block_n

    per_band = {}
    for band, limits in config.BANDS.items():
        Y = data.bandpass(meg, s.sfreq, limits)[:, s.good].T[:common_len]
        per_band[band] = (data.bandpass(seeg, s.sfreq, limits),
                          *cca.ortho_basis(Y - Y.mean(axis=0)))

    for region, idx in groups.items():
        row = {"session": sid, "region": region, "n_contacts": len(idx),
               "mean_depth_mm": s.contacts["depth_mm"][idx].mean(),
               "contacts": s.contacts["bipolar_name"][idx].tolist()}
        for band in config.BANDS:
            seeg_b, U_y, V_k, w_k = per_band[band]
            X = seeg_b[idx][:, s.good].T[:common_len]
            r_obs, null, ex = cca.sensor_cca_null(X, U_y, block_n, rng)
            if ex is None:
                continue
            for k, v in cca.summarise(r_obs, null).items():
                row[f"{k}_{band}"] = v
            nt = ex["nt"]
            p_meg, w_meg, p_seeg, w_seeg = cca.fix_sign(
                cca.sensor_pattern(ex["b"], V_k, w_k, nt),
                cca.sensor_weights(ex["b"], V_k, w_k),
                cca.seeg_pattern(ex["a"], ex["R"], nt),
                cca.seeg_weights(ex["a"], ex["R"]),
            )
            n_meg = len(meg_ch_names)
            row[f"w_meg_{band}"] = scatter(w_meg, cols, n_meg)
            row[f"pat_meg_{band}"] = scatter(p_meg, cols, n_meg)
            row[f"w_seeg_{band}"] = w_seeg
            row[f"pat_seeg_{band}"] = p_seeg
            row[f"sensor_rank_{band}"] = V_k.shape[1]
        rows.append(row)
    print(f"{sid}: {len(groups)} regions, sensor rank "
          f"{per_band['delta'][1].shape[1]}", flush=True)

bands = list(config.BANDS)
max_ct = max(r["n_contacts"] for r in rows)

def stack_vec(key, length):
    out = np.full((len(rows), len(bands), length), np.nan)
    for i, r in enumerate(rows):
        for j, b in enumerate(bands):
            v = r.get(f"{key}_{b}")
            if v is not None:
                out[i, j, :len(v)] = v
    return out

contacts = np.full((len(rows), max_ct), "", dtype=object)
for i, r in enumerate(rows):
    contacts[i, :len(r["contacts"])] = r["contacts"]
out = paths.out_dir("analysis", "cca") / "sensor.npz"
cca.save_trials(
    out, rows, bands,
    w_meg=stack_vec("w_meg", len(meg_ch_names)),
    pat_meg=stack_vec("pat_meg", len(meg_ch_names)),
    meg_ch_names=np.array(meg_ch_names),
    w_seeg=stack_vec("w_seeg", max_ct),
    pat_seeg=stack_vec("pat_seeg", max_ct),
    contacts=contacts.astype(str),
    sensor_rank=np.array([[r.get(f"sensor_rank_{b}", np.nan)
                           for b in bands] for r in rows]),
    n_perm=cca.N_PERM,
)
print(f"{len(rows)} (session, region) trials; saved {out}")
