"""CCA between the SEEG contacts of each region and the MEG virtual
electrodes at the same contacts (Fig S1).

For each session, AAL24 region with at least 3 contacts and frequency band:
the z-scored, band-passed SEEG against the z-scored MEG-VEs at the same
contacts, with a block-shuffle permutation null (100 permutations, 1 s
blocks). Two versions of the VEs:

- broadband (Fig S1A): one beamformer, VEs band-pass filtered here,
- per band (Fig S1B): a beamformer for each band (already band-passed).

Output: ``<out_dir>/analysis/cca/ve.npz`` and ``ve_perband.npz``
"""

import numpy as np

from ivmeg import cca, config, data, paths


def trials(sid, rng, perband):
    s = data.load(sid)
    groups = cca.region_groups(s.contacts)
    if not groups:
        return []
    block_n = int(cca.BLOCK_LEN_S * s.sfreq)

    if perband:
        seeg = data.zscore(s.seeg)
        per_band = {}
        for band, limits in config.BANDS.items():
            sb = data.load(sid, band=band)
            per_band[band] = (data.bandpass(seeg, s.sfreq, limits),
                              data.zscore(sb.meg), sb.good)
    else:
        seeg, meg = data.zscore(s.seeg), data.zscore(s.meg)
        per_band = {band: (data.bandpass(seeg, s.sfreq, limits),
                           data.bandpass(meg, s.sfreq, limits), s.good)
                    for band, limits in config.BANDS.items()}

    rows = []
    for region, idx in groups.items():
        row = {"session": sid, "region": region, "n_contacts": len(idx),
               "mean_depth_mm": s.contacts["depth_mm"][idx].mean()}
        for band in config.BANDS:
            seeg_b, meg_b, good = per_band[band]
            common_len = (int(good.sum()) // block_n) * block_n
            X = seeg_b[idx][:, good].T[:common_len]
            Y = meg_b[idx][:, good].T[:common_len]
            try:
                r_obs, null = cca.cca_null(X, Y, block_n, rng)
            except np.linalg.LinAlgError:
                continue
            for k, v in cca.summarise(r_obs, null).items():
                row[f"{k}_{band}"] = v
        rows.append(row)
    print(f"{sid}: {len(rows)} regions", flush=True)
    return rows


for perband, name in ((False, "ve"), (True, "ve_perband")):
    rng = np.random.default_rng(cca.SEED)
    rows = [r for sid in config.SESSIONS
            for r in trials(sid, rng, perband)]
    out = paths.out_dir("analysis", "cca") / f"{name}.npz"
    cca.save_trials(out, rows, config.BANDS, n_perm=cca.N_PERM)
    print(f"{len(rows)} (session, region) trials; saved {out}")
