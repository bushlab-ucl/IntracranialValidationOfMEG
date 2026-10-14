"""Amplitude envelope correlation (AEC) of the SEEG and the MEG-VEs.

Per session and band:
- SEEG: band-pass (4th-order Butterworth, forwards-backwards), Hilbert
  envelope, z-score over the good samples and correlate over 30 s blocks.
  No orthogonalisation (the bipolar montage removes most volume
  conduction).
- MEG-VE: per-band VEs, pairwise orthogonalised (Hipp et al., 2012) before
  the same 30 s block-averaged correlation.

Then, per band:
- group connectome of each modality: contact-level AEC averaged over
  inter-region contact pairs, then over sessions, keeping edges present in
  at least 3 sessions;
- cross-modal agreement: per-session region matrices, upper-triangular
  edges as paired (SEEG, MEG-VE) points pooled over sessions, Pearson r.

Output: ``<out_dir>/analysis/aec/``
"""

import mne
import numpy as np
from scipy import stats
from scipy.signal import hilbert

from ivmeg import aec, atlas, config, data, electrodes, paths

OUT = paths.out_dir("analysis", "aec")


def session_file(session_id, modality):
    return OUT / f"{session_id}_{modality}.npz"


def process(sid):
    table = electrodes.bipolar_midpoints(sid).set_index("bipolar_name")

    # SEEG: all bipolar channels
    raw = mne.io.read_raw_fif(paths.seeg_bipolar(sid), preload=True)
    names = raw.ch_names
    sfreq = raw.info["sfreq"]
    block_n = int(aec.BLOCK_LEN_S * sfreq)
    good = data.good_mask(raw)
    x = raw.get_data(picks=names)
    M = np.full((len(config.BANDS), len(names), len(names)), np.nan)
    for b, band in enumerate(config.BANDS.values()):
        env = np.abs(hilbert(data.bandpass(x, sfreq, band), axis=-1))
        r = aec.block_corr(data.zscore(env[:, good]), block_n)
        if r is not None:
            M[b] = r
    np.savez(session_file(sid, "seeg"), aec=M, ch_names=names,
             regions=table.loc[names, "region_aal24"].to_numpy(str))

    # MEG-VE: per-band VEs, orthogonalised
    names = mne.io.read_info(paths.ve(sid, "delta"))["ch_names"]
    M = np.full((len(config.BANDS), len(names), len(names)), np.nan)
    for b, band in enumerate(config.BANDS):
        raw = mne.io.read_raw_fif(paths.ve(sid, band), preload=True)
        good = data.good_mask(raw)
        if good.sum() >= block_n:
            analytic = hilbert(raw.get_data(picks=names), axis=-1)
            M[b] = aec.orthogonalised_aec(analytic, good, block_n)
    np.savez(session_file(sid, "meg"), aec=M, ch_names=names,
             regions=table.loc[names, "region_aal24"].to_numpy(str))


def load(sid, modality):
    d = np.load(session_file(sid, modality))
    keep = d["regions"] != "Undefined"
    return (d["aec"][:, keep][:, :, keep], d["ch_names"][keep],
            d["regions"][keep])


def group_connectome(modality):
    """Mean over sessions of each region-pair edge (NaN if the edge is in
    fewer than EDGE_MIN_SESSIONS sessions)."""
    sessions = [load(s, modality) for s in config.SESSIONS]
    seen = {r for _, _, regs in sessions for r in regs}
    region_list = [r for r in atlas.AAL24_NAMES if r in seen]
    n_bands, R = len(config.BANDS), len(region_list)
    total = np.zeros((n_bands, R, R))
    n_sess = np.zeros((n_bands, R, R), dtype=int)
    for M, _, regs in sessions:
        if len(regs) < 2:
            continue
        for b in range(n_bands):
            reg = aec.regional(M[b], regs, region_list)
            ok = np.isfinite(reg)
            total[b] += np.where(ok, reg, 0)
            n_sess[b] += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / n_sess
    mean[n_sess < aec.EDGE_MIN_SESSIONS] = np.nan
    np.savez(OUT / f"connectome_{modality}.npz", regions=region_list,
             bands=list(config.BANDS), aec=mean, n_sess=n_sess)


def cross_modal():
    """Paired (session, edge) SEEG and MEG-VE region-level AEC per band."""
    points = {band: [] for band in config.BANDS}
    for sid in config.SESSIONS:
        M_s, ch_s, regs_s = load(sid, "seeg")
        M_m, ch_m, _ = load(sid, "meg")
        common = [c for c in ch_s if c in set(ch_m)]
        i_s = [list(ch_s).index(c) for c in common]
        i_m = [list(ch_m).index(c) for c in common]
        regs = regs_s[i_s]
        region_list = sorted(set(regs))
        if len(common) < 2 or len(region_list) < 2:
            continue
        iu = np.triu_indices(len(region_list), k=1)
        for b, band in enumerate(config.BANDS):
            v_s = aec.regional(M_s[b][np.ix_(i_s, i_s)], regs, region_list)[iu]
            v_m = aec.regional(M_m[b][np.ix_(i_m, i_m)], regs, region_list)[iu]
            for k in np.flatnonzero(np.isfinite(v_s) & np.isfinite(v_m)):
                points[band].append((sid, v_s[k], v_m[k]))

    out = {"bands": list(config.BANDS)}
    print(f"{'band':>10s}  {'r':>7s}  {'p':>9s}  {'n_edges':>7s}")
    for band, pts in points.items():
        seeg = np.array([p[1] for p in pts])
        meg = np.array([p[2] for p in pts])
        r, p = stats.pearsonr(seeg, meg) if len(pts) >= 3 else (np.nan,) * 2
        print(f"{band:>10s}  {r:+7.3f}  {p:9.3g}  {len(pts):7d}")
        out.update({f"seeg_{band}": seeg, f"meg_{band}": meg,
                    f"session_{band}": [p[0] for p in pts],
                    f"r_{band}": r, f"p_{band}": p})
    np.savez(OUT / "connectome_corr.npz", **out)


for sid in config.SESSIONS:
    process(sid)
    print(f"{sid}: done", flush=True)
group_connectome("seeg")
group_connectome("meg")
cross_modal()
