"""Hippocampal SEEG vs MEG sensor CCA, compared with the lead field (Fig S5).

1. CCA patterns. For each session and hemisphere with at least 3
   hippocampal contacts (midpoint within 3 mm of the subject's own
   hippocampus; the hemisphere is the side of the bipolar midpoint, which
   is where the lead field is computed), the leading CCA between the
   z-scored, band-passed hippocampal SEEG and the MEG sensors
   (as in cca_sensor.py, without a null). The MEG pattern is the covariance
   of each sensor with the canonical variate.

2. Modelled field. The leading left singular vector of the magnetometer lead
   fields of the session's hippocampal contacts (each oriented along its
   bipolar axis), from the subject's own forward model.

3. Spin test. |corr| between the observed pattern and the modelled field,
   against surrogates that rotate the observed pattern azimuthally about the
   head (via a real spherical-harmonic expansion on a sphere fitted to the
   magnetometers), which preserves its smoothness. Patients are the unit:
   the group null draws one spin per patient (100,000 times); one-sided p,
   Bonferroni-corrected over the bands.

Output: ``<out_dir>/analysis/hipp_cca/hipp_sensor.npz`` and
``hipp_leadfield_spin.npz``
"""

import mne
import numpy as np
import pandas as pd
from scipy.special import sph_harm_y

from ivmeg import cca, config, data, paths, source

SENSOR_TYPE = "mag"
L_MAX = 6
N_SPIN = 2000
N_GROUP = 100000
SEED = 0


def scatter(v, cols, n):
    out = np.full(n, np.nan)
    out[cols] = v
    return out


def hipp_cca():
    rng = np.random.default_rng(cca.SEED)
    rows, meg_ch_names = [], None
    for sid in config.SESSIONS:
        s = data.load(sid, meg="sensor")
        df = s.contacts
        hemis = {}
        for i, x in enumerate(df["mni_x"]):
            if df["hippocampus"][i]:
                hemis.setdefault("L" if x < 0 else "R", []).append(i)
        hemis = {h: idx for h, idx in hemis.items()
                 if len(idx) >= cca.MIN_CONTACTS}
        if not hemis:
            continue

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

        xyz = df[["mni_x", "mni_y", "mni_z"]].to_numpy()
        for hemi, idx in sorted(hemis.items()):
            row = {"session": sid, "hemisphere": hemi,
                   "n_contacts": len(idx),
                   "mean_depth_mm": df["depth_mm"][idx].mean(),
                   "contacts": df["bipolar_name"][idx].tolist(),
                   "centroid": xyz[idx].mean(axis=0)}
            for band in config.BANDS:
                seeg_b, U_y, V_k, w_k = per_band[band]
                X = seeg_b[idx][:, s.good].T[:common_len]
                r_obs, _, ex = cca.sensor_cca_null(X, U_y, block_n, rng,
                                                   n_perm=0)
                if ex is None:
                    continue
                row[f"r_obs_{band}"] = r_obs
                p_meg, = cca.fix_sign(cca.sensor_pattern(ex["b"], V_k, w_k,
                                                         ex["nt"]))
                row[f"pat_meg_{band}"] = scatter(p_meg, cols,
                                                 len(meg_ch_names))
            rows.append(row)
            print(f"{sid}: hippocampus {hemi}, {len(idx)} contacts, depth "
                  f"{row['mean_depth_mm']:.1f} mm", flush=True)

    bands = list(config.BANDS)
    max_ct = max(r["n_contacts"] for r in rows)
    pat = np.full((len(rows), len(bands), len(meg_ch_names)), np.nan)
    contacts = np.full((len(rows), max_ct), "", dtype=object)
    for i, r in enumerate(rows):
        contacts[i, :len(r["contacts"])] = r["contacts"]
        for j, b in enumerate(bands):
            if f"pat_meg_{b}" in r:
                pat[i, j] = r[f"pat_meg_{b}"]
    out = paths.out_dir("analysis", "hipp_cca") / "hipp_sensor.npz"
    np.savez(
        out,
        sessions=np.array([r["session"] for r in rows]),
        hemispheres=np.array([r["hemisphere"] for r in rows]),
        n_contacts=np.array([r["n_contacts"] for r in rows]),
        mean_depth_mm=np.array([r["mean_depth_mm"] for r in rows]),
        centroids=np.array([r["centroid"] for r in rows]),
        r_obs=np.array([[r.get(f"r_obs_{b}", np.nan) for b in bands]
                        for r in rows]),
        bands=np.array(bands), pat_meg=pat,
        meg_ch_names=np.array(meg_ch_names), contacts=contacts.astype(str),
    )
    n_l = sum(r["hemisphere"] == "L" for r in rows)
    print(f"{len(rows)} trials (L {n_l}, R {len(rows) - n_l}); saved {out}")
    return out


def leadfields(sid, meg_ch_names):
    """Lead fields (n_channels, n_contacts) of the session's virtual
    electrodes, each along its bipolar axis, rows in ``meg_ch_names``
    order."""
    session = paths.osl_session(sid)
    df = pd.read_csv(paths.ve_sources(sid))
    df = df[df["inside"]]
    pos = source.mni_to_mri(session, df[["mni_x", "mni_y", "mni_z"]])
    nn = df[["nn_mri_x", "nn_mri_y", "nn_mri_z"]].to_numpy()
    fwd, _ = source.forward(session, pos, nn)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                       use_cps=True, verbose=False)
    order = {n: i for i, n in enumerate(fwd["info"]["ch_names"])}
    L = fwd["sol"]["data"][[order[n] for n in meg_ch_names]]
    return L, df["bipolar_name"].tolist()


def abscorr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return abs(np.corrcoef(a[m], b[m])[0, 1])


def abscorr_many(P, T):
    """|r| between each row of ``P`` (m, s) and each column of ``T`` (s, k)."""
    Pz = P - P.mean(1, keepdims=True)
    Tz = T - T.mean(0, keepdims=True)
    pn = np.linalg.norm(Pz, axis=1, keepdims=True)
    tn = np.linalg.norm(Tz, axis=0, keepdims=True)
    pn[pn == 0], tn[tn == 0] = np.nan, np.nan
    return np.abs((Pz / pn) @ (Tz / tn))


def fit_sphere(pos):
    A = np.hstack([2 * pos, np.ones((len(pos), 1))])
    return np.linalg.lstsq(A, (pos ** 2).sum(axis=1), rcond=None)[0][:3]


def real_sh(theta, phi, l_max):
    """Real spherical-harmonic design matrix (n, (l_max + 1)^2)."""
    cols = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            if m < 0:
                cols.append(np.sqrt(2) * (-1) ** m * Y.imag)
            elif m == 0:
                cols.append(Y.real)
            else:
                cols.append(np.sqrt(2) * (-1) ** m * Y.real)
    return np.column_stack(cols)


def spin_test(cca_file):
    d = np.load(cca_file)
    bands, sids = d["bands"].astype(str), d["sessions"].astype(str)
    hemis, contacts = d["hemispheres"].astype(str), d["contacts"].astype(str)
    pat = d["pat_meg"]
    meg_ch_names = list(d["meg_ch_names"].astype(str))
    n_tr, n_b = len(sids), len(bands)

    info = mne.io.read_info(paths.meg_preproc(sids[0]))
    info = mne.pick_info(info, mne.pick_types(info, meg=True, exclude=[]))
    assert info["ch_names"] == meg_ch_names, "channel order mismatch"
    sens = mne.pick_types(info, meg=SENSOR_TYPE, exclude=[])

    # Modelled field per trial
    pred_pc1 = np.full((n_tr, len(meg_ch_names)), np.nan)
    pc1_var = np.full(n_tr, np.nan)
    cache = {}
    for t, sid in enumerate(sids):
        if sid not in cache:
            cache[sid] = leadfields(sid, meg_ch_names)
        L_all, names = cache[sid]
        idx = [names.index(c) for c in contacts[t] if c and c in names]
        U, S, _ = np.linalg.svd(L_all[sens][:, idx], full_matrices=False)
        pred_pc1[t, sens] = U[:, 0]
        pc1_var[t] = S[0] ** 2 / (S ** 2).sum()

    # Spherical-harmonic basis and its azimuthal rotations
    pos = np.array([info["chs"][i]["loc"][:3] for i in sens])
    v = pos - fit_sphere(pos)
    rad = np.linalg.norm(v, axis=1)
    theta = np.arccos(np.clip(v[:, 2] / rad, -1, 1))
    phi = np.arctan2(v[:, 1], v[:, 0])
    Y0 = real_sh(theta, phi, L_MAX)
    rng = np.random.default_rng(SEED)
    alpha = rng.uniform(0, 2 * np.pi, N_SPIN)
    Ys = np.stack([real_sh(theta, phi - a, L_MAX) for a in alpha])

    r_pc1 = np.full((n_tr, n_b), np.nan)
    r_spin = np.full((n_tr, n_b, N_SPIN), np.nan)
    fidelity = np.full((n_tr, n_b), np.nan)
    for t in range(n_tr):
        p_t = pred_pc1[t, sens]
        if not np.isfinite(p_t).all():
            continue
        for b in range(n_b):
            obs = pat[t, b, sens]
            ok = np.isfinite(obs)
            if ok.sum() < 20:
                continue
            coef = np.linalg.lstsq(Y0[ok], obs[ok], rcond=None)[0]
            fidelity[t, b] = abscorr((Y0 @ coef)[ok], obs[ok])
            r_pc1[t, b] = abscorr(obs[ok], p_t[ok])
            r_spin[t, b] = abscorr_many(Ys[:, ok, :] @ coef,
                                        p_t[ok][:, None])[:, 0]

    # Patients as units
    pts = np.array([config.subject(s) for s in sids])
    upt = sorted(set(pts[np.isfinite(r_pc1).any(axis=1)]))
    f = np.array([np.nanmean(r_pc1[pts == p], axis=0) for p in upt])
    nu = np.array([np.nanmean(r_spin[pts == p], axis=0) for p in upt])
    draws = np.random.default_rng(SEED).integers(0, N_SPIN,
                                                 size=(N_GROUP, len(upt)))
    obs_g = np.nanmean(f, axis=0)
    p_group, null_mean, null_p95 = (np.full(n_b, np.nan) for _ in range(3))
    for b in range(n_b):
        nd = nu[np.arange(len(upt))[None, :], b, draws].mean(axis=1)
        null_mean[b], null_p95[b] = nd.mean(), np.percentile(nd, 95)
        p_group[b] = (np.sum(nd >= obs_g[b]) + 1) / (N_GROUP + 1)
    p_bonf = np.minimum(1.0, p_group * n_b)

    print(f"\nlead-field PC1 holds {np.nanmean(pc1_var) * 100:.0f}% of the "
          "variance across a session's hippocampal contacts")
    print(f"spherical-harmonic reconstruction of the observed maps: "
          f"|r| = {np.nanmean(fidelity):.3f} (min {np.nanmin(fidelity):.3f})")
    print(f"{'':22s}" + "".join(f"{b:>11s}" for b in bands))
    for lab, vals, fmt in (("|r| observed", obs_g, "11.2f"),
                           ("|r| spin mean", null_mean, "11.2f"),
                           ("|r| spin 95th pct", null_p95, "11.2f"),
                           ("p (one-sided)", p_group, "11.5f"),
                           ("p (Bonferroni)", p_bonf, "11.5f")):
        print(f"{lab:22s}" + "".join(f"{v:{fmt}}" for v in vals))
    print(f"n = {len(upt)} patients from {n_tr} trials")

    out = paths.out_dir("analysis", "hipp_cca") / "hipp_leadfield_spin.npz"
    np.savez(
        out, sessions=sids, hemispheres=hemis, bands=bands,
        patients=np.array(upt), pred_pc1=pred_pc1, pc1_var=pc1_var,
        r_pc1=r_pc1, fidelity=fidelity, r_group=obs_g, null_mean=null_mean,
        null_p95=null_p95, p_group=p_group, p_bonf=p_bonf,
    )
    print(f"Saved {out}")


spin_test(hipp_cca())
