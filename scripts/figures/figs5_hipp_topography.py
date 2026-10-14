"""Fig S5: MEG sensor CCA patterns of the hippocampal SEEG against the
modelled hippocampal field.

Rows: left and right hippocampus. First column: the modelled field (leading
singular vector of the hippocampal lead fields), then the observed CCA
pattern for each band, on the magnetometers. Sessions are averaged after
resolving each session's arbitrary sign (leading eigenvector of the
between-session correlation matrix), separately for the model and the
observations; each observed panel is then flipped to agree with the model.
|r| is the hemisphere's mean correlation with the model; * and ** mark a
Bonferroni-corrected spin-test p < 0.05 and p < 0.01 (one p per band,
pooling hemispheres). Crosses mark the projected mean hippocampal centroid.
"""

import matplotlib
import mne
import numpy as np

from ivmeg import config, paths
from ivmeg.plotting import BAND_LABELS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SENSOR_TYPE = "mag"
N_NEAR = 10  # sensors over the region that anchor the model's sign
HEAD_CENTRE_MNI = np.array([0.0, -20.0, 20.0])
HEMI_COLOR = {"L": "#1f77b4", "R": "#d62728"}
HEMI_NAME = {"L": "Left hippocampus", "R": "Right hippocampus"}


def session_flips(P):
    """Per-row signs that best align the rows of ``P`` (n, channels)."""
    if len(P) == 1:
        return np.ones(1)
    Z = np.nan_to_num(P - np.nanmean(P, axis=1, keepdims=True))
    s = np.sign(np.linalg.eigh(np.nan_to_num(np.corrcoef(Z)))[1][:, -1])
    s[s == 0] = 1.0
    return s


def agree_sign(obs, mod):
    """Sign of the |mod|-weighted correlation of an observed map with the
    model."""
    m = np.isfinite(obs) & np.isfinite(mod)
    if m.sum() < 10 or np.std(obs[m]) == 0 or np.std(mod[m]) == 0:
        return 1.0
    o, dd = obs[m] - obs[m].mean(), mod[m] - mod[m].mean()
    w = np.abs(mod[m])
    if np.sqrt((w * o * o).sum() * (w * dd * dd).sum()) == 0:
        return 1.0
    return float(np.sign((w * o * dd).sum())) or 1.0


def project_xy(mni, radius=0.095):
    """MNI coordinate -> topomap position (azimuthal equidistant)."""
    v = np.asarray(mni, dtype=float) - HEAD_CENTRE_MNI
    r = np.linalg.norm(v)
    if r == 0:
        return np.zeros(2)
    theta = np.arccos(np.clip(v[2] / r, -1.0, 1.0))
    phi = np.arctan2(v[1], v[0])
    return theta / (np.pi / 2) * radius * np.array([np.cos(phi), np.sin(phi)])


def sensor_positions(meg_ch_names):
    """Magnetometer positions in the standard Vectorview layout, and their
    indices in the 306-channel patterns."""
    info = mne.io.read_info(paths.meg_preproc(next(iter(config.SESSIONS))))
    info = mne.pick_info(info, mne.pick_types(info, meg=True, exclude=[]))
    assert info["ch_names"] == list(meg_ch_names), "channel order mismatch"
    idx = mne.pick_types(info, meg=SENSOR_TYPE, exclude=[])
    info_s = mne.pick_info(info, idx)
    layout = mne.channels.find_layout(info_s, ch_type=SENSOR_TYPE)
    pos = layout.pos[[layout.names.index(n) for n in info_s["ch_names"]], :2]
    pos = pos - pos.mean(axis=0)
    return pos / (np.abs(pos).max() / 0.095), idx


def draw(ax, vec, pos, mark):
    v = np.nanmax(np.abs(vec))
    if not np.isfinite(v) or v == 0:
        ax.set_axis_off()
        return False
    mne.viz.plot_topomap(np.nan_to_num(vec), pos, axes=ax, show=False,
                         cmap="RdBu_r", vlim=(-v, v), contours=4,
                         sensors=False)
    ax.plot(*mark, marker="x", ms=4, mew=1.2, color="k", zorder=5)
    return True


d = np.load(paths.out_dir("analysis", "hipp_cca") / "hipp_sensor.npz")
m = np.load(paths.out_dir("analysis", "hipp_cca") / "hipp_leadfield_spin.npz")
bands = d["bands"].astype(str)
hemis, sids = d["hemispheres"].astype(str), d["sessions"].astype(str)
n_contacts, depth = d["n_contacts"], d["mean_depth_mm"]
centroids, pat = d["centroids"], d["pat_meg"]
pc1, r_pc1, p_bonf = m["pred_pc1"], m["r_pc1"], m["p_bonf"]
n_b = len(bands)

for h in ("L", "R"):
    sel = hemis == h
    if sel.any():
        print(f"{HEMI_NAME[h]}: {sel.sum()} trials, "
              f"{len({config.subject(s) for s in sids[sel]})} patients, "
              f"{n_contacts[sel].sum()} contacts, "
              f"depth {depth[sel].mean():.1f} mm")

pos, sens = sensor_positions(d["meg_ch_names"].astype(str))
fig = plt.figure(figsize=(12.6, 5.6))
gs = fig.add_gridspec(2, n_b + 1, hspace=0.30, wspace=0.04, left=0.12,
                      right=0.99, top=0.78, bottom=0.06)
for k, hemi in enumerate(("L", "R")):
    sel = hemis == hemi
    if not sel.any():
        continue
    r_xy = project_xy(np.nanmean(centroids[sel], axis=0))
    near = np.argsort(np.linalg.norm(pos - r_xy, axis=1))[:N_NEAR]
    rr = np.linalg.norm(r_xy)
    mark = r_xy * min(1.0, 0.93 * 0.095 / rr) if rr else r_xy

    M = pc1[sel][np.isfinite(pc1[sel][:, sens]).all(axis=1)]
    mod = np.nanmean(M * session_flips(M[:, sens])[:, None], axis=0)[sens]
    if np.nanmean(mod[near]) < 0:
        mod = -mod
    ax0 = fig.add_subplot(gs[k, 0])
    draw(ax0, mod, pos, mark)
    ax0.text(-0.42, 0.5, HEMI_NAME[hemi].replace(" hipp", "\nhipp"),
             transform=ax0.transAxes, ha="center", va="center",
             fontsize=13, color=HEMI_COLOR[hemi])
    if k == 0:
        ax0.set_title("Lead field", fontsize=13, pad=8)

    for b, band in enumerate(bands):
        P = pat[sel, b, :]
        ok = np.isfinite(P).any(axis=1)
        if not ok.any():
            continue
        obs = np.nanmean(P[ok] * session_flips(P[ok])[:, None],
                         axis=0)[sens]
        obs = obs * agree_sign(obs, mod)
        ax = fig.add_subplot(gs[k, b + 1])
        if not draw(ax, obs, pos, mark):
            continue
        star = "**" if p_bonf[b] < 0.01 else "*" if p_bonf[b] < 0.05 else ""
        ax.text(0.5, -0.10, f"|r|={np.nanmean(r_pc1[sel, b]):.2f}{star}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=12)
        if k == 0:
            lo, hi = config.BANDS[band]
            ax.set_title(f"{BAND_LABELS[band]}\n{lo:g}-{hi:g} Hz",
                         fontsize=15, pad=8)

b0 = fig.add_subplot(gs[0, 1]).get_position()
b1 = fig.add_subplot(gs[0, n_b]).get_position()
fig.text((b0.x0 + b1.x1) / 2, 0.955, r"Observed CCA pattern, $\Sigma w$",
         ha="center", va="center", fontsize=14)
x = (fig.add_subplot(gs[0, 0]).get_position().x1 + b0.x0) / 2
rows = fig.add_subplot(gs[1, 0]).get_position()
fig.add_artist(plt.Line2D([x, x], [rows.y0 - 0.04, 0.92], color="0.7",
                          lw=1.0))
for a in fig.axes:  # the probe axes added above
    if not a.has_data() and not a.get_title():
        a.remove()

out = paths.figures() / "figs5_hipp_topography.png"
fig.savefig(out, dpi=240)
print(f"Saved {out}")
