"""Canonical correlation analysis (CCA) between SEEG and MEG.

The leading canonical correlation between a set of SEEG contacts ``X``
(samples, p) and MEG channels ``Y`` (samples, q) is tested with a
block-shuffle permutation null: the MEG is cut into 1 s blocks whose order is
permuted, which breaks the alignment between the modalities but keeps the
temporal structure within each block. The bias-corrected effect is
``r_eff = r_obs - mean(null)``.
"""

import numpy as np

N_PERM = 100
BLOCK_LEN_S = 1.0
SEED = 0

# Region-level CCA: minimum contacts of a region in one session.
MIN_CONTACTS = 3

# Sensor-level CCA: keep column-space directions of the sensor data whose
# eigenvalue exceeds RANK_TOL * the largest (removes the SSS null space).
RANK_TOL = 1e-6

SUPER_GROUP = {
    "Sensorimotor L": "cortical", "Sensorimotor R": "cortical",
    "Prefrontal L": "cortical", "Prefrontal R": "cortical",
    "Orbitofrontal L": "cortical", "Orbitofrontal R": "cortical",
    "Parietal L": "cortical", "Parietal R": "cortical",
    "Temporal L": "cortical", "Temporal R": "cortical",
    "Occipital L": "cortical", "Occipital R": "cortical",
    "Insula L": "cortical", "Insula R": "cortical",
    "MTL L": "limbic", "MTL R": "limbic",
    "AntCingulate": "limbic", "PostCingulate": "limbic",
    "BasalGanglia L": "subcortical", "BasalGanglia R": "subcortical",
    "Thalamus": "subcortical",
    "Cerebellum L": "cerebellum", "Cerebellum R": "cerebellum",
    "Vermis": "cerebellum",
}


# ----------------------------------------------------------------------------
# CCA with an equal number of MEG and SEEG channels (virtual electrodes)
# ----------------------------------------------------------------------------
def cca(X, Y):
    """Leading canonical correlation of X (n, p) and Y (n, q)."""
    Q_x, _ = np.linalg.qr(X - X.mean(axis=0))
    return leading_r(Q_x, Y)


def leading_r(Q_x, Y):
    """Leading canonical correlation given an orthonormal basis of X."""
    Q_y, _ = np.linalg.qr(Y - Y.mean(axis=0))
    return float(np.linalg.svd(Q_x.T @ Q_y, compute_uv=False)[0])


def block_shuffle(Y, block_n, rng):
    n_blocks = Y.shape[0] // block_n
    blocks = Y[:n_blocks * block_n].reshape(n_blocks, block_n, -1)
    return blocks[rng.permutation(n_blocks)].reshape(n_blocks * block_n, -1)


def cca_null(X, Y, block_n, rng, n_perm=N_PERM):
    """Observed leading canonical correlation and its block-shuffle null
    (MEG side ``Y`` shuffled)."""
    r_obs = cca(X, Y)
    Q_x, _ = np.linalg.qr(X - X.mean(axis=0))
    null = np.array([leading_r(Q_x, block_shuffle(Y, block_n, rng))
                     for _ in range(n_perm)])
    return r_obs, null


# ----------------------------------------------------------------------------
# CCA against the MEG sensors
# ----------------------------------------------------------------------------
def ortho_basis(Yc, rank_tol=RANK_TOL):
    """Orthonormal basis (n, k) of the column space of centred ``Yc``
    (n, q), with the eigenvectors (q, k) and eigenvalues (k,) of
    ``Yc.T @ Yc`` that map basis weights back onto the sensors."""
    w, V = np.linalg.eigh(Yc.T @ Yc)
    w, V = w[::-1], V[:, ::-1]
    if w[0] <= 0:
        return np.zeros((len(Yc), 0)), np.zeros((Yc.shape[1], 0)), np.zeros(0)
    k = int(np.sum(w > w[0] * rank_tol))
    return (Yc @ V[:, :k]) / np.sqrt(w[:k]), V[:, :k], w[:k]


def sensor_weights(b, V_k, w_k):
    """Basis weights -> sensor weights (the CCA filter)."""
    return V_k @ (b / np.sqrt(w_k))


def sensor_pattern(b, V_k, w_k, n):
    """Basis weights -> sensor pattern: the covariance of each sensor with
    the canonical variate (Haufe et al. 2014)."""
    return (V_k * np.sqrt(w_k)) @ b / np.sqrt(n - 1)


def seeg_weights(a, R):
    return np.linalg.solve(R, a)


def seeg_pattern(a, R, n):
    return R.T @ a / np.sqrt(n - 1)


def fix_sign(pattern_meg, *arrays):
    """Flip a canonical pair so the largest MEG pattern element is
    positive."""
    if pattern_meg.size == 0 or not np.any(np.isfinite(pattern_meg)):
        return (pattern_meg,) + arrays
    sign = np.sign(pattern_meg[np.nanargmax(np.abs(pattern_meg))]) or 1.0
    return tuple(sign * x for x in (pattern_meg,) + arrays)


def sensor_cca_null(X, U_y, block_n, rng, n_perm=N_PERM):
    """Leading CCA between SEEG ``X`` (n, p) and the orthonormal sensor
    basis ``U_y`` (n, k), with a block-shuffle null on the rows of ``U_y``.

    Returns ``(r_obs, null, extras)``; ``extras`` holds the leading singular
    vectors ``a`` (on the SEEG basis) and ``b`` (on ``U_y``), the SEEG ``R``
    factor and the number of samples ``nt``. ``r_obs`` is NaN (and
    ``extras`` None) if there are too few blocks.
    """
    n_blocks = X.shape[0] // block_n
    if n_blocks < 4 or U_y.shape[1] == 0:
        return np.nan, np.full(n_perm, np.nan), None
    nt = n_blocks * block_n
    Q_x, R_x = np.linalg.qr(X[:nt] - X[:nt].mean(axis=0))
    U_y = U_y[:nt]
    A, S, Bt = np.linalg.svd(Q_x.T @ U_y, full_matrices=False)
    blocks = U_y.reshape(n_blocks, block_n, -1)
    null = np.empty(n_perm)
    for i in range(n_perm):
        U_p = blocks[rng.permutation(n_blocks)].reshape(nt, -1)
        null[i] = np.linalg.svd(Q_x.T @ U_p, compute_uv=False)[0]
    return float(S[0]), null, {"a": A[:, 0], "b": Bt[0], "R": R_x, "nt": nt}


def summarise(r_obs, null):
    """Per-test summary fields."""
    return {
        "r_obs": r_obs,
        "null_mean": null.mean(),
        "null_lo": np.percentile(null, 5),
        "null_hi": np.percentile(null, 95),
        "p_perm": (np.sum(null >= r_obs) + 1) / (len(null) + 1),
    }


def region_groups(contacts, min_contacts=MIN_CONTACTS):
    """{AAL24 region: [row indices into ``contacts``]} for the regions
    with at least ``min_contacts`` contacts, in order of appearance."""
    groups = {}
    for i, region in enumerate(contacts["region_aal24"]):
        if region != "Undefined":
            groups.setdefault(region, []).append(i)
    return {r: idx for r, idx in groups.items() if len(idx) >= min_contacts}


def save_trials(path, rows, bands, **extra):
    """Save per-(session, region) CCA results as long-form arrays."""
    def stack(key):
        return np.array([[r.get(f"{key}_{b}", np.nan) for b in bands]
                         for r in rows])

    r_obs, null_mean = stack("r_obs"), stack("null_mean")
    np.savez(
        path,
        sessions=np.array([r["session"] for r in rows]),
        regions=np.array([r["region"] for r in rows]),
        super_groups=np.array([SUPER_GROUP.get(r["region"], "other")
                               for r in rows]),
        n_contacts=np.array([r["n_contacts"] for r in rows], dtype=int),
        mean_depth_mm=np.array([r["mean_depth_mm"] for r in rows]),
        r_obs=r_obs, null_mean=null_mean, null_lo=stack("null_lo"),
        null_hi=stack("null_hi"), p_perm=stack("p_perm"),
        r_eff=r_obs - null_mean, bands=np.array(list(bands)), **extra,
    )


# ----------------------------------------------------------------------------
# Plot: r_eff against region depth
# ----------------------------------------------------------------------------
def plot_depth(npz_file, out_file):
    """One panel per band: r_eff of each (session, region) against the
    region's mean depth, filled if p_perm < 0.05, with the Spearman
    correlation. Regions are kept if they have trials in at least
    MIN_SESSIONS sessions and MIN_CONTACTS contacts in total."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    from ivmeg import atlas, config
    from ivmeg.plotting import BAND_LABELS

    d = np.load(npz_file)
    bands = d["bands"].astype(str)
    regions, sids = d["regions"].astype(str), d["sessions"].astype(str)
    n_contacts, depth = d["n_contacts"], d["mean_depth_mm"]
    r_eff, p_perm = d["r_eff"], d["p_perm"]

    kept = sorted(r for r in set(regions)
                  if len(set(sids[regions == r])) >= config.MIN_SESSIONS
                  and n_contacts[regions == r].sum() >= config.MIN_CONTACTS)
    print(f"{len(kept)} regions kept; dropped "
          f"{sorted(set(regions) - set(kept))}")
    keep = np.isin(regions, kept)
    regions, depth = regions[keep], depth[keep]
    r_eff, p_perm = r_eff[keep], p_perm[keep]
    colors = atlas.region_colors()

    n_band = len(bands)
    fig, axes = plt.subplots(1, n_band, figsize=(3.9 * n_band, 4.5),
                             constrained_layout=True, sharey=True)
    y = r_eff[~np.isnan(r_eff)]
    y_pad = 0.08 * (y.max() - y.min())
    x = depth[np.any(~np.isnan(r_eff), axis=1)]
    x_pad = 0.05 * (x.max() - x.min())

    print("Spearman correlation of r_eff with depth:")
    for bi, band in enumerate(bands):
        ax = axes[bi]
        valid = ~np.isnan(r_eff[:, bi]) & ~np.isnan(p_perm[:, bi])
        xb, yb = depth[valid], r_eff[valid, bi]
        c = np.array([colors[r] for r in regions[valid]])
        sig = p_perm[valid, bi] < 0.05
        ax.scatter(xb[sig], yb[sig], c=c[sig], s=46, alpha=0.9,
                   edgecolor="k", linewidth=0.5)
        ax.scatter(xb[~sig], yb[~sig], facecolors="none",
                   edgecolors=c[~sig], s=46, alpha=0.9, linewidth=1.2)
        if len(xb) >= 2:
            xs = np.linspace(xb.min(), xb.max(), 100)
            ax.plot(xs, np.polyval(np.polyfit(xb, yb, 1), xs), color="k",
                    linewidth=1.0, linestyle="--", alpha=0.7)
        if len(xb) >= 3:
            rho, p = stats.spearmanr(xb, yb)
            print(f"  {band:>10s}: rho = {rho:+.3f}, p = {p:.3g} "
                  f"(n = {len(xb)})")
            ax.text(0.04, 0.96, f"corr = {rho:+.2f}\np = {p:.1g}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=19, bbox=dict(facecolor="white", alpha=0.7,
                                           edgecolor="none", pad=2.0))
        ax.axhline(0, color="0.7", linewidth=0.6, linestyle=":")
        ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
        ax.set_ylim(y.min() - y_pad, y.max() + y_pad)
        ax.tick_params(axis="both", labelsize=18)
        ax.set_xlabel("Depth (mm)", fontsize=20)
        if bi == 0:
            ax.set_ylabel(r"CCA $r_{\rm eff}$", fontsize=20)
        lo, hi = config.BANDS[band]
        ax.set_title(f"{BAND_LABELS[band]}  ({lo:g}–{hi:g} Hz)", fontsize=22)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=colors[r], markeredgecolor="k",
                          markersize=10, label=r) for r in kept]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.28),
               ncol=int(np.ceil(len(kept) / 2)), frameon=False, fontsize=19,
               handletextpad=0.6, columnspacing=2.6, labelspacing=0.6)
    sig_h = plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor="0.4", markeredgecolor="k",
                       markersize=11, label=r"filled: $p_{\rm perm} < 0.05$")
    ns_h = plt.Line2D([0], [0], marker="o", linestyle="",
                      markerfacecolor="none", markeredgecolor="0.4",
                      markeredgewidth=1.4, markersize=11,
                      label=r"hollow: $p_{\rm perm} \geq 0.05$")
    fig.legend(handles=[sig_h, ns_h], loc="lower center",
               bbox_to_anchor=(0.5, -0.40), ncol=2, frameon=False,
               fontsize=19, handletextpad=0.6, columnspacing=3.5)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")
