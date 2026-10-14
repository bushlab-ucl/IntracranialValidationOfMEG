"""Oscillatory bursts and the SEEG amplitude modulation during MEG bursts.

Bursts are detected on a band-limited amplitude envelope: samples above the
80th percentile of the envelope, runs separated by at most one cycle (at the
band centre) merged, and runs shorter than three cycles discarded.

The SEEG amplitude modulation index during a burst train is

    mod = (A_burst - A_non-burst) / (A_burst + A_non-burst)

where A is the mean SEEG envelope inside / outside the bursts.
"""

import numpy as np
from scipy import stats
from scipy.signal import hilbert

from ivmeg import config
from ivmeg.data import bandpass

PERCENTILE = 80

# Contacts whose mean SEEG band envelope is below this fraction of the
# session median are excluded.
ENV_MIN_FRAC = 0.5

# A (session, region) point needs at least this many contacts.
MIN_CONTACTS_PER_SESSION = 3

# Block-shuffle permutation null.
BLOCK_LEN_S = 1.0
N_PERM = 500
SEED = 0


def _centre(band):
    lo, hi = config.BANDS[band]
    return (lo + hi) / 2


def envelope(x, sfreq, band):
    """Hilbert amplitude envelope of ``x`` band-passed to ``band``."""
    return np.abs(hilbert(bandpass(x, sfreq, config.BANDS[band]), axis=-1))


def burst_train(env, sfreq, band):
    """Boolean train, True inside the bursts of a 1D envelope."""
    above = env > np.percentile(env, PERCENTILE)
    edges = np.diff(above.astype(np.int8), prepend=0, append=0)
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    train = np.zeros(env.size, dtype=bool)
    if not runs:
        return train
    merge_gap = int(round(sfreq / _centre(band)))
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    min_len = int(round(3 * sfreq / _centre(band)))
    for s, e in merged:
        if e - s >= min_len:
            train[s:e] = True
    return train


def strong_contacts(env):
    """Mask of contacts (rows of ``env``) with enough SEEG band amplitude."""
    means = env.mean(axis=1)
    return means >= ENV_MIN_FRAC * np.median(means)


def readout(env, train):
    """(A_burst, A_non-burst) of a 1D envelope; NaNs if a side is empty."""
    n_in = int(train.sum())
    if n_in == 0 or n_in == train.size:
        return np.nan, np.nan
    return float(env[train].mean()), float(env[~train].mean())


def mod_index(a_in, a_out):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a_in - a_out) / (a_in + a_out)


def mod_index_matrix(env, trains):
    """Modulation index of each envelope (C, T) for each train (T, K) as a
    (C, K) array."""
    trains = trains.astype(float)
    n_in = trains.sum(axis=0)
    n = env.shape[1]
    sums = env @ trains
    total = env.sum(axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return mod_index(sums / n_in, (total - sums) / (n - n_in))


def block_shuffle_null(env, train, sfreq):
    """Block-shuffle permutation test of the log-ratio log(A_burst /
    A_non-burst).

    The SEEG envelope is cut into 1 s blocks whose order is permuted while
    the MEG burst train is held fixed. Returns the z-score of the observed
    log-ratio against the null and the one-sided p-value (NaNs if
    degenerate).
    """
    block_n = max(1, int(round(BLOCK_LEN_S * sfreq)))
    n_blocks = train.size // block_n
    if n_blocks < 4:
        return np.nan, np.nan
    nt = n_blocks * block_n
    env, train = env[:nt], train[:nt]
    n_in = int(train.sum())
    n_out = nt - n_in
    if n_in == 0 or n_out == 0:
        return np.nan, np.nan
    m_in, m_out = env[train].mean(), env[~train].mean()
    if not (m_in > 0 and m_out > 0):
        return np.nan, np.nan
    obs = np.log(m_in / m_out)

    # In-burst sum for a block permutation: sum_t M[perm[t], t]
    M = env.reshape(n_blocks, block_n) @ \
        train.reshape(n_blocks, block_n).astype(float).T
    rng = np.random.default_rng(SEED)
    perms = np.array([rng.permutation(n_blocks) for _ in range(N_PERM)])
    in_sums = M[perms, np.arange(n_blocks)[None, :]].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        null = np.log((in_sums / n_in) / ((env.sum() - in_sums) / n_out))
    null = null[np.isfinite(null)]
    if null.size < 2:
        return np.nan, np.nan
    sd = null.std(ddof=1)
    z = (obs - null.mean()) / sd if sd > 1e-12 else np.nan
    p = (np.sum(null >= obs) + 1) / (null.size + 1)
    return float(z), float(p)


def stouffer_significant(z, alpha=0.05):
    """Whether Stouffer's combined Z of finite ``z`` is significantly
    positive (one-sided)."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return False
    return bool(z.sum() / np.sqrt(z.size) > stats.norm.ppf(1 - alpha))


def kept_regions(df):
    """Regions with contacts in >= MIN_SESSIONS sessions and >=
    MIN_CONTACTS (session, contact) pairs in a contact-level table."""
    g = df.drop_duplicates(["session", "contact"]).groupby("region")
    ok = (g["session"].nunique() >= config.MIN_SESSIONS) & \
         (g.size() >= config.MIN_CONTACTS)
    return set(ok.index[ok])


# ----------------------------------------------------------------------------
# Plotting shared by the burst figures
# ----------------------------------------------------------------------------
def box_strip(ax, df, column, label_fmt, rng):
    """Box + jittered points per band, coloured by region, filled if sig.
    Returns the regions shown."""
    from ivmeg import atlas
    from ivmeg.plotting import BAND_LABELS

    colors = atlas.region_colors()
    for pos, band in enumerate(config.BANDS, start=1):
        b = df[df["band"] == band]
        y = b[column].to_numpy()
        sig = b["sig"].to_numpy(dtype=bool)
        c = np.array([colors[r] for r in b["region"]]).reshape(-1, 3)
        ax.boxplot([y], positions=[pos], widths=0.6, showfliers=False,
                   medianprops=dict(color="k", linewidth=1.6),
                   boxprops=dict(color="0.4"), whiskerprops=dict(color="0.4"),
                   capprops=dict(color="0.4"), zorder=1)
        x = pos + rng.uniform(-0.22, 0.22, size=y.size)
        ax.scatter(x[sig], y[sig], c=c[sig], s=34, alpha=0.9, edgecolor="k",
                   linewidth=0.4, zorder=3)
        ax.scatter(x[~sig], y[~sig], facecolors="none", edgecolors=c[~sig],
                   s=34, alpha=0.8, linewidth=1.0, zorder=2)
        print(label_fmt.format(band=band, n=len(y), med=np.median(y),
                               pos=100 * np.mean(y > 0),
                               sig=100 * np.mean(sig)))
    ax.axhline(0, color="0.6", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xticks(range(1, len(config.BANDS) + 1))
    ax.set_xticklabels([f"{BAND_LABELS[b]}\n{lo:g}–{hi:g} Hz"
                        for b, (lo, hi) in config.BANDS.items()], fontsize=14)
    ax.set_xlim(0.4, len(config.BANDS) + 0.6)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return sorted(df["region"].unique())


def legends(fig, regions):
    """Region colour legend and filled/hollow significance legend."""
    import matplotlib.pyplot as plt

    from ivmeg import atlas

    colors = atlas.region_colors()
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=colors[r], markeredgecolor="k",
                          markersize=9, label=r) for r in regions]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.20),
               ncol=int(np.ceil(len(regions) / 2)), frameon=False, fontsize=13,
               handletextpad=0.6, columnspacing=2.4, labelspacing=0.6)
    fill = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.4",
                   markeredgecolor="k", markersize=10,
                   label=r"filled: $p_{\rm perm} < 0.05$"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor="0.4", markeredgewidth=1.3, markersize=10,
                   label=r"hollow: $p_{\rm perm} \geq 0.05$"),
    ]
    fig.legend(handles=fill, loc="lower center", bbox_to_anchor=(0.5, -0.28),
               ncol=2, frameon=False, fontsize=13, handletextpad=0.6,
               columnspacing=3.0)
