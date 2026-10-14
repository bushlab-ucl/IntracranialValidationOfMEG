"""Power spectra, band power and the plotting shared by the power analyses."""

import io

import numpy as np
from scipy import stats
from scipy.signal import welch as _welch

from ivmeg import atlas, config
from ivmeg.plotting import BAND_LABELS

# Welch segments for band power (s)
NPERSEG_S = 4.0

EPS = 1e-30

SEEG_COLOR = "#c0392b"
MEG_COLOR = "#1f77b4"


def welch(x, sfreq, nperseg_s=NPERSEG_S):
    """Welch PSD along the last axis (Hann window, 50% overlap)."""
    nperseg = min(int(nperseg_s * sfreq), x.shape[-1])
    return _welch(x, fs=sfreq, nperseg=nperseg, axis=-1)


def band_power(freqs, psd, band):
    """Integral of the PSD over ``band`` (inclusive), along the last axis."""
    m = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapezoid(psd[..., m], freqs[m], axis=-1)


def log10_median(x):
    return float(np.log10(np.median(x) + EPS))


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (NaNs are ignored)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = stats.false_discovery_control(p[ok], method="bh")
    return out


def null_test(obs, null):
    """One-sided Monte-Carlo p-value of ``obs`` against ``null`` (larger is
    more extreme) and the z-score of ``obs`` relative to the null."""
    null = null[np.isfinite(null)]
    if not null.size or not np.isfinite(obs):
        return np.nan, np.nan
    p = (np.sum(null >= obs) + 1) / (null.size + 1)
    sd = null.std(ddof=1)
    return float(p), float((obs - null.mean()) / sd) if sd > 1e-12 else np.nan


def pearson(x, y, min_points=3):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_points:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[ok], y[ok])
    return float(r), float(p)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def _glass_brain(xyz, color):
    """Small glass brain (coronal + axial) with a marker at ``xyz``."""
    import matplotlib.pyplot as plt
    from nilearn import plotting

    fig = plt.figure(figsize=(3.6, 1.8))
    display = plotting.plot_glass_brain(
        None, display_mode="yz", figure=fig, colorbar=False, annotate=False,
        alpha=0.75,
    )
    display.add_markers([xyz], marker_color=color, marker_size=400, alpha=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def plot_region_psds(freqs, traces, n_contacts, centroids, legend, filename,
                     frange=(2, 45), ylim=(-5.0, 0.0)):
    """Grid of per-region PSDs, mean +/- SEM over sessions.

    traces : dict region -> list of (color, (n_sessions, n_freqs) log10 PSD)
    n_contacts : dict region -> total contacts
    centroids : dict region -> MNI coordinate for the glass-brain inset
    legend : list of (color, label)
    """
    import matplotlib.pyplot as plt

    regions = [r for r in atlas.AAL24_NAMES if r in traces]
    n = len(regions)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    last = n - (nrows - 1) * ncols

    # Two grid cells per panel so a partial last row can be centred
    mosaic = []
    for row in range(nrows):
        n_row = ncols if row < nrows - 1 else last
        cells = ["."] * (ncols - n_row)
        for c in range(n_row):
            cells += [f"p{row * ncols + c}"] * 2
        mosaic.append(cells + ["."] * (2 * ncols - len(cells)))
    fig, axes = plt.subplot_mosaic(
        mosaic, figsize=(3.9 * ncols, 3.0 * nrows), sharex=True, sharey=True,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.15, h_pad=0.20, wspace=0.12,
                                hspace=0.10)

    fmask = (freqs >= frange[0]) & (freqs <= frange[1])
    f = freqs[fmask]
    colors = atlas.region_colors()
    for i, r in enumerate(regions):
        ax = axes[f"p{i}"]
        n_s = 0
        for color, log_psd in traces[r]:
            log_psd = np.asarray(log_psd)[:, fmask]
            n_s = log_psd.shape[0]
            mean = log_psd.mean(axis=0)
            sem = (log_psd.std(axis=0, ddof=1) / np.sqrt(n_s) if n_s > 1
                   else np.zeros_like(mean))
            ax.fill_between(f, mean - sem, mean + sem, color=color,
                            alpha=0.18, linewidth=0)
            ax.plot(f, mean, color=color, linewidth=1.6)
        ax.set_title(f"{r}  (N={n_s}, {n_contacts[r]} contacts)", fontsize=12)
        ax.set_ylim(*ylim)
        ax.set_xlim(*frange)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Frequency (Hz)", fontsize=13)
        if i % ncols == 0:
            ax.set_ylabel(
                r"$\log_{10}\left[\mathrm{PSD}\,/\,(1/\mathrm{Hz})\right]$",
                fontsize=13,
            )
        inset = ax.inset_axes([0.58, 0.70, 0.40, 0.26])
        inset.imshow(_glass_brain(centroids[r], colors[r]))
        inset.set_axis_off()

    handles = [plt.Line2D([0], [0], color=c, linewidth=3.5, label=l)
               for c, l in legend]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=len(handles), frameon=False,
               fontsize=15, handlelength=2.4, handletextpad=0.7,
               columnspacing=3.0)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


def plot_null_hists(obs_r, null_r, p_fdr, null_label, filename,
                    text_y=0.77):
    """Per-band histogram of a null distribution of correlations with the
    observed correlation."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.1),
                             constrained_layout=True)
    for bi, (band, limits) in enumerate(config.BANDS.items()):
        ax = axes.ravel()[bi]
        nr = null_r[bi][np.isfinite(null_r[bi])]
        if nr.size:
            ax.hist(nr, bins=22, density=True, color="#C9CED6",
                    edgecolor="#6B7280", linewidth=0.7)
            ax.axvline(nr.mean(), color="#4B5563", linewidth=1.2,
                       linestyle="--")
        if np.isfinite(obs_r[bi]):
            ax.axvline(obs_r[bi], color="#B91C1C", linewidth=2.4)
        ax.text(0.98, text_y,
                rf"$r_{{obs}}={obs_r[bi]:+.2f}$" "\n"
                rf"$p_{{FDR}}={p_fdr[bi]:.1g}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                          pad=1.5))
        ax.set_title(f"{BAND_LABELS[band]} ({limits[0]:g}–{limits[1]:g} Hz)",
                     fontsize=13)
        if bi >= 3:
            ax.set_xlabel("Correlation", fontsize=11.5)
        if bi % 3 == 0:
            ax.set_ylabel("Density", fontsize=11.5)
        ax.tick_params(axis="both", labelsize=10.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.legend(handles=[
        Patch(facecolor="#C9CED6", edgecolor="#6B7280", linewidth=0.7,
              label=null_label),
        Line2D([0], [0], color="#4B5563", linewidth=1.2, linestyle="--",
               label="Null mean"),
        Line2D([0], [0], color="#B91C1C", linewidth=2.4,
               label="Observed correlation"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3,
        frameon=False, fontsize=10.5, handletextpad=0.8, columnspacing=1.8)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")
