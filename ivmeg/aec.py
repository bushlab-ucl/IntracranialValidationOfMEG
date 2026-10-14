"""Amplitude envelope correlation (AEC) connectomes."""

import numpy as np

from ivmeg.data import zscore

# Envelope correlations are computed in blocks of this length and averaged.
BLOCK_LEN_S = 30.0

# A region-pair edge enters the group connectome only if it is present in
# at least this many sessions.
EDGE_MIN_SESSIONS = 3


def block_corr(env, block_n):
    """Block-averaged Pearson correlation matrix of z-scored envelopes
    (channels, samples). Returns None if there is less than one block."""
    n_blocks = env.shape[1] // block_n
    if n_blocks == 0:
        return None
    r = np.zeros((env.shape[0], env.shape[0]))
    for b in range(n_blocks):
        x = env[:, b * block_n:(b + 1) * block_n]
        x = x - x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        denom = sd @ sd.T
        denom[denom == 0] = 1
        r += (x @ x.T) / (block_n * denom)
    return r / n_blocks


def _block_corr_1d(x, y, block_n):
    rs = []
    for b in range(x.shape[0] // block_n):
        xb = x[b * block_n:(b + 1) * block_n]
        yb = y[b * block_n:(b + 1) * block_n]
        xb, yb = xb - xb.mean(), yb - yb.mean()
        denom = xb.std() * yb.std()
        if denom != 0:
            rs.append(np.mean(xb * yb) / denom)
    return np.mean(rs) if rs else np.nan


def _zscore_1d(x):
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1)


def orthogonalised_aec(analytic, good, block_n):
    """AEC with pairwise orthogonalisation (Hipp et al., 2012).

    For each pair, the envelope of one signal is correlated with the
    envelope of the other after removing the component at zero phase lag
    (in the complex Hilbert domain), in both directions, and the two are
    averaged. ``analytic`` is (channels, samples) complex; ``good`` selects
    the samples used. Returns a symmetric matrix with a zero diagonal.
    """
    n = analytic.shape[0]
    amp = np.abs(analytic)
    env = zscore(amp[:, good])
    eps = 1e-30
    aec = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            j_orth = np.abs(np.imag(
                analytic[j] * np.conj(analytic[i]) / (amp[i] + eps)))[good]
            i_orth = np.abs(np.imag(
                analytic[i] * np.conj(analytic[j]) / (amp[j] + eps)))[good]
            r_ij = _block_corr_1d(env[i], _zscore_1d(j_orth), block_n)
            r_ji = _block_corr_1d(env[j], _zscore_1d(i_orth), block_n)
            aec[i, j] = aec[j, i] = 0.5 * (r_ij + r_ji)
    return aec


def regional(M, regions, region_list):
    """Contact-level matrix -> region matrix: the mean over pairs of
    contacts in two different regions. The diagonal (within-region pairs,
    which include adjacent bipolars on the same shaft) is NaN."""
    R = len(region_list)
    out = np.full((R, R), np.nan)
    idx = [np.flatnonzero(regions == r) for r in region_list]
    for a in range(R):
        for b in range(a + 1, R):
            if len(idx[a]) and len(idx[b]):
                out[a, b] = out[b, a] = np.nanmean(M[np.ix_(idx[a], idx[b])])
    return out
