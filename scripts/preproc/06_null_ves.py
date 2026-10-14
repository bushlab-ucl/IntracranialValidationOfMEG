"""Null virtual electrodes (VEs) for the spatial-specificity tests.

- random: K=100 locations sampled uniformly in MNI space inside the
  subject's inner skull.
- matched_depth: K=100 locations at the same depth below the scalp as the
  session's hippocampal contacts (within 5 mm of their depth range), at
  least 20 mm from any hippocampal contact and more than 3 mm from the
  subject's own hippocampus. Only for sessions with hippocampal contacts
  (bipolar midpoint within 3 mm of the subject's hippocampus, FastSurfer
  segmentation in the release).

Each null VE has a random orientation, and a beamformer is computed for each
frequency band, as for the per-band VEs at the contacts.

Output: ``<out_dir>/preproc/null_<kind>/<session>_<band>-raw.fif`` (channels
``null_0001``...) and ``<session>_sources.csv``.

Usage: python 06_null_ves.py [--session SESSION ...] [--kind KIND ...]
                              [--n-workers N]
"""

import argparse

import mne
import numpy as np
import pandas as pd

from ivmeg import atlas, config, electrodes, paths, source
from osl_dynamics.meeg import parallel

K = 100
KINDS = ["random", "matched_depth"]

# Matched-depth null
DEPTH_TOL_MM = 5.0
MIN_DIST_MM = 20.0

# Candidate locations are drawn from this MNI bounding box (mm) in batches.
MNI_BOX = np.array([[-90, -126, -72], [90, 90, 108]])
BATCH = 2000
MAX_BATCHES = 200


def sample(session, rng, keep=None):
    """K MNI locations inside the inner skull for which ``keep(mni)`` is
    True. Returns the MNI coordinates (mm) and MRI positions (m)."""
    mni, mri = [], []
    for _ in range(MAX_BATCHES):
        cand = rng.uniform(*MNI_BOX, size=(BATCH, 3))
        if keep is not None:
            cand = cand[keep(cand)]
        if len(cand) == 0:
            continue
        pos = source.mni_to_mri(session, cand)
        ok = source.inside_inner_skull(session, pos)
        mni.append(cand[ok])
        mri.append(pos[ok])
        if sum(len(m) for m in mni) >= K:
            return np.concatenate(mni)[:K], np.concatenate(mri)[:K]
    raise RuntimeError(f"Found only {sum(len(m) for m in mni)} locations")


def reconstruct(sid, kind, mni, pos, nn, raw):
    session = paths.osl_session(sid)
    fwd, inside = source.forward(session, pos, nn)
    names = [f"null_{i + 1:04d}" for i in range(len(mni))]
    labels = [atlas.aal116_at_mni(*m) for m in mni]
    pd.DataFrame({
        "channel": names, "inside": inside,
        "mni_x": mni[:, 0], "mni_y": mni[:, 1], "mni_z": mni[:, 2],
        "depth_mm": atlas.depth(mni),
        "region_aal116": labels,
        "region_aal24": [atlas.aal116_to_aal24(l) for l in labels],
        "nn_mri_x": nn[:, 0], "nn_mri_y": nn[:, 1], "nn_mri_z": nn[:, 2],
    }).to_csv(paths.null_sources(sid, kind), index=False)

    names = [n for n, k in zip(names, inside) if k]
    for band, limits in config.BANDS.items():
        ve = source.beamform(source.band_pass(raw, limits), fwd, names)
        ve.save(paths.null_ve(sid, kind, band), overwrite=True)


def process(item):
    sid, kinds = item["id"], item["kinds"]
    session = paths.osl_session(sid)
    rng = np.random.default_rng(config.SESSIONS[sid])
    raw = mne.io.read_raw_fif(paths.meg_preproc(sid), preload=True)

    # The random locations are always drawn, so the matched-depth locations
    # are the same whether or not the random VEs are recomputed
    mni, pos = sample(session, rng)
    nn = source.random_unit_vectors(K, rng)
    if "random" in kinds:
        reconstruct(sid, "random", mni, pos, nn, raw)
    if "matched_depth" not in kinds:
        return

    bp = electrodes.bipolar_midpoints(sid)
    sub = config.subject(sid)
    hipp = bp[electrodes.is_hippocampal(
        bp[["mni_x", "mni_y", "mni_z"]].to_numpy(), sub)]
    if hipp.empty:
        print("No hippocampal contacts: no matched-depth null")
        return
    hipp_mni = hipp[["mni_x", "mni_y", "mni_z"]].to_numpy()
    hipp_depth = atlas.depth(hipp_mni)
    lo, hi = hipp_depth.min() - DEPTH_TOL_MM, hipp_depth.max() + DEPTH_TOL_MM
    print(f"{len(hipp)} hippocampal contacts, depth {lo:.0f}-{hi:.0f} mm "
          "(with tolerance)")

    def matched(cand):
        d = atlas.depth(cand)
        dist = np.linalg.norm(cand[:, None] - hipp_mni[None], axis=2).min(1)
        ok = (d >= lo) & (d <= hi) & (dist >= MIN_DIST_MM)
        ok[ok] = ~electrodes.is_hippocampal(cand[ok], sub)
        return ok

    mni, pos = sample(session, rng, keep=matched)
    reconstruct(sid, "matched_depth", mni, pos,
                source.random_unit_vectors(K, rng), raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--session", nargs="+", choices=list(config.SESSIONS))
    parser.add_argument("--kind", nargs="+", choices=KINDS, default=KINDS)
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    sessions = args.session if args.session else list(config.SESSIONS)
    parallel.run(process,
                 items=[{"id": s, "kinds": args.kind} for s in sessions],
                 n_workers=args.n_workers, log_dir=paths.logs("06_null_ves"))
