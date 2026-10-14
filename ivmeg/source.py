"""LCMV virtual electrodes (VEs) with a fixed orientation.

MNI coordinates are moved into the subject's MRI space with the nonlinear
(FNIRT) registration computed by RHINO, so a VE sits where the contact was
localised in MNI space.
"""

import mne
import numpy as np
from mne.surface import _CheckInside
from osl_dynamics.meeg import rhino

# LCMV beamformer
REG = 0.01
RANK = {"meg": 64}
WEIGHT_NORM = "unit-noise-gain"

# Runs where the VE output is ~zero (SSS stretches in the MEG) are
# annotated as bad.
ZERO_THRESH = 5e-3
ZERO_MERGE_GAP_S = 2.0
ZERO_PAD_S = 0.5
ZERO_MIN_DUR_S = 0.25


def mni_to_mri(session, coords_mni):
    """MNI (mm) -> the subject's MRI space (m), as used by the forward
    model."""
    coords_mni = np.atleast_2d(np.asarray(coords_mni, dtype=float))
    return rhino._mni_to_scaledmri(session, coords_mni) / 1000


def axis_to_mri(session, start_mni, end_mni):
    """Unit vectors in MRI space along the line from ``start_mni`` to
    ``end_mni`` (both mapped through the registration)."""
    d = mni_to_mri(session, end_mni) - mni_to_mri(session, start_mni)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def inside_inner_skull(session, pos_mri):
    """Whether each point (MRI space, m) is inside the BEM surface."""
    rr, tris = mne.read_surface(f"{session.bem_dir}/inner_skull.surf")
    return _CheckInside(dict(rr=rr / 1000, tris=tris.astype(int)))(pos_mri)


def random_unit_vectors(n, rng):
    nn = rng.normal(size=(n, 3))
    return nn / np.linalg.norm(nn, axis=1, keepdims=True)


def forward(session, pos_mri, nn_mri):
    """Forward model for dipoles at ``pos_mri`` (m) oriented along
    ``nn_mri``.

    Returns the forward model (whose third leadfield component per dipole
    is along ``nn_mri``) and a mask of the dipoles inside the inner skull,
    which are the only ones it contains.
    """
    src = mne.setup_volume_source_space(
        pos=dict(rr=pos_mri, nn=nn_mri), verbose=False,
    )
    info, head_mri_t = rhino._read_head_model(session)
    fwd = rhino._make_fwd_solution(
        info, head_mri_t, src=src, bem=session.bem_solution_file,
        meg=True, eeg=False, ignore_ref=True, verbose=False,
    )
    inside = fwd["src"][0]["inuse"].astype(bool)
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=False, use_cps=True, verbose=False,
    )
    return fwd, inside


def band_pass(raw, band):
    lo, hi = band
    return raw.copy().filter(
        lo, hi, method="iir", iir_params={"order": 5, "ftype": "butter"},
        verbose=False,
    )


def _ad_hoc_noise_cov(raw, data_cov):
    """Diagonal noise covariance: the mean variance of each sensor type."""
    var = np.diag(data_cov.data)
    diag = np.zeros_like(var)
    for ch_type in ("mag", "grad"):
        names = [raw.ch_names[i] for i in
                 mne.pick_types(raw.info, meg=ch_type, exclude="bads")]
        idx = [data_cov.ch_names.index(n) for n in names
               if n in data_cov.ch_names]
        if idx:
            diag[idx] = var[idx].mean()
    return mne.Covariance(
        diag, data_cov.ch_names,
        [b for b in raw.info["bads"] if b in data_cov.ch_names],
        raw.info["projs"], nfree=2 ** 30,
    )


def _zero_annotations(data, raw):
    """``bad_segment_zero_meg`` annotations for runs of ~zero VE output."""
    sfreq = raw.info["sfreq"]
    is_zero = np.abs(data).max(axis=0) < ZERO_THRESH
    edges = np.diff(is_zero.astype(np.int8), prepend=0, append=0)
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= ZERO_MERGE_GAP_S * sfreq:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    pad = int(ZERO_PAD_S * sfreq)
    onsets, durations = [], []
    for s, e in merged:
        if e - s >= int(ZERO_MIN_DUR_S * sfreq):
            s, e = max(0, s - pad), min(data.shape[1], e + pad)
            onsets.append((s + raw.first_samp) / sfreq)
            durations.append((e - s) / sfreq)
    return mne.Annotations(onsets, durations,
                           ["bad_segment_zero_meg"] * len(onsets),
                           orig_time=raw.info["meas_date"])


def beamform(raw, fwd, names):
    """LCMV VEs for the dipoles in ``fwd``, one channel per name.

    The data covariance is computed from ``raw`` excluding bad segments.
    Returns a Raw object (channel type ``misc``) with the annotations of
    ``raw`` plus any ``bad_segment_zero_meg`` stretches.
    """
    data_cov = mne.compute_raw_covariance(
        raw, picks="meg", reject_by_annotation=True, verbose=False,
    )
    filters = mne.beamformer.make_lcmv(
        raw.info, fwd, data_cov, noise_cov=_ad_hoc_noise_cov(raw, data_cov),
        reg=REG, pick_ori="normal", weight_norm=WEIGHT_NORM, rank=RANK,
        verbose=False,
    )
    stc = mne.beamformer.apply_lcmv_raw(raw.copy().pick("meg"), filters,
                                        verbose=False)
    info = mne.create_info(list(names), raw.info["sfreq"], ch_types="misc")
    info.set_meas_date(raw.info["meas_date"])
    ve = mne.io.RawArray(stc.data, info, first_samp=raw.first_samp,
                         verbose=False)
    ve.set_annotations(raw.annotations + _zero_annotations(stc.data, raw))
    return ve
