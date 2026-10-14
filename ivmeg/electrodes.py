"""SEEG shafts, bipolar montage and bipolar-contact locations."""

from functools import lru_cache

import mne
import nibabel as nib
import numpy as np
import pandas as pd

from ivmeg import atlas, paths, release
from ivmeg.config import subject

# Maximum distance (mm) between adjacent contacts on the same shaft.
SHAFT_THRESHOLD_MM = 6.0

# A contact joins the current shaft if it is close to any of this many of the
# shaft's most recent contacts, which tolerates a single mislocalised contact.
SHAFT_LOOKBACK = 3


def _coords(df):
    return df[["mni_x", "mni_y", "mni_z"]].to_numpy(dtype=float)


def infer_shafts(df, threshold=SHAFT_THRESHOLD_MM):
    """Group contacts into shafts.

    Shafts are wired to consecutive amplifier inputs, so a shaft is a run of
    contacts in amplifier-input order where each contact lies within
    ``threshold`` mm of one of the previous ``SHAFT_LOOKBACK`` contacts.

    Returns a list of shafts, each a list of ``eeg_channel`` names.
    """
    df = df.assign(_idx=df["eeg_channel"].str[3:].astype(int))
    df = df.sort_values("_idx").reset_index(drop=True)
    xyz = _coords(df)
    names = df["eeg_channel"].tolist()

    shafts, current = [], []
    for i in range(len(df)):
        if np.isnan(xyz[i]).any():
            continue
        recent = current[-SHAFT_LOOKBACK:]
        if recent and np.linalg.norm(xyz[recent] - xyz[i], axis=1).min() \
                < threshold:
            current.append(i)
        else:
            if current:
                shafts.append([names[k] for k in current])
            current = [i]
    if current:
        shafts.append([names[k] for k in current])
    return shafts


def bipolar_pairs(df, threshold=SHAFT_THRESHOLD_MM):
    """(anode, cathode) ``eeg_channel`` pairs: adjacent contacts on a shaft
    that are less than ``threshold`` mm apart. The anode is the
    lower-numbered contact."""
    xyz = dict(zip(df["eeg_channel"], _coords(df)))
    pairs = []
    for shaft in infer_shafts(df, threshold):
        for a, c in zip(shaft[:-1], shaft[1:]):
            if np.linalg.norm(xyz[a] - xyz[c]) < threshold:
                pairs.append((a, c))
    return pairs


# FastSurfer (aseg) labels of the left and right hippocampus
HIPPOCAMPUS_LABELS = (17, 53)

# A location is hippocampal within this distance (mm) of the subject's
# hippocampus, allowing for the error in localising the contacts and
# registering them to the MRI.
HIPPOCAMPUS_TOL_MM = 3.0

@lru_cache(maxsize=None)
def _hippocampus_distance_map(sub):
    from scipy.ndimage import distance_transform_edt

    seg = nib.load(release.segmentation(sub))
    mask = np.isin(np.asarray(seg.dataobj), HIPPOCAMPUS_LABELS)
    dist = distance_transform_edt(~mask, sampling=seg.header.get_zooms()[:3])
    return dist, np.linalg.inv(seg.affine)


def hippocampus_distance(coords_mni, sub):
    """Distance (mm) from MNI coordinates to the subject's own hippocampus
    (FastSurfer segmentation in the release), after warping them into the
    subject's MRI with the RHINO (FNIRT) registration. 0 inside the
    hippocampus."""
    from osl_dynamics.meeg import rhino
    from osl_dynamics.meeg.session import SurfaceFilenames

    surfaces = SurfaceFilenames(str(paths.surfaces(sub)))
    mri = rhino._mni_to_mri(surfaces, np.atleast_2d(coords_mni))
    dist, inv = _hippocampus_distance_map(sub)
    ijk = np.round(nib.affines.apply_affine(inv, mri)).astype(int)
    ijk = np.clip(ijk, 0, np.array(dist.shape) - 1)
    return dist[tuple(ijk.T)]


def is_hippocampal(coords_mni, sub):
    """Whether each MNI coordinate (e.g. bipolar midpoints) is within
    ``HIPPOCAMPUS_TOL_MM`` of the subject's own hippocampus."""
    return hippocampus_distance(coords_mni, sub) <= HIPPOCAMPUS_TOL_MM


def bipolar_table(sub, bipolar_names):
    """One row per bipolar channel (``"<anode>-<cathode>"``, clinical names).

    The location is the midpoint of the two contacts, and the region is
    looked up at the midpoint (``region_aal116``, collapsed to
    ``region_aal24``).
    """
    df = release.electrodes(sub).set_index("clinical_name")
    rows = []
    for ch in bipolar_names:
        a, c = ch.split("-")
        pa, pc = _coords(df.loc[[a]])[0], _coords(df.loc[[c]])[0]
        mid = (pa + pc) / 2
        label = atlas.aal116_at_mni(*mid)
        rows.append({
            "bipolar_name": ch, "anode": a, "cathode": c,
            "region_aal116": label,
            "region_aal24": atlas.aal116_to_aal24(label),
            "mni_x": mid[0], "mni_y": mid[1], "mni_z": mid[2],
            "anode_mni_x": pa[0], "anode_mni_y": pa[1], "anode_mni_z": pa[2],
            "cathode_mni_x": pc[0], "cathode_mni_y": pc[1],
            "cathode_mni_z": pc[2],
        })
    return pd.DataFrame(rows)


def bipolar_midpoints(session_id):
    """:func:`bipolar_table` for the channels in the preprocessed SEEG."""
    info = mne.io.read_info(paths.seeg_bipolar(session_id), verbose=False)
    return bipolar_table(subject(session_id), info["ch_names"])
