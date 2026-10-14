"""AAL atlas lookups.

Regions are assigned by looking up an MNI coordinate in the 116-region AAL
atlas (8 mm, as distributed with osl-dynamics) and collapsing the label into
a coarse 24-region scheme (AAL24):
https://osl-dynamics.readthedocs.io/en/latest/parcellations/aal24.html
"""

from functools import lru_cache

import nibabel as nib
import numpy as np

AAL116_FILE = "atlas-AAL_nparc-116_space-MNI_res-8x8x8.nii.gz"

# (name, hemisphere) in the order of the atlas' 4th axis.
AAL116_LABELS = [
    ("Precentral", "L"), ("Precentral", "R"),
    ("Frontal_Sup", "L"), ("Frontal_Sup", "R"),
    ("Frontal_Sup_Orb", "L"), ("Frontal_Sup_Orb", "R"),
    ("Frontal_Mid", "L"), ("Frontal_Mid", "R"),
    ("Frontal_Mid_Orb", "L"), ("Frontal_Mid_Orb", "R"),
    ("Frontal_Inf_Oper", "L"), ("Frontal_Inf_Oper", "R"),
    ("Frontal_Inf_Tri", "L"), ("Frontal_Inf_Tri", "R"),
    ("Frontal_Inf_Orb", "L"), ("Frontal_Inf_Orb", "R"),
    ("Rolandic_Oper", "L"), ("Rolandic_Oper", "R"),
    ("Supp_Motor_Area", "L"), ("Supp_Motor_Area", "R"),
    ("Olfactory", "L"), ("Olfactory", "R"),
    ("Frontal_Sup_Medial", "L"), ("Frontal_Sup_Medial", "R"),
    ("Frontal_Med_Orb", "L"), ("Frontal_Med_Orb", "R"),
    ("Rectus", "L"), ("Rectus", "R"),
    ("Insula", "L"), ("Insula", "R"),
    ("Cingulum_Ant", "L"), ("Cingulum_Ant", "R"),
    ("Cingulum_Mid", "L"), ("Cingulum_Mid", "R"),
    ("Cingulum_Post", "L"), ("Cingulum_Post", "R"),
    ("Hippocampus", "L"), ("Hippocampus", "R"),
    ("ParaHippocampal", "L"), ("ParaHippocampal", "R"),
    ("Amygdala", "L"), ("Amygdala", "R"),
    ("Calcarine", "L"), ("Calcarine", "R"),
    ("Cuneus", "L"), ("Cuneus", "R"),
    ("Lingual", "L"), ("Lingual", "R"),
    ("Occipital_Sup", "L"), ("Occipital_Sup", "R"),
    ("Occipital_Mid", "L"), ("Occipital_Mid", "R"),
    ("Occipital_Inf", "L"), ("Occipital_Inf", "R"),
    ("Fusiform", "L"), ("Fusiform", "R"),
    ("Postcentral", "L"), ("Postcentral", "R"),
    ("Parietal_Sup", "L"), ("Parietal_Sup", "R"),
    ("Parietal_Inf", "L"), ("Parietal_Inf", "R"),
    ("SupraMarginal", "L"), ("SupraMarginal", "R"),
    ("Angular", "L"), ("Angular", "R"),
    ("Precuneus", "L"), ("Precuneus", "R"),
    ("Paracentral_Lobule", "L"), ("Paracentral_Lobule", "R"),
    ("Caudate", "L"), ("Caudate", "R"),
    ("Putamen", "L"), ("Putamen", "R"),
    ("Pallidum", "L"), ("Pallidum", "R"),
    ("Thalamus", "L"), ("Thalamus", "R"),
    ("Heschl", "L"), ("Heschl", "R"),
    ("Temporal_Sup", "L"), ("Temporal_Sup", "R"),
    ("Temporal_Pole_Sup", "L"), ("Temporal_Pole_Sup", "R"),
    ("Temporal_Mid", "L"), ("Temporal_Mid", "R"),
    ("Temporal_Pole_Mid", "L"), ("Temporal_Pole_Mid", "R"),
    ("Temporal_Inf", "L"), ("Temporal_Inf", "R"),
    ("Cerebellum_Crus1", "L"), ("Cerebellum_Crus1", "R"),
    ("Cerebellum_Crus2", "L"), ("Cerebellum_Crus2", "R"),
    ("Cerebellum_3", "L"), ("Cerebellum_3", "R"),
    ("Cerebellum_4_5", "L"), ("Cerebellum_4_5", "R"),
    ("Cerebellum_6", "L"), ("Cerebellum_6", "R"),
    ("Cerebellum_7b", "L"), ("Cerebellum_7b", "R"),
    ("Cerebellum_8", "L"), ("Cerebellum_8", "R"),
    ("Cerebellum_9", "L"), ("Cerebellum_9", "R"),
    ("Cerebellum_10", "L"), ("Cerebellum_10", "R"),
    ("Vermis_1_2", ""), ("Vermis_3", ""),
    ("Vermis_4_5", ""), ("Vermis_6", ""),
    ("Vermis_7", ""), ("Vermis_8", ""),
    ("Vermis_9", ""), ("Vermis_10", ""),
]
assert len(AAL116_LABELS) == 116

# AAL24: bilateral groups contribute "<group> L" and "<group> R".
AAL24_BILATERAL = {
    "Sensorimotor": ["Precentral", "Postcentral", "Paracentral_Lobule",
                     "Rolandic_Oper", "Supp_Motor_Area"],
    "Prefrontal": ["Frontal_Sup", "Frontal_Mid", "Frontal_Inf_Oper",
                   "Frontal_Inf_Tri", "Frontal_Sup_Medial"],
    "Orbitofrontal": ["Frontal_Inf_Orb", "Frontal_Med_Orb", "Frontal_Sup_Orb",
                      "Frontal_Mid_Orb", "Rectus", "Olfactory"],
    "Parietal": ["Parietal_Sup", "Parietal_Inf", "Precuneus",
                 "SupraMarginal", "Angular"],
    "Temporal": ["Temporal_Sup", "Temporal_Mid", "Temporal_Inf",
                 "Temporal_Pole_Sup", "Temporal_Pole_Mid", "Heschl",
                 "Fusiform"],
    "Occipital": ["Occipital_Sup", "Occipital_Mid", "Occipital_Inf",
                  "Lingual", "Calcarine", "Cuneus"],
    "Insula": ["Insula"],
    "MTL": ["Hippocampus", "Amygdala", "ParaHippocampal"],
    "BasalGanglia": ["Caudate", "Putamen", "Pallidum"],
    "Cerebellum": ["Cerebellum_Crus1", "Cerebellum_Crus2", "Cerebellum_3",
                   "Cerebellum_4_5", "Cerebellum_6", "Cerebellum_7b",
                   "Cerebellum_8", "Cerebellum_9", "Cerebellum_10"],
}

# AAL24: midline groups are a single parcel built from (name, hemi) pairs.
AAL24_MIDLINE = {
    "AntCingulate": [("Cingulum_Ant", "L"), ("Cingulum_Ant", "R")],
    "PostCingulate": [("Cingulum_Mid", "L"), ("Cingulum_Mid", "R"),
                      ("Cingulum_Post", "L"), ("Cingulum_Post", "R")],
    "Thalamus": [("Thalamus", "L"), ("Thalamus", "R")],
    "Vermis": [("Vermis_1_2", ""), ("Vermis_3", ""), ("Vermis_4_5", ""),
               ("Vermis_6", ""), ("Vermis_7", ""), ("Vermis_8", ""),
               ("Vermis_9", ""), ("Vermis_10", "")],
}

AAL24_NAMES = (
    [f"{g} {h}" for g in AAL24_BILATERAL for h in ("L", "R")]
    + list(AAL24_MIDLINE)
)

_AAL116_TO_AAL24 = {
    **{(m, h): f"{g} {h}"
       for g, members in AAL24_BILATERAL.items()
       for m in members for h in ("L", "R")},
    **{pair: g for g, members in AAL24_MIDLINE.items() for pair in members},
}


def _atlas_file():
    from osl_dynamics.files import parcellation
    try:
        return parcellation.file(AAL116_FILE)  # downloaded on first use
    except AttributeError:  # older osl-dynamics, which bundles the atlas
        from osl_dynamics.files.functions import check_exists
        return check_exists(AAL116_FILE, parcellation.directory)


@lru_cache(maxsize=1)
def _atlas():
    img = nib.load(_atlas_file())
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    return data, np.linalg.inv(img.affine)


def aal116_at_mni(x, y, z, min_weight=0.05):
    """AAL116 label (e.g. ``"Hippocampus R"``) at an MNI coordinate in mm.

    Takes the parcel with the largest weight at the nearest voxel, or
    returns ``"Undefined"`` if no parcel has weight >= ``min_weight``.
    """
    data, inv = _atlas()
    i, j, k = np.round(inv @ [x, y, z, 1.0])[:3].astype(int)
    if not all(0 <= v < n for v, n in zip((i, j, k), data.shape[:3])):
        return "Undefined"
    weights = data[i, j, k]
    idx = int(np.argmax(weights))
    if weights[idx] < min_weight:
        return "Undefined"
    name, hemi = AAL116_LABELS[idx]
    return f"{name} {hemi}".strip()


def aal116_to_aal24(label):
    """AAL116 label -> AAL24 region (``"Undefined"`` if not recognised)."""
    if not isinstance(label, str) or label in ("", "Undefined"):
        return "Undefined"
    base, _, hemi = label.strip().rpartition(" ")
    if hemi not in ("L", "R"):
        base, hemi = label.strip(), ""
    return _AAL116_TO_AAL24.get((base, hemi), "Undefined")


def region_colors():
    """Fixed AAL24 region -> RGB colour, shared by all plots."""
    import matplotlib.pyplot as plt

    palette = list(plt.get_cmap("tab20").colors) + list(
        plt.get_cmap("tab20b").colors
    )
    return {r: palette[i] for i, r in enumerate(sorted(AAL24_NAMES))}


@lru_cache(maxsize=1)
def _scalp_tree():
    import mne
    from scipy.spatial import cKDTree

    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    surfs = mne.read_bem_surfaces(f"{fs_dir}/bem/fsaverage-head.fif",
                                  verbose=False)
    scalp = next(s for s in surfs if s["id"] == 4)["rr"] * 1000
    return cKDTree(scalp)


def depth(coords_mni):
    """Distance (mm) from MNI coordinates to the nearest point on the
    fsaverage scalp."""
    return _scalp_tree().query(np.atleast_2d(coords_mni))[0]
