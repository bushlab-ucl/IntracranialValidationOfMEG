"""Readers for the public MEG-SEEG release (BIDS).

Each session's MEG fif holds the 306 MEG channels and the 60 SEEG amplifier
inputs (``EEG001``-``EEG060``, type ``eeg``). Only localised contacts have an
entry in ``electrodes.tsv``, which links each amplifier input to its clinical
name and MNI coordinate.
"""

import numpy as np
import pandas as pd

from ivmeg.config import data_dir, run, subject


def _ses_dir(sub):
    return data_dir() / sub / "ses-001"


def meg_file(session_id):
    sub = subject(session_id)
    return (_ses_dir(sub) / "meg"
            / f"{sub}_ses-001_task-rest_run-{run(session_id)}_meg.fif")


def t1_file(sub):
    return _ses_dir(sub) / "anat" / f"{sub}_ses-001_T1w.nii.gz"


def segmentation(sub, desc="aseg"):
    """FastSurfer label map (``desc``: ``aseg`` or ``aparcaseg``) in the grid
    of the released T1w."""
    return (data_dir() / "derivatives" / "fastsurfer" / sub / "ses-001" / "anat"
            / f"{sub}_ses-001_space-T1w_desc-{desc}_dseg.nii.gz")


def participants():
    return pd.read_csv(data_dir() / "participants.tsv", sep="\t")


def electrodes(sub):
    """One row per localised contact.

    Columns: ``eeg_channel`` (amplifier input, e.g. ``EEG001``),
    ``clinical_name`` (e.g. ``R_HIPP1``), ``mni_x``, ``mni_y``, ``mni_z``,
    ``region_aal3``, ``group`` (shaft), ``hemisphere``.
    """
    path = (_ses_dir(sub) / "ieeg"
            / f"{sub}_ses-001_space-MNI152NLin2009bAsym_electrodes.tsv")
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={
        "amplifier_input": "eeg_channel", "name": "clinical_name",
        "x": "mni_x", "y": "mni_y", "z": "mni_z",
    })
    return df[["eeg_channel", "clinical_name", "mni_x", "mni_y", "mni_z",
               "region_aal3", "group", "hemisphere"]]


def ied_onsets(session_id):
    """Times (s from the start of the recording) of the clinician-marked IEDs.

    One entry per (event, contact) mark, as in ``events.tsv``.
    """
    sub = subject(session_id)
    path = (_ses_dir(sub) / "ieeg"
            / f"{sub}_ses-001_task-rest_run-{run(session_id)}_events.tsv")
    df = pd.read_csv(path, sep="\t")
    df = df[df["trial_type"] == "interictal_epileptiform_discharge"]
    return df["onset"].to_numpy(dtype=float)
