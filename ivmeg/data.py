"""Loading paired SEEG / MEG data for the analyses."""

from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from ivmeg import atlas, config, electrodes, paths

mne.set_log_level("ERROR")


def good_mask(raw, n=None):
    """True for samples not covered by a ``bad*`` annotation."""
    n = raw.n_times if n is None else n
    mask = np.ones(n, dtype=bool)
    sfreq = raw.info["sfreq"]
    for ann in raw.annotations:
        if ann["description"].startswith("bad"):
            start = int(ann["onset"] * sfreq) - raw.first_samp
            mask[max(0, start):max(0, start + int(ann["duration"] * sfreq))] \
                = False
    return mask


def contacts(session_id):
    """Bipolar SEEG channels of a session that have a virtual electrode
    (i.e. are inside the inner skull), with their location, depth and
    region."""
    df = pd.read_csv(paths.ve_sources(session_id))
    df = df[df["inside"]].drop(columns="inside").reset_index(drop=True)
    mni = df[["mni_x", "mni_y", "mni_z"]].to_numpy()
    df["depth_mm"] = atlas.depth(mni)
    df["hippocampus"] = electrodes.is_hippocampal(
        mni, config.subject(session_id))
    df.insert(0, "session", session_id)
    return df


def all_contacts():
    return pd.concat([contacts(s) for s in config.SESSIONS],
                     ignore_index=True)


def analysed_regions(df=None):
    """AAL24 regions that enter the region-level analyses: contacts in at
    least ``MIN_SESSIONS`` sessions and ``MIN_CONTACTS`` pooled."""
    df = all_contacts() if df is None else df
    df = df[df["region_aal24"] != "Undefined"]
    g = df.groupby("region_aal24")
    ok = (g["session"].nunique() >= config.MIN_SESSIONS) & \
         (g.size() >= config.MIN_CONTACTS)
    return sorted(ok.index[ok])


@dataclass
class Session:
    """Simultaneous SEEG and MEG of one session, sample-aligned.

    ``seeg`` and ``meg`` are (channels, samples) arrays over the full
    recording, ``good`` marks samples that are clean in both modalities.
    """

    id: str
    sfreq: float
    seeg: np.ndarray
    meg: np.ndarray
    good: np.ndarray
    seeg_names: list
    meg_names: list
    contacts: pd.DataFrame


def load(session_id, meg="ve", band=None):
    """Load a session.

    meg : ``"ve"`` for the virtual electrodes at the contacts (``band``
        selects the per-band beamformer, None the broadband one), or
        ``"sensor"`` for the MEG sensors (magnetometers and gradiometers,
        excluding bad channels).

    The SEEG is restricted to the contacts with a virtual electrode, in the
    order of ``contacts``. For ``meg="ve"`` the MEG channels are the same
    contacts.
    """
    df = contacts(session_id)
    names = df["bipolar_name"].tolist()
    raw_s = mne.io.read_raw_fif(paths.seeg_bipolar(session_id), preload=True)
    if meg == "ve":
        raw_m = mne.io.read_raw_fif(paths.ve(session_id, band), preload=True)
        meg_names = names
    elif meg == "sensor":
        raw_m = mne.io.read_raw_fif(paths.meg_preproc(session_id),
                                    preload=True)
        meg_names = [raw_m.ch_names[i]
                     for i in mne.pick_types(raw_m.info, meg=True)]
    else:
        raise ValueError(f"Unknown meg={meg!r}")
    if raw_s.first_samp != raw_m.first_samp:
        raise RuntimeError(f"{session_id}: SEEG and MEG are not aligned")

    n = min(raw_s.n_times, raw_m.n_times)
    return Session(
        id=session_id,
        sfreq=raw_s.info["sfreq"],
        seeg=raw_s.get_data(picks=names)[:, :n],
        meg=raw_m.get_data(picks=meg_names)[:, :n],
        good=good_mask(raw_s, n) & good_mask(raw_m, n),
        seeg_names=names,
        meg_names=meg_names,
        contacts=df,
    )


def bandpass(x, sfreq, band, order=4):
    """Zero-phase Butterworth band-pass along the last axis."""
    b, a = butter(order, band, btype="band", fs=sfreq)
    return filtfilt(b, a, x, axis=-1)


def zscore(x):
    sd = x.std(axis=-1, keepdims=True)
    return (x - x.mean(axis=-1, keepdims=True)) / np.where(sd == 0, 1, sd)
