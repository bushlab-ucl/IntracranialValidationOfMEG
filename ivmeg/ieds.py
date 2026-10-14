"""Interictal epileptiform discharge (IED) windows."""

import mne
import numpy as np

from ivmeg import release

IED_HALF_WINDOW_S = 0.5


def pooled_windows(onsets, half_window_s=IED_HALF_WINDOW_S):
    """Merge ``onset +/- half_window_s`` windows pooled across contacts.

    Returns a list of ``(onset_s, duration_s)`` in seconds from the start of
    the recording.
    """
    if len(onsets) == 0:
        return []
    times = np.sort(np.asarray(onsets, dtype=float))
    merged = []
    start, end = times[0] - half_window_s, times[0] + half_window_s
    for t in times[1:]:
        if t - half_window_s <= end:
            end = max(end, t + half_window_s)
        else:
            merged.append((float(start), float(end - start)))
            start, end = t - half_window_s, t + half_window_s
    merged.append((float(start), float(end - start)))
    return merged


def annotate(raw, session_id, half_window_s=IED_HALF_WINDOW_S):
    """Add the session's pooled IED windows to ``raw`` as ``bad_ied``.

    Returns the number of windows added.
    """
    windows = pooled_windows(release.ied_onsets(session_id), half_window_s)
    if windows:
        t0 = raw.first_samp / raw.info["sfreq"]
        raw.set_annotations(raw.annotations + mne.Annotations(
            onset=[w[0] + t0 for w in windows],
            duration=[w[1] for w in windows],
            description=["bad_ied"] * len(windows),
            orig_time=raw.info["meas_date"],
        ))
    return len(windows)
