"""MEG preprocessing.

The release MEG has already been cleaned with temporal signal space
separation (tSSS, applied at acquisition).

1. Notch filter (50, 100 Hz), band-pass 2-95 Hz, resample to 250 Hz.
2. Detect bad segments (generalised ESD on the standard deviation and its
   temporal derivative), separately for magnetometers and gradiometers.
3. Detect bad channels, separately for magnetometers and gradiometers.

Output: ``<out_dir>/preproc/meg/<session>_preproc-raw.fif`` and QC plots.

Usage: python 02_meg.py [--session SESSION ...] [--n-workers N]
"""

import argparse

import matplotlib
import mne
from osl_dynamics.meeg import parallel, preproc

from ivmeg import config, paths, release

matplotlib.use("Agg")

L_FREQ, H_FREQ = 2, 95
NOTCH_FREQS = [50, 100]


def process(item):
    sid = item["id"]

    raw = mne.io.read_raw_fif(release.meg_file(sid), preload=True)
    raw.notch_filter(NOTCH_FREQS)
    raw.filter(L_FREQ, H_FREQ, method="iir",
               iir_params={"order": 5, "ftype": "butter"})
    raw.resample(config.SFREQ)

    for picks in ("mag", "grad"):
        raw = preproc.detect_bad_segments(raw, picks=picks)
        raw = preproc.detect_bad_segments(raw, picks=picks, mode="diff")
    for picks in ("mag", "grad"):
        raw = preproc.detect_bad_channels(raw, picks=picks)

    preproc.save_qc_plots(paths.qc("02_meg"), sid, raw)
    raw.save(paths.meg_preproc(sid), overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--session", nargs="+", choices=list(config.SESSIONS))
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    sessions = args.session if args.session else list(config.SESSIONS)
    parallel.run(process, items=[{"id": s} for s in sessions],
                 n_workers=args.n_workers, log_dir=paths.logs("02_meg"))
