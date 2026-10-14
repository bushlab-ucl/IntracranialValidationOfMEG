"""SEEG preprocessing.

1. Take the 60 SEEG amplifier inputs from the simultaneous MEG fif.
2. Detect and drop bad channels on the wide-band monopolar signal.
3. Notch filter (50, 100 Hz), band-pass 2-95 Hz, resample to 250 Hz.
4. Bipolar montage: adjacent contacts on the same shaft < 6 mm apart.
5. Detect bad segments on the bipolar signal (generalised ESD; standard
   deviation, its temporal derivative, kurtosis and its derivative).
6. Annotate +/- 0.5 s around each clinician-marked IED as ``bad_ied``.
7. Drop low-variance bipolar channels.

Output: ``<out_dir>/preproc/seeg/<session>_ieeg-bipolar-raw.fif`` and a
summary of the channels and time removed per session.

Usage: python 01_seeg.py [--session SESSION ...] [--n-workers N]
"""

import argparse

import matplotlib
import mne
import numpy as np
import pandas as pd
from osl_dynamics.meeg import parallel, preproc

from ivmeg import config, electrodes, ieds, paths, release

matplotlib.use("Agg")

L_FREQ, H_FREQ = 2, 95
NOTCH_FREQS = [50, 100]

# Bad segments: generalised ESD, 0.25 s windows, at most 30% per pass.
BAD_SEG = dict(significance_level=0.1, maximum_fraction=0.3)
BAD_SEG_WINDOW_S = 0.25
BAD_SEG_PASSES = [
    dict(),
    dict(mode="diff"),
    dict(metric="kurtosis"),
    dict(mode="diff", metric="kurtosis"),
]

# Low-variance bipolars: log-variance below median - 3 * MAD.
LOWVAR_N_MAD = 3


def bad_seconds(raw, prefix):
    return sum(a["duration"] for a in raw.annotations
               if a["description"].startswith(prefix))


def process(item):
    sid = item["id"]
    sub = config.subject(sid)

    raw = mne.io.read_raw_fif(release.meg_file(sid), preload=True)
    raw.pick("eeg")
    n_mono = len(raw.ch_names)

    # Bad monopolar channels are dropped so they can't enter a bipolar pair.
    raw = preproc.detect_bad_channels(raw, picks="eeg")
    bad_mono = list(raw.info["bads"])
    raw.drop_channels(bad_mono)
    print(f"Dropped {len(bad_mono)} bad monopolar channels: {bad_mono}")

    # Bipolar pairs from the localised contacts, named by clinical name.
    el = release.electrodes(sub)
    pairs = [(a, c) for a, c in electrodes.bipolar_pairs(el)
             if a in raw.ch_names and c in raw.ch_names]
    to_name = dict(zip(el["eeg_channel"], el["clinical_name"]))
    raw.rename_channels({ch: to_name[ch] for ch in raw.ch_names
                         if ch in to_name})
    pairs = [(to_name[a], to_name[c]) for a, c in pairs]
    print(f"{len(pairs)} bipolar pairs")

    raw.notch_filter(NOTCH_FREQS)
    raw.filter(L_FREQ, H_FREQ, method="iir",
               iir_params={"order": 5, "ftype": "butter"})
    raw.resample(config.SFREQ)

    names = [f"{a}-{c}" for a, c in pairs]
    bip = mne.set_bipolar_reference(
        raw, [a for a, _ in pairs], [c for _, c in pairs],
        ch_name=names, copy=True,
    )
    bip.pick(names)

    window = int(round(BAD_SEG_WINDOW_S * bip.info["sfreq"]))
    for kw in BAD_SEG_PASSES:
        bip = preproc.detect_bad_segments(
            bip, picks="eeg", window_length=window, **BAD_SEG, **kw,
        )

    n_ied = ieds.annotate(bip, sid)
    print(f"Added {n_ied} IED windows")

    log_var = np.log10(np.var(bip.get_data(), axis=1) + 1e-30)
    med = np.median(log_var)
    mad = np.median(np.abs(log_var - med))
    lowvar = [ch for ch, v in zip(bip.ch_names, log_var)
              if v < med - LOWVAR_N_MAD * mad]
    bip.drop_channels(lowvar)
    print(f"Dropped {len(lowvar)} low-variance bipolars: {lowvar}")

    bip.save(paths.seeg_bipolar(sid), overwrite=True)

    dur = bip.times[-1]
    pd.DataFrame([{
        "session": sid,
        "n_monopolar": n_mono,
        "n_monopolar_bad": len(bad_mono),
        "n_bipolar_built": len(pairs),
        "n_bipolar_lowvar": len(lowvar),
        "n_bipolar": len(bip.ch_names),
        "duration_s": round(dur, 1),
        "pct_bad": round(100 * bad_seconds(bip, "bad") / dur, 1),
        "pct_bad_segment": round(100 * bad_seconds(bip, "bad_segment") / dur, 1),
        "pct_bad_ied": round(100 * bad_seconds(bip, "bad_ied") / dur, 1),
    }]).to_csv(paths.logs("01_seeg") / f"{sid}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--session", nargs="+", choices=list(config.SESSIONS))
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    sessions = args.session if args.session else list(config.SESSIONS)
    parallel.run(process, items=[{"id": s} for s in sessions],
                 n_workers=args.n_workers, log_dir=paths.logs("01_seeg"))

    rows = [paths.logs("01_seeg") / f"{s}.csv" for s in config.SESSIONS]
    rows = [pd.read_csv(p) for p in rows if p.exists()]
    if rows:
        pd.concat(rows).to_csv(paths.seeg_summary(), index=False)
