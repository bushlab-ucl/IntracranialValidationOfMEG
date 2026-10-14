"""Power spectra of each bipolar SEEG channel and MEG virtual electrode (VE).

Each channel is z-scored within session (spectra are in fractional variance
per Hz) and its PSD is estimated with Welch's method (2 s segments), excluding
the bad segments of that modality. For Figure S2, the SEEG spectra are also
computed after adding white noise (sigma = 1) to the z-scored SEEG.

Output: ``<out_dir>/analysis/psd/<session>.npz``

Usage: python psd.py
"""

import warnings

import mne
import numpy as np

from ivmeg import config, data, electrodes, paths
from ivmeg.spectral import welch

FMIN, FMAX = 1.0, 95.0
N_FFT = 500  # 2 s
NOISE_SD = 1.0
SEED = 0

# MNE warns for good segments shorter than N_FFT, which use a shorter window
warnings.filterwarnings("ignore", message="nperseg = ")


def zscore_raw(raw):
    return raw.apply_function(data.zscore, channel_wise=False)


def mne_psd(raw):
    spectrum = raw.compute_psd(method="welch", fmin=FMIN, fmax=FMAX,
                               n_fft=N_FFT, reject_by_annotation=True)
    return spectrum.freqs, spectrum.get_data()


out_dir = paths.out_dir("analysis", "psd")
rng = np.random.default_rng(SEED)
for sid in config.SESSIONS:
    bp = electrodes.bipolar_midpoints(sid)

    raw = mne.io.read_raw_fif(paths.seeg_bipolar(sid), preload=True)
    raw.pick(bp["bipolar_name"].tolist())
    freqs, psd_seeg = mne_psd(zscore_raw(raw.copy()))

    ve = mne.io.read_raw_fif(paths.ve(sid), preload=True)
    ve.set_channel_types({ch: "eeg" for ch in ve.ch_names})
    _, psd_ve = mne_psd(zscore_raw(ve))

    # SEEG with and without added white noise (good segments only)
    x = data.zscore(raw.get_data(reject_by_annotation="omit"))
    f, clean = welch(x, raw.info["sfreq"], N_FFT / raw.info["sfreq"])
    noisy = []
    for row in x:
        noisy.append(data.zscore(row + rng.normal(0, NOISE_SD, row.shape)))
    _, noisy = welch(np.array(noisy), raw.info["sfreq"],
                     N_FFT / raw.info["sfreq"])
    fmask = (f >= FMIN) & (f <= FMAX)

    np.savez(
        out_dir / f"{sid}.npz",
        freqs=freqs, seeg=psd_seeg, ve=psd_ve,
        seeg_names=np.array(bp["bipolar_name"], dtype=str),
        ve_names=np.array(ve.ch_names, dtype=str),
        region=np.array(bp["region_aal24"], dtype=str),
        mni=bp[["mni_x", "mni_y", "mni_z"]].to_numpy(),
        noise_freqs=f[fmask], seeg_clean=clean[:, fmask],
        seeg_noisy=noisy[:, fmask],
    )
    print(f"{sid}: {len(bp)} SEEG, {len(ve.ch_names)} VE channels")
