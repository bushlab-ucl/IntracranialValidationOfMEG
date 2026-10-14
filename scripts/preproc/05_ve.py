"""MEG virtual electrodes (VEs) at the bipolar SEEG contacts.

One VE per bipolar channel, at the midpoint of its two contacts and oriented
along the bipolar axis (anode -> cathode). VEs outside the inner skull are
dropped. Two sets of VEs are computed with an LCMV beamformer:

- broadband: one beamformer for the 2-95 Hz data,
- per band: the data are band-pass filtered and a beamformer is computed for
  each band.

Output: ``<out_dir>/preproc/ve/<session>_ve[_<band>]-raw.fif`` (channels named
after the bipolar SEEG channels) and ``<session>_sources.csv``.

Usage: python 05_ve.py [--session SESSION ...] [--n-workers N]
"""

import argparse

import mne
import numpy as np

from ivmeg import config, electrodes, paths, source
from osl_dynamics.meeg import parallel


def _xyz(df, prefix=""):
    return df[[f"{prefix}mni_x", f"{prefix}mni_y", f"{prefix}mni_z"]].to_numpy()


def process(item):
    sid = item["id"]
    session = paths.osl_session(sid)

    bp = electrodes.bipolar_midpoints(sid)
    pos = source.mni_to_mri(session, _xyz(bp))
    nn = source.axis_to_mri(session, _xyz(bp, "anode_"), _xyz(bp, "cathode_"))
    fwd, inside = source.forward(session, pos, nn)
    print(f"{inside.sum()}/{len(bp)} VEs inside the inner skull; "
          f"dropped {bp['bipolar_name'][~inside].tolist()}")

    # Adjacent bipolars on a shaft should be near parallel
    nn_by_name = dict(zip(bp["bipolar_name"], nn))
    for a, c in zip(bp["bipolar_name"], bp["bipolar_name"][1:]):
        if a.split("-")[1] == c.split("-")[0]:
            dot = nn_by_name[a] @ nn_by_name[c]
            if dot < 0.9:
                print(f"WARNING: {a} and {c} are not parallel (dot {dot:.2f})")

    bp = bp.assign(inside=inside, nn_mri_x=nn[:, 0], nn_mri_y=nn[:, 1],
                   nn_mri_z=nn[:, 2])
    bp.to_csv(paths.ve_sources(sid), index=False)
    names = bp["bipolar_name"][inside].tolist()

    raw = mne.io.read_raw_fif(paths.meg_preproc(sid), preload=True)
    source.beamform(raw, fwd, names).save(paths.ve(sid), overwrite=True)
    for band, limits in config.BANDS.items():
        ve = source.beamform(source.band_pass(raw, limits), fwd, names)
        ve.save(paths.ve(sid, band), overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--session", nargs="+", choices=list(config.SESSIONS))
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    sessions = args.session if args.session else list(config.SESSIONS)
    parallel.run(process, items=[{"id": s} for s in sessions],
                 n_workers=args.n_workers, log_dir=paths.logs("05_ve"))
