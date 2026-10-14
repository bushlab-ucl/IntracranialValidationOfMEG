"""Coregister the MEG to the MRI and build the BEM (RHINO).

1. Extract the Polhemus fiducials and head shape points from the MEG fif
   and remove stray head shape points (e.g. on the nose).
2. Coregister the head shape to the scalp surface (without the nose).
3. Single-shell BEM and forward model.

Output: ``<out_dir>/preproc/osl/<session>/{coreg,bem}/``

Usage: python 04_coreg_bem.py [--session SESSION ...] [--n-workers N]
"""

import argparse

import matplotlib
from osl_dynamics.meeg import parallel, rhino

from ivmeg import config, paths

matplotlib.use("Agg")


def process(item):
    session = paths.osl_session(item["id"])
    rhino.extract_fiducials_and_headshape_from_fif(session)
    rhino.remove_stray_headshape_points(session)
    rhino.coregister_head_and_mri(session, use_nose=False)
    rhino.forward_model(session, model="Single Layer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--session", nargs="+", choices=list(config.SESSIONS))
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    sessions = args.session if args.session else list(config.SESSIONS)
    parallel.run(process, items=[{"id": s} for s in sessions],
                 n_workers=args.n_workers, log_dir=paths.logs("04_coreg_bem"),
                 output_dir=paths.out_dir("preproc", "osl"),
                 qc_dir=paths.qc("04_coreg_bem"))
