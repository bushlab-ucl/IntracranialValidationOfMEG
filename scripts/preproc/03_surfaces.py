"""Extract the head and skull surfaces from each subject's T1w MRI (RHINO).

Uses FSL (BET, FLIRT and FNIRT): the MRI is registered to MNI space
nonlinearly, and this registration is later used to place the virtual
electrodes at the SEEG contact locations (given in MNI space).

Output: ``<out_dir>/preproc/surfaces/<subject>/``

Usage: python 03_surfaces.py [--subject SUBJECT ...] [--n-workers N]
"""

import argparse

import matplotlib
from osl_dynamics.meeg import parallel, rhino

from ivmeg import config, paths, release

matplotlib.use("Agg")

# BET fractional intensity threshold, for subjects where the default
# skull-strips poorly.
BET_FVAL = {"sub-04": 0.6}


def process(sub):
    rhino.extract_surfaces(
        mri_file=str(release.t1_file(sub)),
        outdir=str(paths.surfaces(sub)),
        include_nose=False,
        bet_fval=BET_FVAL.get(sub),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", nargs="+", choices=config.subjects())
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    subjects = args.subject if args.subject else config.subjects()
    parallel.run(process, items=subjects, n_workers=args.n_workers,
                 log_dir=paths.logs("03_surfaces"))
