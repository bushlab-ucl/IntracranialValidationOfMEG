"""Locations of everything this repo writes: under ``out_dir``, except the
figure panels (``figures/`` at the repository root)."""

from ivmeg.config import REPO_DIR, out_dir


def figures():
    """Figure panels, in ``figures/`` at the repository root."""
    p = REPO_DIR / "figures"
    p.mkdir(exist_ok=True)
    return p


def logs(step):
    return out_dir("logs", step)


def qc(step):
    return out_dir("qc", step)



def seeg_bipolar(session_id):
    return out_dir("preproc", "seeg") / f"{session_id}_ieeg-bipolar-raw.fif"


def seeg_summary():
    return out_dir("preproc", "seeg") / "summary.csv"


def meg_preproc(session_id):
    return out_dir("preproc", "meg") / f"{session_id}_preproc-raw.fif"


def surfaces(sub):
    return out_dir("preproc", "surfaces", sub)


def osl_session(session_id):
    """osl-dynamics Session holding the coregistration, BEM and forward
    model of a session."""
    from osl_dynamics.meeg import Session

    from ivmeg.config import subject

    return Session(
        outdir=str(out_dir("preproc", "osl")),
        id=session_id,
        preproc_file=str(meg_preproc(session_id)),
        surfaces_dir=str(surfaces(subject(session_id))),
    )


def ve(session_id, band=None):
    """MEG virtual electrodes at the bipolar SEEG contacts. ``band=None`` is
    the broadband beamformer; otherwise the beamformer for that band."""
    tag = "" if band is None else f"_{band}"
    return out_dir("preproc", "ve") / f"{session_id}_ve{tag}-raw.fif"


def ve_sources(session_id):
    return out_dir("preproc", "ve") / f"{session_id}_sources.csv"


def null_ve(session_id, kind, band):
    """Null VEs; ``kind`` is ``"random"`` or ``"matched_depth"``."""
    return out_dir("preproc", f"null_{kind}") / f"{session_id}_{band}-raw.fif"


def null_sources(session_id, kind):
    return out_dir("preproc", f"null_{kind}") / f"{session_id}_sources.csv"
