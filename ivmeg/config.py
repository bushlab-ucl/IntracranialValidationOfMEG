"""Paths, sessions and analysis-wide constants.

Paths are read from ``config.yaml`` at the repository root (copy
``config.example.yaml``), or from the file named by the ``IVMEG_CONFIG``
environment variable:

    data_dir: /path/to/release      # root of the MEG-SEEG BIDS release
    out_dir:  /path/to/derivatives  # everything this repo writes
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _load():
    path = Path(os.environ.get("IVMEG_CONFIG", REPO_DIR / "config.yaml"))
    if not path.exists():
        raise FileNotFoundError(
            f"No config at {path}. Copy config.example.yaml to config.yaml "
            "and set data_dir and out_dir."
        )
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {k: Path(v).expanduser() for k, v in cfg.items()}


def data_dir():
    """Root of the BIDS release."""
    return _load()["data_dir"]


def out_dir(*parts):
    """Output directory (created on demand), optionally a sub-directory."""
    p = _load()["out_dir"].joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------------
# One entry per resting-state session in the release: session id -> seed.
#
# The seed is the session's index in the session list of the original
# analysis, so the random-location and depth-matched null virtual
# electrodes are identical to the ones used in the paper.
SESSIONS = {
    "sub-01_run-001": 0,
    "sub-02_run-001": 1,
    "sub-03_run-001": 2,
    "sub-04_run-001": 3,
    "sub-05_run-001": 6,
    "sub-05_run-002": 7,
    "sub-06_run-004": 8,
    "sub-07_run-001": 9,
    "sub-07_run-003": 10,
    "sub-08_run-001": 11,
    "sub-09_run-001": 12,
    "sub-10_run-001": 13,
    "sub-11_run-001": 14,
    "sub-12_run-001": 15,
    "sub-13_run-001": 16,
    "sub-14_run-001": 17,
}


def subject(session_id):
    """``sub-05_run-002`` -> ``sub-05``."""
    return session_id.split("_")[0]


def run(session_id):
    """``sub-05_run-002`` -> ``002``."""
    return session_id.split("_run-")[1]


def subjects():
    return sorted({subject(s) for s in SESSIONS})


# ----------------------------------------------------------------------------
# Analysis constants
# ----------------------------------------------------------------------------
SFREQ = 250  # Hz, after preprocessing

BANDS = {
    "delta": (2, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "low_gamma": (30, 45),
    "high_gamma": (55, 95),
}

# Region filter for the region-level analyses: a region is analysed only if
# it has contacts in at least MIN_SESSIONS sessions and MIN_CONTACTS bipolar
# contacts pooled over sessions.
MIN_SESSIONS = 3
MIN_CONTACTS = 10
