# Intracranial Validation of MEG

Code to reproduce the results of *Intracranial Validation of Magnetoencephalography
Across Oscillatory Frequency and Depth* (Gohil et al.), from simultaneous resting-state
MEG and stereo-EEG (SEEG) recordings in 14 patients with focal epilepsy.

Every figure panel in the paper is produced by the scripts here, starting from the
public dataset.

## Data

The analyses use the public MEG-SEEG dataset (BIDS; DOI to be added), which contains
for each of the 14 participants (16 sessions):

- the simultaneous MEG and SEEG recording (`meg/*_meg.fif`) and a SEEG-only copy
  (`ieeg/`), with the clinician-marked interictal epileptiform discharges
  (`*_events.tsv`),
- the SEEG contact locations in MNI space (`ieeg/*_electrodes.tsv`),
- the defaced T1-weighted MRI (`anat/`),
- FastSurfer segmentations of the MRI (`derivatives/fastsurfer/`), used to identify
  the hippocampal contacts.

## Installation

```bash
conda env create -f environment.yml
conda activate ivmeg
```

This installs the `ivmeg` helper package from this repository in editable mode,
together with MNE-Python and [osl-dynamics](https://github.com/OHBA-analysis/osl-dynamics)
(3.7.2). The preprocessing also needs [FSL](https://fsl.fmrib.ox.ac.uk/fsl/) (6.0.7).

Then copy `config.example.yaml` to `config.yaml` and set:

```yaml
data_dir: /path/to/meg-seeg-release   # root of the BIDS dataset
out_dir: /path/to/derivatives          # where all intermediate outputs are written
```

## Running

```bash
./run_all.sh            # everything: preprocessing, analyses, figures
./run_all.sh preproc    # or one stage at a time: preproc, analysis, figures
```

On a 10-core laptop the preprocessing takes 1-2 hours, the analyses 30-40 minutes and
the figures about a minute. The individual scripts can also be run on their own
(`python scripts/<stage>/<script>.py`) in the order listed in `run_all.sh`; the
preprocessing scripts accept `--session` and `--n-workers`.

Figure panels are written to `figures/`; everything else goes to `out_dir`.

## What each script does

**Preprocessing** (`scripts/preproc/`):

| Script | Step |
|---|---|
| `01_seeg.py` | SEEG: bad channels, filtering, bipolar montage, bad segments, IED windows |
| `02_meg.py` | MEG: filtering, bad segments and channels (tSSS was applied at acquisition) |
| `03_surfaces.py` | Head surfaces and nonlinear (FNIRT) registration to MNI space (RHINO) |
| `04_coreg_bem.py` | MEG-MRI coregistration and boundary element model |
| `05_ve.py` | LCMV virtual electrodes (VEs) at the bipolar contacts, broadband and per band |
| `06_null_ves.py` | Random-location and depth-matched null VEs |

**Figures** (`scripts/analysis/` computes, `scripts/figures/` plots):

| Figure | Analysis | Plot | Panel(s) in `figures/` |
|---|---|---|---|
| 1A | (schematic) | | |
| 1B | | `fig1b_demographics.py` | `fig1b_demographics.png` |
| 1C | | `fig1c_contacts.py` | `fig1c_contact_counts.png`, `fig1c_glass_brain.png` |
| 1D | `cca_sensor.py` | `fig1d_cca_sensor.py` | `fig1d_cca_sensor.png` |
| 2A | `psd.py` | `fig2a_psd.py` | `fig2a_psd.png` |
| 2B | `band_power.py` | `fig2b_band_power.py` | `fig2b_band_power.png` |
| 3 | `amp_modulation.py` | `fig3_amp_modulation.py` | `fig3_amp_modulation.png` |
| 4A | `hipp_power.py` | `fig4a_hipp_power.py` | `fig4a_hipp_psd.png`, `fig4a_hipp_power_corr.png` |
| 4B | `hipp_amp_modulation.py` | `fig4b_hipp_amp_modulation.py` | `fig4b_hipp_amp_modulation.png` |
| S1 | `cca_ve.py` | `figs1_cca_ve.py` | `figs1a_cca_ve.png`, `figs1b_cca_ve_perband.png` |
| S2 | `psd.py` | `figs2_psd_noise.py` | `figs2_psd_noise.png` |
| S3 | `band_power_null.py` | `figs3_band_power_null.py` | `figs3_band_power_null.png` |
| S4 | `amp_modulation_null.py` | `figs4_amp_modulation_null.py` | `figs4_amp_modulation_null.png` |
| S5 | `hipp_cca_sensor.py` | `figs5_hipp_topography.py` | `figs5_hipp_topography.png` |
| S6 | `hipp_power_null.py` | `figs6_hipp_power_null.py` | `figs6_hipp_power_null.png` |
| S7 | `aec.py` | `figs7_aec.py` | `figs7a_edge_coverage.png` ... `figs7d_aec_scatter.png` |

Each script starts with a description of what it computes.

## Repository layout

```
ivmeg/          helper package: configuration, data loading, atlas lookups,
                source reconstruction, spectra, CCA, bursts, connectivity
scripts/        preproc/, analysis/ and figures/ (see above)
figures/        the figure panels
run_all.sh      runs everything in order
```

## Citation

If you use this code, please cite the paper (reference to be added).

## Licence

BSD 3-Clause; see `LICENSE`.
