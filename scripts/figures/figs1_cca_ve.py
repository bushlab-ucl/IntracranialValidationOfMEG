"""Fig S1: CCA between the MEG virtual electrodes and the SEEG of each region
against the region's depth, for the broadband (A) and per-band (B)
beamformers."""

from ivmeg import cca, paths

for name, panel in (("ve", "a"), ("ve_perband", "b")):
    print(f"--- Fig S1{panel.upper()} ({name})")
    cca.plot_depth(paths.out_dir("analysis", "cca") / f"{name}.npz",
                   paths.figures() / f"figs1{panel}_cca_{name}.png")
