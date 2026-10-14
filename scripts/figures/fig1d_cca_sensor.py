"""Fig 1D: CCA between the MEG sensors and the SEEG of each region against
the region's depth."""

from ivmeg import cca, paths

cca.plot_depth(paths.out_dir("analysis", "cca") / "sensor.npz",
               paths.figures() / "fig1d_cca_sensor.png")
