# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

import os
import numpy as np
from tvb_multiscale.tvb_nest.config import Config
from tvb.datatypes.connectivity import Connectivity

from examples.tvb_nest.example import default_example


def launch_example(write_files=True, **kwargs):

    config = Config(output_base="outputs/")
    config.figures.SAVE_FLAG = False
    config.figures.SHOW_FLAG = False
    config.figures.MATPLOTLIB_BACKEND = "Agg"

    connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)

    results, simulator = default_example(connectivity=connectivity, config=config, **kwargs)

    if write_files:
        np.save(os.path.join(config.out.FOLDER_RES, "connectivity_weights.npy"), simulator.connectivity.weights)
        np.save(os.path.join(config.out.FOLDER_RES, "connectivity_lengths.npy"), simulator.connectivity.tract_lengths)
        np.save(os.path.join(config.out.FOLDER_RES, "results.npy"), results[0][1])

    return simulator, results[0][1]


if __name__ == "__main__":
    launch_example()
