# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

import os
import numpy as np
from tvb_multiscale.tvb_nest.config import Config
from tvb.datatypes.connectivity import Connectivity

from examples.tvb_nest.models.wilson_cowan import wilson_cowan_example


def launch_example():

    config = Config(output_base="outputs/")
    config.figures.SAVE_FLAG = False
    config.figures.SHOW_FLAG = False
    config.figures.MATPLOTLIB_BACKEND = "Agg"

    # Select the regions for the fine scale modeling with NEST spiking networks
    nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
    # In this example, we model parahippocampal cortices (left and right) with NEST
    connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)
    for id in range(connectivity.region_labels.shape[0]):
        if connectivity.region_labels[id].find("hippo") > 0:
            nest_nodes_ids.append(id)

    results, simulator = wilson_cowan_example(connectivity=connectivity, config=config)

    np.save(os.path.join(config.out.FOLDER_RES, "connectivity_weights.npy"), simulator.connectivity.weights)
    np.save(os.path.join(config.out.FOLDER_RES, "connectivity_lengths.npy"), simulator.connectivity.tract_lengths)
    np.save(os.path.join(config.out.FOLDER_RES, "results.npy"), results[0][1])


if __name__ == "__main__":
    launch_example()
