# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

import os
import numpy as np
from tvb_nest.config import Config
from tvb_nest.examples.example import main_example
from tvb_nest.nest_models.builders.models.wilson_cowan import WilsonCowanBuilder
from tvb_nest.interfaces.builders.models.wilson_cowan \
    import WilsonCowanBuilder as InterfaceWilsonCowanBuilder
from tvb.simulator.models.wilson_cowan_constraint import WilsonCowan
from tvb.datatypes.connectivity import Connectivity


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

    model_params = {
            "r_e": np.array([0.0]),
            "r_i": np.array([0.0]),
            "k_e": np.array([1.0]),
            "k_i": np.array([1.0]),
            "tau_e": np.array([10.0]),
            "tau_i": np.array([10.0]),
            "c_ee": np.array([10.0]),
            "c_ei": np.array([6.0]),
            "c_ie": np.array([10.0]),
            "c_ii": np.array([1.0]),
            "alpha_e": np.array([1.2]),
            "alpha_i": np.array([2.0]),
            "a_e": np.array([1.0]),
            "a_i": np.array([1.0]),
            "b_e": np.array([0.0]),
            "b_i": np.array([0.0]),
            "c_e": np.array([1.0]),
            "c_i": np.array([1.0]),
            "theta_e": np.array([2.0]),
            "theta_i": np.array([3.5]),
            "P": np.array([0.5]),
            "Q": np.array([0.0])
        }

    results, simulator = \
        main_example(WilsonCowan, WilsonCowanBuilder, InterfaceWilsonCowanBuilder,
                     nest_nodes_ids, nest_populations_order=100,
                     tvb_to_nest_mode="rate", nest_to_tvb=True, exclusive_nodes=True,
                     connectivity=connectivity, delays_flag=True,
                     simulation_length=110.0, transient=10.0,
                     config=config, **model_params)

    np.save(os.path.join(config.out.FOLDER_RES, "connectivity_weights.npy"), simulator.connectivity.weights)
    np.save(os.path.join(config.out.FOLDER_RES, "connectivity_lengths.npy"), simulator.connectivity.tract_lengths)
    np.save(os.path.join(config.out.FOLDER_RES, "results.npy"), results[0][1])


if __name__ == "__main__":
    launch_example()
