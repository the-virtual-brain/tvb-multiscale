# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

import os
import numpy as np


def launch_example(config_type, example_fun, write_files=True, **kwargs):

    config = config_type(output_base="outputs/")
    config.figures.SAVE_FLAG = False
    config.figures.SHOW_FLAG = False
    config.figures.MATPLOTLIB_BACKEND = "Agg"

    simulation_length = kwargs.pop("simulation_length", 11.0)
    transient = kwargs.pop("simulation_length", 11.0) / 11
    results, simulator = example_fun(config=config, simulation_length=simulation_length, transient=transient, **kwargs)

    if write_files:
        np.save(os.path.join(config.out.FOLDER_RES, "connectivity_weights.npy"), simulator.connectivity.weights)
        np.save(os.path.join(config.out.FOLDER_RES, "connectivity_lengths.npy"), simulator.connectivity.tract_lengths)
        np.save(os.path.join(config.out.FOLDER_RES, "results.npy"), results[0][1])

    return simulator, results[0][1]


def launch_example_nest(write_files=True, **kwargs):

    from tvb_multiscale.tvb_nest.config import Config as NESTConfig
    from examples.tvb_nest.example import default_example as nest_default_example

    return launch_example(NESTConfig, nest_default_example, write_files, **kwargs)


def launch_example_annarchy(write_files=True, **kwargs):

    from tvb_multiscale.tvb_annarchy.config import Config as ANNarchyConfig
    from examples.tvb_annarchy.example import default_example as annarchy_default_example

    return launch_example(ANNarchyConfig, annarchy_default_example, write_files, **kwargs)


def launch_example_netpyne(write_files=True, **kwargs):

    from tvb_multiscale.tvb_netpyne.config import Config as NetPyNEConfig
    from examples.tvb_netpyne.example import default_example as netpyne_default_example

    return launch_example(NetPyNEConfig, netpyne_default_example, write_files, **kwargs)


if __name__ == "__main__":
    launch_example_nest()
    launch_example_annarchy()
