# -*- coding: utf-8 -*-

import os
import numpy as np

# Set up the environment
from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.core.config import Config


def configure(config_class=Config):

    # ----------- Simulation options ----------------
    SIM_MODE = "tvb-spikeNet"  # "tvb-spikeNet"  for multiscale cosimulation, "tvb" ("spikeNet")
                               # for only TVB (spikeNET) simulation, respectively
    SPIKENET_MODEL_BUILDERS = None  # only None will work!, "opinionated", "nonopinionated", None

    # For a minimal example, select:
    n_regions = 4  # total TVB brain regions
    spiking_nodes_inds = np.array(
        [0, 1])  # the brain region nodes to place spiking networks from [0, n_regions-1] interval
    n_neurons = 10  # number of neurons per spiking population
    # -----------------------------------------------

    # Base paths
    work_path = os.getcwd()
    outputs_path = os.path.join(work_path, "outputs/WilsonCowanMin/Front_Back_End_Separated_Trans")
    if SPIKENET_MODEL_BUILDERS is None:
        outputs_path += "NoSpikeNetBuilders"
    elif SPIKENET_MODEL_BUILDERS == "opinionated":
        outputs_path += "OpinionBuilders"
    elif SPIKENET_MODEL_BUILDERS == "nonopinionated":
        outputs_path += "NonOpinionBuilders"

    if SIM_MODE.lower() == "spikeNet":
        outputs_path += "spikeNetonly"
    elif SIM_MODE.lower() == "tvb":
        outputs_path += "TVBonly"

    # Generate a configuration class instance
    config = config_class(output_base=outputs_path)
    config.figures.SHOW_FLAG = True
    config.figures.SAVE_FLAG = True
    config.figures.FIG_FORMAT = 'png'
    # config.figures.DEFAULT_SIZE = config.figures.NOTEBOOK_SIZE

    return config, SIM_MODE, n_regions, SPIKENET_MODEL_BUILDERS, spiking_nodes_inds, n_neurons
