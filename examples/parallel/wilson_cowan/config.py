# -*- coding: utf-8 -*-

import os
import numpy as np

# Set up the environment
from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_multiscale.config import Config


def configure(config_class=Config):

    # ----------- Simulation options ----------------
    SIM_MODE = "tvb-spikeNet"  # "tvb-spikeNet"  for multiscale cosimulation, "tvb" ("spikeNet")
                               # for only TVB (spikeNET) simulation, respectively
    SPIKENET_MODEL_BUILDERS = None  # only None will work!, "opinionated", "nonopinionated", None

    # For a minimal example, select:
    N_REGIONS = 4  # total TVB brain regions
    SPIKING_NODES_INDS = np.array(
        [0, 1])  # the brain region nodes to place spiking networks from [0, N_REGIONS-1] interval
    N_NEURONS = 10  # number of neurons per spiking population

    # Interface basic configurations:
    INTERFACE_MODEL = "RATE"  # "RATE" (or "SPIKES", "CURRENT") TVB->NEST interface
    TVB_TO_SPIKENET_PROXY_MODEL = "RATE"
    INTERFACE_COUPLING_MODE = "TVB"  # "spikeNet" # "TVB"
    EXCLUSIVE_NODES = True
    W_TVB_TO_SPIKENET = 5000.0  # TVB->NEST interface scaling weight
    # -----------------------------------------------

    # Base paths
    work_path = os.getcwd()
    outputs_path = os.path.join(work_path, "outputs/WilsonCowan")
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

    config.SIM_MODE = SIM_MODE
    config.N_REGIONS = N_REGIONS
    config.SPIKENET_MODEL_BUILDERS = SPIKENET_MODEL_BUILDERS
    config.SPIKING_NODES_INDS = SPIKING_NODES_INDS
    config.N_NEURONS = N_NEURONS
    config.INTERFACE_MODEL = INTERFACE_MODEL
    config.TVB_TO_SPIKENET_PROXY_MODEL = TVB_TO_SPIKENET_PROXY_MODEL
    config.INTERFACE_COUPLING_MODE = INTERFACE_COUPLING_MODE
    config.EXCLUSIVE_NODES = EXCLUSIVE_NODES
    config.W_TVB_TO_SPIKENET = W_TVB_TO_SPIKENET

    return config
