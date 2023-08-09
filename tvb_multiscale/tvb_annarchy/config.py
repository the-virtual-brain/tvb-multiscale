# -*- coding: utf-8 -*-

import os

import numpy as np

from tvb_multiscale.core.config import Config as ConfigBase
from tvb_multiscale.core.utils.log_utils import initialize_logger as initialize_logger_base
from tvb_multiscale.core.utils.file_utils import get_tvb_annarchy_path_from_abs_filepath


TVB_ANNARCHY_DIR = get_tvb_annarchy_path_from_abs_filepath(os.path.abspath(__file__))
WORKING_DIR = os.environ.get("WORKING_DIR", os.getcwd())
MYMODELS_DIR = os.environ.get("MYMODELS_DIR",
                              os.path.join(TVB_ANNARCHY_DIR, "annarchy", "models"))


class Config(ConfigBase):
    # WORKING DIRECTORY:
    TVB_ANNARCHY_DIR = TVB_ANNARCHY_DIR
    WORKING_DIR = WORKING_DIR
    MYMODELS_DIR = MYMODELS_DIR

    ANNARCHY_SEED = 0

    VERBOSE = False

    MIN_SPIKING_DT = 0.001

    DEFAULT_SPIKING_MODEL = "IF_cond_alpha"

    # Delays should be at least equal to ANNarchy time resolution
    DEFAULT_SYNAPSE = "DefaultSpikingSynapse"
    DEFAULT_CONNECTION = {"synapse_model": DEFAULT_SYNAPSE, "params": {},
                          "weight": 1.0, "delay": 0.01, 'receptor_type': "exc",
                          "source_inds": None, "target_inds": None,
                          "syn_spec": {"synapse_model": DEFAULT_SYNAPSE, "params": {}},
                          "conn_spec": {"rule": "all_to_all"}}  # , "allow_self_connections": True, force_multiple_weights: False??

    DEFAULT_TVB_TO_ANNARCHY_INTERFACE = "PoissonPopulation"
    DEFAULT_ANNARCHY_TO_TVB_INTERFACE = "spike_monitor"

    # Available ANNARCHY output devices for the interface and their default properties
    ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF = {"SpikeMonitor": {"variables": "spike"},
                                          "spike_multimeter": {"variables": "spike", "period": 1.0},
                                          "Monitor": {"variables": ["v", 'g_exc', 'g_inh'], "period": 1.0}}

    ANNARCHY_INPUT_DEVICES_PARAMS_DEF = {"SpikeSourceArray": {"spike_times": [[0.1]]},
                                         "PoissonPopulation": {"rates": 0.0},
                                         "Poisson_neuron": {"rates": 0.0},
                                         "HomogeneousCorrelatedSpikeTrains": {"rates": [0.001], "corr": 0.0, "tau": 1.0,
                                                                              "schedule": [0.0], "period": -1.0,
                                                                              "refractory": None},
                                         "TimedArray": {"rates": np.array([[0.0]]), "schedule": [0.0], "period": -1.0,
                                                        "proxy": True, "proxy_target": "exc"},
                                         "TimedPoissonPopulation": {"rates": np.array([[0.0]]),
                                                                    "schedule": [0.0], "period": -1.0},
                                         }

    def __init__(self, output_base=None, separate_by_run=False, initialize_logger=True, verbosity=1):
        super(Config, self).__init__(output_base, separate_by_run, initialize_logger, verbosity)
        self.TVB_ANNARCHY_DIR = TVB_ANNARCHY_DIR
        self.WORKING_DIR = WORKING_DIR
        self.MYMODELS_DIR = MYMODELS_DIR
        self.MYMODELS_IMPORT_PATH = self.MYMODELS_DIR.split("tvb-multiscale")[-1].replace("/", ".")


CONFIGURED = Config(initialize_logger=False)


def initialize_logger(name="tvb_annarchy", target_folder=None, config=CONFIGURED):
    if target_folder is None:
        target_folder = config.out.FOLDER_LOGS
    return initialize_logger_base(name, target_folder)
