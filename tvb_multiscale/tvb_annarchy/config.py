# -*- coding: utf-8 -*-

import os

from tvb_multiscale.core.config import Config as ConfigBase
from tvb_multiscale.core.utils.log_utils import initialize_logger as initialize_logger_base

TVB_ANNARCHY_DIR = os.path.abspath(__file__).split("tvb_annarchy")[0]
WORKING_DIR = os.environ.get("WORKING_DIR", os.getcwd())
MYMODELS_DIR = os.environ.get("MYMODELS_DIR",
                              os.path.join(TVB_ANNARCHY_DIR, "tvb_annarchy/annarchy/models"))


class Config(ConfigBase):
    # WORKING DIRECTORY:
    TVB_ANNARCHY_DIR = TVB_ANNARCHY_DIR
    WORKING_DIR = WORKING_DIR
    MYMODELS_DIR = MYMODELS_DIR

    # TVB_ANNARCHY_DIR_VERBOCITY = ??

    ANNARCHY_SEED = 0

    ANNARCHY_MIN_DT = 0.001

    DEFAULT_MODEL = "izhikevich"

    # Delays should be at least equal to ANNarchy time resolution
    DEFAULT_CONNECTION = {"synapse_model": "DefaultSpikingSynapse", "params": {},
                          "weight": 1.0, "delay": 0.01, 'receptor_type': "exc",
                          "source_inds": None, "target_inds": None,
                          "conn_spec": {"method": "all_to_all"}}  # , "allow_self_connections": True, force_multiple_weights: False??

    DEFAULT_TVB_TO_ANNARCHY_INTERFACE = "PoissonPopulation"
    DEFAULT_ANNARCHY_TO_TVB_INTERFACE = "spike_monitor"

    # Available ANNARCHY output devices for the interface and their default properties
    ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF = {"SpikeMonitor": {"record_from": "spike", "period": 1.0},
                                          "spike_multimeter": {"record_from": "spike", "period": 1.0},
                                          "Monitor": {"record_from": ["v", 'g_exc', 'g_inh'], "period": 1.0}}

    ANNARCHY_INPUT_DEVICES_PARAMS_DEF = {"SpikeSourceArray": {"spike_times": []},
                                         "PoissonPopulation": {"rates": 0.0},
                                         "HomogeneousCorrelatedSpikeTrains":
                                             {"rates": 0.0, "corr": 0.0, "tau": 1.0},
                                         # "CurrentInjector": {"amplitude": 0.0},
                                         # "DCCurrentInjector": {"amplitude": 0.0},
                                         # "ACCurrentInjector": {"frequency": 0.0, "amplitude": 1.0,
                                         #                       "phase": 0.0, "offset": 0.0},
                                         "TimedArray": {"rates": 0.0, "schedule": 0.0, "period": -1.0},
                                         }

    def __init__(self, output_base=None, separate_by_run=False):
        super(Config, self).__init__(output_base, separate_by_run)
        self.TVB_ANNARCHY_DIR = TVB_ANNARCHY_DIR
        self.WORKING_DIR = WORKING_DIR
        self.MYMODELS_DIR = MYMODELS_DIR
        self.MYMODELS_IMPORT_PATH = self.MYMODELS_DIR.split("tvb-multiscale")[-1].replace("/", ".")


CONFIGURED = Config()


def initialize_logger(name, target_folder=CONFIGURED.out.FOLDER_LOGS):
    return initialize_logger_base(name, target_folder)
