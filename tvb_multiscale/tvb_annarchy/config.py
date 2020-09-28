# -*- coding: utf-8 -*-

import os

from tvb_multiscale.core.config import Config as ConfigBase
from tvb_multiscale.core.utils.log_utils import initialize_logger as initialize_logger_base

TVB_ANNARCHY_DIR = os.path.abspath(__file__).split("tvb_annarchy")[0]
WORKING_DIR = os.environ.get("WORKING_DIR", os.getcwd())
MYMODULES_DIR = os.environ.get("MYMODULES_DIR",
                               os.path.join(TVB_ANNARCHY_DIR, "tvb_annarchy/annarchy/modules"))
MYMODULES_BLD_DIR = os.environ.get("MYMODULES_BLD_DIR",
                                   os.path.join(TVB_ANNARCHY_DIR, "tvb_annarchy/annarchy/modules_builds"))


class Config(ConfigBase):
    # WORKING DIRECTORY:
    TVB_ANNARCHY_DIR = TVB_ANNARCHY_DIR
    WORKING_DIR = WORKING_DIR
    MYMODULES_DIR = MYMODULES_DIR
    MYMODULES_BLD_DIR = MYMODULES_BLD_DIR

    # TVB_ANNARCHY_DIR_VERBOCITY = ??

    ANNARCHY_MIN_DT = 0.001

    DEFAULT_MODEL = "izhikevich"

    # Delays should be at least equal to NEST time resolution
    DEFAULT_CONNECTION = {"synapse_model": "static_synapse", "weight": 1.0, "delay": 0.01, 'receptor_type': 0,
                          "source_inds": None, "target_inds": None,
                          "conn_spec": {"allow_autapses": True, 'allow_multapses': True, 'rule': "all_to_all",
                                        "indegree": None, "outdegree": None, "N": None, "p": 0.1}}

    DEFAULT_TVB_TO_ANNARCHY_INTERFACE = "poisson_generator"
    DEFAULT_ANNARCHY_TO_TVB_INTERFACE = "spike_detector"

    # Available ANNARCHY output devices for the interface and their default properties
    ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF = {"multimeter": {"record_from": ["V_m"], "record_to": "memory"},
                                          "voltmeter": {"record_to": "memory"},
                                          "spike_detector": {"record_to": "memory"},
                                          "spike_multimeter": {'record_from': ["spike"], "record_to": "memory"}}

    ANNARCHY_INPUT_DEVICES_PARAMS_DEF = {"spike_generator": {"allow_offgrid_times": False},
                                         "poisson_generator": {},
                                         "mip_generator": {"p_copy": 0.5, "mother_seed": 0},
                                         "inhomogeneous_poisson_generator": {"allow_offgrid_times": False}}

    def __init__(self, output_base=None, separate_by_run=False):
        super(Config, self).__init__(output_base, separate_by_run)
        self.TVB_NEST_DIR = TVB_ANNARCHY_DIR
        self.WORKING_DIR = WORKING_DIR
        self.MYMODULES_DIR = MYMODULES_DIR
        self.MYMODULES_BLD_DIR = MYMODULES_BLD_DIR


CONFIGURED = Config()


def initialize_logger(name, target_folder=CONFIGURED.out.FOLDER_LOGS):
    return initialize_logger_base(name, target_folder)
