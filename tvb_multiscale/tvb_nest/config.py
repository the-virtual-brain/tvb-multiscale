# -*- coding: utf-8 -*-

import os

from tvb_multiscale.core.config import Config as ConfigBase
from tvb_multiscale.core.utils.log_utils import initialize_logger as initialize_logger_base

TVB_NEST_DIR = os.path.abspath(__file__).split("tvb_nest")[0]
WORKING_DIR = os.environ.get("WORKING_DIR", os.getcwd())
MYMODULES_DIR = os.environ.get("MYMODULES_DIR",
                               os.path.join(TVB_NEST_DIR, "tvb_nest/nest/modules"))
MYMODULES_BLD_DIR = os.environ.get("MYMODULES_BLD_DIR",
                                   os.path.join(TVB_NEST_DIR, "tvb_nest/nest/modules_builds"))


class Config(ConfigBase):
    # WORKING DIRECTORY:
    TVB_NEST_DIR = TVB_NEST_DIR
    WORKING_DIR = WORKING_DIR
    MYMODULES_DIR = MYMODULES_DIR
    MYMODULES_BLD_DIR = MYMODULES_BLD_DIR

    # NEST properties:
    # M_ALL=0,  display all messages
    # M_DEBUG=5,  display debugging messages and above
    # M_STATUS=7,  display status messages and above
    # M_INFO=10, display information messages and above
    # M_DEPRECATED=18, display deprecation warnings and above
    # M_WARNING=20, display warning messages and above
    # M_ERROR=30, display error messages and above
    # M_FATAL=40, display failure messages and above
    # M_QUIET=100, suppress all messages
    NEST_VERBOCITY = 40

    NEST_PRINT_TIME = False

    NEST_MIN_DT = 0.001

    DEFAULT_MODEL = "iaf_cond_alpha"  # "iaf_cond_deco2014"

    # Delays should be at least equal to NEST time resolution
    DEFAULT_CONNECTION = {"synapse_model": "static_synapse", "weight": 1.0, "delay": 0.01, 'receptor_type': 0,
                          "source_inds": None, "target_inds": None,
                          "conn_spec": {"allow_autapses": True, 'allow_multapses': True, 'rule': "all_to_all",
                                        "indegree": None, "outdegree": None, "N": None, "p": 0.1}}

    DEFAULT_TVB_TO_NEST_INTERFACE = "poisson_generator"
    DEFAULT_NEST_TO_TVB_INTERFACE = "spike_detector"

    # Available NEST output devices for the interface and their default properties
    NEST_OUTPUT_DEVICES_PARAMS_DEF = {"multimeter": {"record_from": ["V_m"], "record_to": "memory"},
                                      "voltimeter": {"record_to": "memory"},
                                      "spike_detector": {"record_to": "memory"},
                                      "spike_multimeter": {'record_from': ["spike"], "record_to": "memory"}}

    NEST_INPUT_DEVICES_PARAMS_DEF = {"spike_generator": {"allow_offgrid_times": False},
                                     "poisson_generator": {},
                                     "mip_generator": {"p_copy": 0.5, "mother_seed": 0},
                                     "inhomogeneous_poisson_generator": {"allow_offgrid_times": False}}

    def __init__(self, output_base=None, separate_by_run=False):
        super(Config, self).__init__(output_base, separate_by_run)
        self.NEST_PATH = os.environ["NEST_INSTALL_DIR"]
        self.PYTHON = os.environ["NEST_PYTHON_PREFIX"]
        self.DATA_DIR = os.path.join(self.NEST_PATH, "share/nest")
        self.SLI_PATH = os.path.join(self.DATA_DIR, "sli")
        self.DOC_DIR = os.path.join(self.NEST_PATH, "share/doc/nest")
        self.MODULE_PATH = os.path.join(self.NEST_PATH, "lib/nest")
        self.TVB_NEST_DIR = TVB_NEST_DIR
        self.WORKING_DIR = WORKING_DIR
        self.MYMODULES_DIR = MYMODULES_DIR
        self.MYMODULES_BLD_DIR = MYMODULES_BLD_DIR


CONFIGURED = Config()


def initialize_logger(name, target_folder=CONFIGURED.out.FOLDER_LOGS):
    return initialize_logger_base(name, target_folder)
