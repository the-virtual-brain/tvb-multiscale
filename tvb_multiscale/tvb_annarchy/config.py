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

    ANNARCHY_MIN_DT = 0.001

    DEFAULT_MODEL = "izhikevich"

    # Delays should be at least equal to NEST time resolution
    DEFAULT_CONNECTION = {"model": "DefaultSpikingSynapse", "params": {},
                          "weight": 1.0, "delay": 0.01, 'receptor_type': "exc",
                          "source_inds": None, "target_inds": None,
                          "conn_spec": {'method': "all_to_all", "allow_self_connections": False,
                                        "force_multiple_weights": False}}

    DEFAULT_TVB_TO_ANNARCHY_INTERFACE = "poisson_population"
    DEFAULT_ANNARCHY_TO_TVB_INTERFACE = "spike_monitor"

    # Available ANNARCHY output devices for the interface and their default properties
    ANNARCHY_OUTPUT_DEVICES_PARAMS_DEF = {"spike_monitor": {"record_from": "spike"},
                                          "spike_multimeter": {"record_from": "spike"},
                                          "monitor": {"record_from": ["v", 'g_exc', 'g_inh']}}

    ANNARCHY_INPUT_DEVICES_PARAMS_DEF = {"spike_source_array": {},
                                         "poisson_population": {},
                                         "current_injector": {"equations":
                                                                  """
                                                                     r = current
                                                                  """,
                                                              "parameters": {"current": 0.0}},
                                         "dc_current_injector": {"equations":
                                                                     """
                                                                        r = current
                                                                     """,
                                                                 "parameters": {"current": 0.0}},
                                         "ac_current_injector": {"equations":
                                                                     """
                                                                        r = sin(omega*t)
                                                                     """,
                                                                 "parameters": {"f": 0.0}},
                                         }

    def __init__(self, output_base=None, separate_by_run=False):
        super(Config, self).__init__(output_base, separate_by_run)
        self.TVB_NEST_DIR = TVB_ANNARCHY_DIR
        self.WORKING_DIR = WORKING_DIR
        self.MYMODELS_DIR = MYMODELS_DIR
        self.MYMODELS_IMPORT_PATH = self.MYMODELS_DIR.split("tvb-multicale")[-1].replace("/", ".")


CONFIGURED = Config()


def initialize_logger(name, target_folder=CONFIGURED.out.FOLDER_LOGS):
    return initialize_logger_base(name, target_folder)
