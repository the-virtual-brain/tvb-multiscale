# -*- coding: utf-8 -*-

import os
from tvb_multiscale.config import Config as ConfigBase
from tvb_multiscale.config import DEFAULT_SUBJECT_PATH, initialize_logger


TVB_NEST_DIR = os.path.abspath(__file__).split("tvb_nest")[0]
WORKING_DIR = os.path.join(TVB_NEST_DIR, "tvb_nest/examples/outputs")
MODULES_DIR = os.path.join(TVB_NEST_DIR, "tvb_nest/nest/modules")
MODULES_BLDS_DIR = os.path.join(TVB_NEST_DIR, "tvb_nest/nest/modules_builds")

TVB_NEST_DIR = os.path.abspath(__file__).split("tvb_nest")[0]
WORKING_DIR = os.path.join(TVB_NEST_DIR, "tvb_nest/examples/outputs")
MODULES_DIR = os.path.join(TVB_NEST_DIR, "tvb_nest/nest/modules")
MODULES_BLDS_DIR = os.path.join(TVB_NEST_DIR, "tvb_nest/nest/modules_builds")


class Config(ConfigBase):
    # WORKING DIRECTORY:
    TVB_NEST_DIR = TVB_NEST_DIR
    WORKING_DIR = WORKING_DIR
    MODULES_DIR = MODULES_DIR
    MODULES_BLDS_DIR = MODULES_BLDS_DIR

    # NEST properties:
    NEST_MIN_DT = 0.001

    DEFAULT_MODEL = "iaf_cond_beta"  # "iaf_cond_deco2014"

    # Delays should be at least equal to NEST time resolution
    DEFAULT_CONNECTION = {"model": "static_synapse", "weight": 1.0, "delay": 0.0, 'receptor_type': 0,
                          "conn_spec": {"autapses": False, 'multapses': True, 'rule': "all_to_all",
                                        "indegree": None, "outdegree": None, "N": None, "p": 0.1}}

    DEFAULT_TVB_TO_NEST_INTERFACE = "poisson_generator"
    DEFAULT_NEST_TO_TVB_INTERFACE = "spike_detector"

    # Available NEST output devices for the interface and their default properties
    NEST_OUTPUT_DEVICES_PARAMS_DEF = {"multimeter": {"withtime": True, "withgid": True, 'record_from': ["V_m"]},
                                      "voltimeter": {"withtime": True, "withgid": True},
                                      "spike_detector": {"withgid": True, "withtime": True, 'precise_times': True},
                                      "spike_multimeter": {"withtime": True, "withgid": True, 'record_from': ["spike"]}}

    NEST_INPUT_DEVICES_PARAMS_DEF = {"poisson_generator": {},
                                     "mip_generator": {"p_copy": 0.5, "mother_seed": 0},
                                     "inhomogeneous_poisson_generator": {"allow_offgrid_times": False}}

    def __init__(self, head_folder=WORKING_DIR, raw_data_folder=DEFAULT_SUBJECT_PATH,
                 output_base=WORKING_DIR, separate_by_run=False):
        super(Config, self).__init__(head_folder, raw_data_folder, output_base, separate_by_run)
        self.NEST_PATH = os.environ["NEST_INSTALL_DIR"]
        self.PYTHON = os.environ["NEST_PYTHON_PREFIX"]
        self.DATA_DIR = os.path.join(self.NEST_PATH, "share/nest")
        self.SLI_PATH = os.path.join(self.DATA_DIR, "sli")
        self.DOC_DIR = os.path.join(self.NEST_PATH, "share/doc/nest")
        self.MODULE_PATH = os.path.join(self.NEST_PATH, "lib/nest")
        self.TVB_NEST_DIR = TVB_NEST_DIR
        self.WORKING_DIR = WORKING_DIR
        self.MODULES_DIR = MODULES_DIR
        self.MODULES_BLDS_DIR = MODULES_BLDS_DIR

        
CONFIGURED = Config()
