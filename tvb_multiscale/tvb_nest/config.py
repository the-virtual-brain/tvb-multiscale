# -*- coding: utf-8 -*-

import sys
import os
import warnings

from tvb_multiscale.core.config import Config as ConfigBase, log_path
from tvb_multiscale.core.utils.log_utils import initialize_logger as initialize_logger_base
from tvb_multiscale.core.utils.file_utils import get_tvb_nest_path_from_abs_filepath

HOME_DIR = os.environ.get("HOME")
TVB_NEST_DIR = get_tvb_nest_path_from_abs_filepath(os.path.abspath(__file__))
WORKING_DIR = os.environ.get("WORKING_DIR", os.getcwd())
MYMODULES_DIR = os.environ.get("MYMODULES_DIR",
                               os.path.join(TVB_NEST_DIR, "nest", "modules"))
MYMODULES_BLD_DIR = os.environ.get("MYMODULES_BLD_DIR",
                                   os.path.join(HOME_DIR, "build", "nest_modules_builds"))

DEFAULT_NEST_INSTALL_DIR = os.path.join(HOME_DIR, "build/nest")


class Config(ConfigBase):

    def __init__(self, output_base=None, separate_by_run=False, initialize_logger=True, verbosity=1):
        super(Config, self).__init__(output_base, separate_by_run, initialize_logger, verbosity)
        try:
            self._NEST_PATH = os.environ["NEST_INSTALL_DIR"]
        except Exception as e:
            warnings.warn("NEST_INSTALL_DIR nor set!\n%s" % str(e))
            self._NEST_PATH = ""
        try:
            self.PYTHON = os.environ["NEST_PYTHON_PREFIX"]
        except Exception as e:
            warnings.warn("NEST_PYTHON_PREFIX nor set!\n%s" % str(e))
            self.PYTHON = ""

        self.NEST_BIN_PATH = os.path.join(self._NEST_PATH, "bin")
        self.NEST_CONFIG_PATH = os.path.join(self.NEST_BIN_PATH, "nest-config")
        self.NEST_INCLUDE_PATH = os.path.join(self._NEST_PATH, "include")
        self.NEST_DATA_DIR = os.path.join(self._NEST_PATH, "share/nest")
        self.NEST_SLI_PATH = os.path.join(self.NEST_DATA_DIR, "sli")
        self.NEST_DOC_DIR = os.path.join(self._NEST_PATH, "share/doc/nest")
        self.NEST_MODULE_PATH = os.path.join(self._NEST_PATH, "lib/nest")

        # WORKING DIRECTORY:
        self.TVB_NEST_DIR = TVB_NEST_DIR
        self.WORKING_DIR = WORKING_DIR
        self.MYMODULES_DIR = MYMODULES_DIR
        self.MYMODULES_BLD_DIR = MYMODULES_BLD_DIR
        self.RECORDINGS_DIR = os.path.join(self.out.FOLDER_RES, "nest_recordings")

        self.NEST_MASTER_SEED = 143202461  # Seems to be the default for NEST 3.5

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
        self.NEST_VERBOCITY = 40

        self.DEFAULT_NUM_PROCS = 1
        self.DEFAULT_LOCAL_NUM_THREADS = 4

        self.DEFAULT_SPIKING_MODEL = "iaf_cond_alpha"

        # Delays should be at least equal to NEST time resolution
        self.DEFAULT_SYNAPSE = "static_synapse"

        self.LOCK_MIN_DELAY = False

        self.DEFAULT_TVB_TO_NEST_INTERFACE = "inhomogeneous_poisson_generator"
        self.DEFAULT_NEST_TO_TVB_INTERFACE = "spike_recorder"

        self.DEFAULT_DEVICE_RECORD_TO = "memory"  # "ascii"  # "memory"

        self.NEST_INPUT_DEVICES_PARAMS_DEF = {
            "spike_generator": {"allow_offgrid_times": True}, # "shift_now_spikes": True
            "poisson_generator": {},
            "mip_generator": {"p_copy": 0.5, "mother_seed": 0},
            "inhomogeneous_poisson_generator": {"allow_offgrid_times": True},
            "parrot_spike_generator": {"allow_offgrid_times": True},  # "shift_now_spikes": True
            "parrot_poisson_generator": {},
            "parrot_mip_generator": {"p_copy": 0.5, "mother_seed": 0},
            "parrot_inhomogeneous_poisson_generator": {"allow_offgrid_times": True}
                                             }

    @property
    def TOTAL_NUM_VIRTUAL_PROCS(self):
        return self.DEFAULT_NUM_PROCS * self.DEFAULT_LOCAL_NUM_THREADS

    @property
    def DEFAULT_NEST_KERNEL_CONFIG(self):
        return {"data_path": self.RECORDINGS_DIR, "overwrite_files": True,
                "local_num_threads": self.DEFAULT_LOCAL_NUM_THREADS,
                'rng_seed': self.NEST_MASTER_SEED}

    @property
    def DEFAULT_CONNECTION(self):
        return {"synapse_model": self.DEFAULT_SYNAPSE, "weight": 1.0,
                "delay": self.DEFAULT_SPIKING_MIN_DELAY, 'receptor_type': 0,
                "source_inds": None, "target_inds": None, "params": {},
                "syn_spec": {"synapse_model": self.DEFAULT_SYNAPSE, "params": {}},
                "conn_spec": {"allow_autapses": True, 'allow_multapses': True, 'rule': "all_to_all",
                              "indegree": None, "outdegree": None, "N": None, "p": 0.1}}

    @property
    def NEST_OUTPUT_DEVICES_PARAMS_DEF(self):
        # Available NEST output devices for the interface and their default properties
        return {"multimeter": {"record_from": ["V_m"], "record_to": self.DEFAULT_DEVICE_RECORD_TO},
                "voltmeter": {"record_to": self.DEFAULT_DEVICE_RECORD_TO},
                "spike_recorder": {"record_to": self.DEFAULT_DEVICE_RECORD_TO},
                "spike_multimeter": {'record_from': ["spike"], "record_to": self.DEFAULT_DEVICE_RECORD_TO}}

    @property
    def NEST_PATH(self):
        if os.path.isdir(self._NEST_PATH):
            return self._NEST_PATH
        return ""

    def configure_nest_path(self, logger=None):
            if logger is None:
                logger = initialize_logger_base(__name__, self.out.FOLDER_LOGS)
            if os.path.isdir(self._NEST_PATH):
                logger.info("Configuring NEST path...")
                nest_path = self._NEST_PATH
                os.environ['NEST_INSTALL_DIR'] = nest_path
                log_path('NEST_INSTALL_DIR', logger)
                os.environ['NEST_DATA_DIR'] = os.path.join(nest_path, "share/nest")
                log_path('NEST_DATA_DIR', logger)
                os.environ['NEST_DOC_DIR'] = os.path.join(nest_path, "share/doc/nest")
                log_path('NEST_DOC_DIR', logger)
                os.environ['NEST_MODULE_PATH'] = os.path.join(nest_path, "lib/nest")
                log_path('NEST_MODULE_PATH', logger)
                os.environ['PATH'] = os.path.join(nest_path, "bin") + ":" + os.environ['PATH']
                log_path('PATH', logger)
                LD_LIBRARY_PATH = os.environ.get('LD_LIBRARY_PATH', '')
                if len(LD_LIBRARY_PATH) > 0:
                    LD_LIBRARY_PATH = ":" + LD_LIBRARY_PATH
                os.environ['LD_LIBRARY_PATH'] = os.environ['NEST_MODULE_PATH'] + LD_LIBRARY_PATH
                log_path('LD_LIBRARY_PATH', logger)
                os.environ['SLI_PATH'] = os.path.join(os.environ['NEST_DATA_DIR'], "sli")
                log_path('SLI_PATH', logger)
                os.environ['NEST_PYTHON_PREFIX'] = self.PYTHON
                log_path('NEST_PYTHON_PREFIX', logger)
                sys.path.insert(0, os.environ['NEST_PYTHON_PREFIX'])
                logger.info("%s: %s" % ("system path", sys.path))
            else:
                warnings.warn("NEST_PATH is not set! Configuring NEST_PATH is not possible!\n%s")


CONFIGURED = Config(initialize_logger=False)
CONFIGURED.configure_nest_path()


def initialize_logger(name="tvb_nest", target_folder=None, config=CONFIGURED):
    if target_folder is None:
        target_folder = config.out.FOLDER_LOGS
    return initialize_logger_base(name, target_folder)
