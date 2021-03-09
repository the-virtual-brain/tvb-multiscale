# -*- coding: utf-8 -*-

import inspect
import os
from datetime import datetime

import numpy
from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb.datatypes import cortex, connectivity
from tvb.simulator.plot.config import FiguresConfig

import tvb_data

from tvb_multiscale.core.utils.log_utils import initialize_logger as initialize_logger_base
from tvb.contrib.scripts.utils.file_utils import safe_makedirs

TVB_NEST_DIR = os.path.abspath(__file__).split("tvb_multiscale")[0]
WORKING_DIR = os.path.join(TVB_NEST_DIR, "tvb_multiscale/examples/outputs")
MODULES_DIR = os.path.join(TVB_NEST_DIR, "tvb_multiscale/nest/modules")
MODULES_BLDS_DIR = os.path.join(TVB_NEST_DIR, "tvb_multiscale/nest/modules_builds")

# DATA:
TVB_DATA_PATH = os.path.dirname(inspect.getabsfile(tvb_data))
DEFAULT_SUBJECT_PATH = os.path.join(TVB_DATA_PATH, "berlinSubjects", "QL_20120814")
DEFAULT_CONNECTIVITY_ZIP = os.path.join(DEFAULT_SUBJECT_PATH, "QL_20120814_Connectivity.zip")
DEFAULT_CORT_SURFACE_ZIP = "QL_20120814_Surface_Cortex.zip"
DEFAULT_CORT_REGION_MAPPING_TXT = "QL_20120814_RegionMapping.txt"
DEFAULT_EEG_LOCATIONS_TXT = "QL_20120814_EEGLocations.txt"
# Only for surface simulations for this subject:
# DEFAULT_EEG_PROJECTION_MAT = "QL_20120814_ProjectionMatrix.mat"
DEFAULT_EEG_PROJECTION_MAT = ""
try:
    DEFAULT_SUBJECT = {"connectivity": connectivity.Connectivity.from_file(DEFAULT_CONNECTIVITY_ZIP),
                       "cortex": cortex.Cortex.from_file(os.path.join(DEFAULT_SUBJECT_PATH,
                                                                      DEFAULT_CORT_SURFACE_ZIP),
                                                         region_mapping_file=os.path.join(DEFAULT_SUBJECT_PATH,
                                                                                          DEFAULT_CORT_REGION_MAPPING_TXT))
                       }
except:
    DEFAULT_SUBJECT = None


class OutputConfig(object):
    subfolder = None

    def __init__(self, out_base=None, separate_by_run=False, initialize_logger=True):
        """
        :param work_folder: Base folder where logs/figures/results should be kept
        :param separate_by_run: Set TRUE, when you want logs/results/figures to be in different files / each run
        """
        self._out_base = out_base or os.path.join(os.getcwd(), "outputs")
        self._separate_by_run = separate_by_run
        if initialize_logger:
            initialize_logger_base("logs", self.FOLDER_LOGS)

    def _folder(self, ftype=""):
        folder = os.path.join(self._out_base, ftype)
        if self._separate_by_run and len(ftype) > 0:
            folder = folder + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        return folder

    @property
    def _folder_logs(self):
        return self._folder("logs")

    @property
    def _folder_res(self):
        return self._folder("res")

    @property
    def FOLDER_LOGS(self):
        folder = self._folder_logs
        safe_makedirs(folder)
        return folder

    @property
    def FOLDER_RES(self):
        folder = self._folder_res
        safe_makedirs(folder)
        return folder


class CalculusConfig(object):
    # Normalization configuration
    WEIGHTS_NORM_PERCENT = 99

    # If True a plot will be generated to choose the number of eigenvalues to keep
    INTERACTIVE_ELBOW_POINT = False

    MIN_SINGLE_VALUE = numpy.finfo("single").min
    MAX_SINGLE_VALUE = numpy.finfo("single").max
    MAX_INT_VALUE = numpy.iinfo(numpy.int64).max
    MIN_INT_VALUE = numpy.iinfo(numpy.int64).max


class Config(object):
    calcul = CalculusConfig()

    DEFAULT_DT = 0.1
    TVB_TO_SPIKING_DT_RATIO = 2
    MIN_DELAY_RATIO = 1
    MIN_SPIKING_DT = 0.001
    MIN_DELAY = 0.001

    # Delays should be at least equal to NEST time resolution
    DEFAULT_CONNECTION = {"weight": 1.0, "delay": 1.0, 'receptor_type': 0,
                          "source_inds": None, "target_inds": None, "params": {},
                          "syn_spec": {}, "conn_spec": {}}

    def __init__(self, output_base=None, separate_by_run=False, initialize_logger=True):
        self.out = OutputConfig(output_base, separate_by_run, initialize_logger)
        self.figures = FiguresConfig(output_base, separate_by_run)
        self.DEFAULT_SUBJECT = DEFAULT_SUBJECT
        self.DEFAULT_SUBJECT_PATH = DEFAULT_SUBJECT_PATH
        self.TVB_DATA_PATH = os.path.dirname(inspect.getabsfile(tvb_data))
        self.DEFAULT_CONNECTIVITY_ZIP = DEFAULT_CONNECTIVITY_ZIP


CONFIGURED = Config(initialize_logger=False)


def initialize_logger(name="tvb-multiscale", target_folder=None, config=CONFIGURED):
    if target_folder is None:
        target_folder = config.out.FOLDER_LOGS
    return initialize_logger_base(name, target_folder)


def log_path(name, logger):
    logger.info("%s: %s" % (name, os.environ.get(name, "")))


# Used for nice __str__() outputs
LINE = "\n" + "-" * 100 + "\n"
