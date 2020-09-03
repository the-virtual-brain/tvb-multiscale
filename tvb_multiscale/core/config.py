# -*- coding: utf-8 -*-

import inspect
import os
from datetime import datetime

import numpy
from tvb.basic.profile import TvbProfile
from tvb.simulator.plot.config import FiguresConfig

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb.datatypes import cortex, connectivity

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

    def __init__(self, out_base=None, separate_by_run=False):
        """
        :param work_folder: Base folder where logs/figures/results should be kept
        :param separate_by_run: Set TRUE, when you want logs/results/figures to be in different files / each run
        """
        self._out_base = out_base or os.path.join(os.getcwd(), "outputs")
        self._separate_by_run = separate_by_run

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

    def __init__(self, output_base=None, separate_by_run=False):
        self.out = OutputConfig(output_base, separate_by_run)
        self.figures = FiguresConfig(output_base, separate_by_run)
        self.DEFAULT_SUBJECT = DEFAULT_SUBJECT
        self.DEFAULT_SUBJECT_PATH = DEFAULT_SUBJECT_PATH
        self.TVB_DATA_PATH = os.path.dirname(inspect.getabsfile(tvb_data))
        self.DEFAULT_CONNECTIVITY_ZIP = DEFAULT_CONNECTIVITY_ZIP

        # TODO: confirm if the following is correct:
        # We assume that all quantities of
        # current, membrane voltage or spike rate
        # that are communicated between nodes
        # are mean field quantities, i.e., averages,
        # and therefore must have the same order of magnitude,
        # at all of the region-population-single neuron levels
        # So, the default behavior is for NO scaling
        # among TVB regions (1 to 1 coupling) and
        # between TVB and Spiking Network regions (1 to all, and all to 1) coupling,
        # where average quantities will be computed in the respective interface functions
        # However, within and among Spiking Network nodes' coupling
        # will need such a weight scaling, so that a neuron receives spikes
        # weighted by the possible spikes it could have received,
        # according to its total number of connections (indegree)

    def DEFAULT_SPIKING_SYNAPTIC_WEIGHT_SCALING(self, weight, number_of_connections=1):
        if isinstance(weight, dict):
            return weight
        return weight  # / number_of_connections


CONFIGURED = Config()


def initialize_logger(name, target_folder=CONFIGURED.out.FOLDER_LOGS):
    return initialize_logger_base(name, target_folder)


# Used for nice __str__() outputs
LINE = "\n" + "-" * 100 + "\n"
