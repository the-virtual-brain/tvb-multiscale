# -*- coding: utf-8 -*-

import sys
import os
import numpy
import inspect
from datetime import datetime

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb.datatypes import cortex, connectivity

import tvb_data

from tvb_utils.log_utils import initialize_logger as initialize_logger_base
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


class GenericConfig(object):
    _module_path = os.path.dirname(__file__)

    # Identify and choose the Simulator, or data folder type to read.
    MODE_H5 = "H5"
    MODE_TVB = "TVB"


class InputConfig(object):
    _base_input = os.getcwd()

    @property
    def HEAD(self):
        if self._head_folder is not None:
            return self._head_folder

        # or else, try to find tvb_data module
        try:
            import tvb_data
            # TODO: Create a default example Head in tvb-data
            return os.path.join(os.path.dirname(tvb_data.__file__), "berlinSubjects", "QL_20120814")
        except ImportError:
            return self._base_input

    @property
    def IS_TVB_MODE(self):
        """Identify and choose the Input data type to use"""
        return self._data_mode == GenericConfig.MODE_TVB

    @property
    def RAW_DATA_FOLDER(self):
        if self._raw_data is not None:
            return self._raw_data

        return os.path.join(self._base_input, "data", "raw")

    def __init__(self, head_folder=None, raw_folder=None, data_mode=GenericConfig.MODE_TVB):
        self._head_folder = head_folder
        self._raw_data = raw_folder
        self._data_mode = data_mode


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
        if self._separate_by_run and len(ftype)>0:
            folder = folder + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
        return folder

    @property
    def _folder_logs(self):
        return self._folder("logs")

    @property
    def _folder_res(self):
        return self._folder("res")

    @property
    def _folder_figs(self):
        return self._folder("figs")

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

    @property
    def FOLDER_FIGURES(self):
        folder = self._folder_figs
        safe_makedirs(folder)
        return folder

    @property
    def FOLDER_TEMP(self):
        return os.path.join(self._out_base, "temp")


class FiguresConfig(object):
    VERY_LARGE_SIZE = (40, 20)
    VERY_LARGE_PORTRAIT = (30, 50)
    SUPER_LARGE_SIZE = (80, 40)
    LARGE_SIZE = (20, 15)
    SMALL_SIZE = (15, 10)
    NOTEBOOK_SIZE = (20, 10)
    DEFAULT_SIZE = (15, 10)
    FIG_FORMAT = 'png'
    SAVE_FLAG = True
    SHOW_FLAG = False
    MOUSE_HOOVER = False
    MATPLOTLIB_BACKEND = "Agg"  # "Qt4Agg"
    WEIGHTS_NORM_PERCENT = 99
    FONTSIZE = 10
    SMALL_FONTSIZE = 8
    LARGE_FONTSIZE = 12

    def largest_size(self):
        import sys
        if 'IPython' not in sys.modules:
            return self.LARGE_SIZE
        from IPython import get_ipython
        if getattr(get_ipython(), 'kernel', None) is not None:
            return self.NOTEBOOK_SIZE
        else:
            return self.LARGE_SIZE

    def __init__(self, out_base=None, separate_by_run=False):
        """
        :param out_base: Base folder where figures should be kept
        :param separate_by_run: Set TRUE, when you want figures to be in different files / each run
        """
        self._out_base = out_base or os.path.join(os.getcwd(), "outputs")  # or TvbProfile.current.TVB_STORAGE
        self._separate_by_run = separate_by_run

    @property
    def FOLDER_FIGURES(self):
        folder = os.path.join(self._out_base, "figs")
        if self._separate_by_run:
            folder = folder + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
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
    generic = GenericConfig()
    figures = FiguresConfig()
    calcul = CalculusConfig()

    def __init__(self, head_folder=None, raw_data_folder=None, output_base=None, separate_by_run=False):
        self.input = InputConfig(head_folder, raw_data_folder)
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
