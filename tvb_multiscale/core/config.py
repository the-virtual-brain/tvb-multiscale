# -*- coding: utf-8 -*-

import inspect
import os
from datetime import datetime

import numpy
from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb.datatypes import cortex, connectivity
from tvb.simulator.plot.config import FiguresConfig as FiguresConfigTVB
from tvb.simulator.models.wilson_cowan import WilsonCowan
from tvb.simulator.coupling import Linear
from tvb.simulator.integrators import HeunDeterministic, HeunStochastic, EulerDeterministic
from tvb.simulator.noise import Additive
from tvb.simulator.monitors import Raw

from tvb.contrib.scripts.utils.file_utils import safe_makedirs

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.utils.log_utils import initialize_logger as initialize_logger_base

def find_root_dir():
    script_path = os.path.abspath(__file__)
    dir_path = os.path.split(script_path)[0] # drop file name
    # We assume that the path contains 'tvb-multiscale*' directory at certain level.
    # To find this dir, we traverse path components in reverse:
    for dir in reversed(dir_path.split(os.sep)):
        if dir.startswith('tvb-multiscale'): # found
            break
        else:
            # drop last path component and keep on traversing
            dir_path = os.path.split(dir_path)[0]
            if dir_path == '/': # reached the root, dir not found
                dir_path = None; break

    if dir_path is None:
        # if didn't succeed to find it this way, try another assumption:
        if script_path.find("tvb_multiscale/core") > 0:
            dir_path = script_path.split("tvb_multiscale/core")[0]

    return dir_path

ROOT_PATH = find_root_dir()

def _folder(base, separate_by_run=False, ftype=""):
    folder = os.path.join(base, ftype)
    if separate_by_run and len(ftype) > 0:
        folder = folder + datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
    return folder


class OutputConfig(HasTraits):

    def __init__(self, out_base=None, separate_by_run=False, initialize_logger=True):
        """
        :param work_folder: Base folder where logs/figures/results should be kept
        :param separate_by_run: Set TRUE, when you want logs/results/figures to be in different files / each run
        """
        super(OutputConfig, self).__init__()
        self._out_base = out_base or os.path.join(os.getcwd(), "outputs")
        self._separate_by_run = separate_by_run
        self.title = "OutputConfig"

        if initialize_logger:
            initialize_logger_base("logs", self.FOLDER_LOGS)

    def _folder(self, ftype=""):
        return _folder(self._out_base, self._separate_by_run, ftype)

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

    def info(self, recursive=0):
        info = super(OutputConfig, self).info(recursive=recursive)
        info.update(self._info_dict('OutputConfig as dict', self.__dict__))
        return info


class CalculusConfig(HasTraits):

    def __init__(self):
        super(CalculusConfig, self).__init__()
        self.title = "CalculusConfig"

        # Normalization configuration
        self.WEIGHTS_NORM_PERCENT = 99

        # If True a plot will be generated to choose the number of eigenvalues to keep
        self.INTERACTIVE_ELBOW_POINT = False

        self.MIN_SINGLE_VALUE = numpy.finfo("single").min
        self.MAX_SINGLE_VALUE = numpy.finfo("single").max
        self.MAX_INT_VALUE = numpy.iinfo(numpy.int64).max
        self.MIN_INT_VALUE = numpy.iinfo(numpy.int64).max

    def info(self, recursive=0):
        info = super(CalculusConfig, self).info(recursive=recursive)
        info.update(self._info_dict('CalculusConfig as dict', self.__dict__))
        return info


class FiguresConfig(FiguresConfigTVB, HasTraits):

    title = "FiguresConfig"

    def __init__(self, out_base=None, separate_by_run=False, **kwargs):
        FiguresConfigTVB.__init__(self, out_base=out_base, separate_by_run=separate_by_run)
        HasTraits.__init__(self, **kwargs)
        self.title = "FiguresConfig"

    def info(self, recursive=0):
        info = super(FiguresConfig, self).info(recursive=recursive)
        info.update(self._info_dict('FiguresConfig as dict', self.__dict__))
        return info


class Config(HasTraits):

    def __init__(self, output_base=None, separate_by_run=False, initialize_logger=True, verbosity=1):
        super(Config, self).__init__()

        self.BASEPATH = output_base
        self.VERBOSITY = verbosity

        self.out = OutputConfig(output_base, separate_by_run, initialize_logger)
        self.figures = FiguresConfig(output_base, separate_by_run)
        self.calcul = CalculusConfig()

        self.RAY_PARALLEL = False

        # DATA:
        if ROOT_PATH is not None:
            self.DATA_PATH = os.path.join(ROOT_PATH, 'examples/data')
            self.TVB_DATA_PATH = os.path.join(self.DATA_PATH, 'tvb_data')
            self.DEFAULT_SUBJECT_PATH = os.path.join(self.TVB_DATA_PATH, "berlinSubjects", "QL_20120814")
            self.DEFAULT_CONNECTIVITY_ZIP = os.path.join(self.DEFAULT_SUBJECT_PATH, "QL_20120814_Connectivity.zip")

            cort_surface = "QL_20120814_Surface_Cortex.zip"
            cort_region_mapping = "QL_20120814_RegionMapping.txt"
            region_mapping = os.path.join(self.DEFAULT_SUBJECT_PATH, cort_region_mapping)
            try:
                self.DEFAULT_SUBJECT = {
                    "connectivity": connectivity.Connectivity.from_file(self.DEFAULT_CONNECTIVITY_ZIP),
                    "cortex": cortex.Cortex.from_file(os.path.join(self.DEFAULT_SUBJECT_PATH, cort_surface),
                                                                    region_mapping_file=region_mapping)}
            except:
                self.DEFAULT_SUBJECT = None
        else:
            # ptrint warning
            self.DATA_PATH = None
            self.TVB_DATA_PATH = None
            self.DEFAULT_SUBJECT = None
            self.DEFAULT_SUBJECT_PATH = None
            self.DEFAULT_CONNECTIVITY_ZIP = None

        self.DEFAULT_DT = 0.1
        self.TVB_TO_SPIKING_DT_RATIO = 2
        self.MIN_SPIKING_DT = 0.001
        self.MIN_DELAY_RATIO = 1
        self.DEFAULT_SPIKING_MIN_DELAY = 1.0

        self.DEFAULT_TVB_MODEL = WilsonCowan
        self.DEFAULT_TVB_COUPLING_MODEL = Linear
        self.DEFAULT_DETERMINISTIC_INTEGRATOR = HeunDeterministic
        self.DEFAULT_STOCHASTIC_INTEGRATOR = HeunStochastic
        self.DEFAULT_INTEGRATOR = self.DEFAULT_STOCHASTIC_INTEGRATOR
        self.DEFAULT_TRANSFORMER_INTEGRATOR_MODEL = EulerDeterministic
        self.DEFAULT_NOISE = Additive
        self.DEFAULT_NSIG = 1e-3
        self.DEFAULT_TVB_NOISE_SEED = 42
        self.DEFAULT_MONITOR = Raw

    @property
    def output_base(self):
        return self.out._out_base

    @property
    def separate_by_run(self):
        return self.out._separate_by_run

    def _folder(self, ftype=""):
        return _folder(self.BASEPATH, self.separate_by_run, ftype)

    @property
    def _folder_config(self):
        return self._folder("config")

    @property
    def FOLDER_CONFIG(self):
        folder = self._folder_config
        safe_makedirs(folder)
        return folder

    @property
    def _folder_runtime(self):
        return self._folder("runtime")

    @property
    def FOLDER_RUNTIME(self):
        folder = self._folder_runtime
        safe_makedirs(folder)
        return folder

    def info(self, recursive=0):
        info = super(Config, self).info(recursive=recursive)
        for key, val in self.__dict__.items():
            info["config.%s" % key] = val
        return info


CONFIGURED = Config(initialize_logger=False)


def initialize_logger(name="tvb-multiscale", target_folder=None, config=CONFIGURED):
    if target_folder is None:
        target_folder = config.out.FOLDER_LOGS
    return initialize_logger_base(name, target_folder)


def log_path(name, logger):
    logger.info("%s: %s" % (name, os.environ.get(name, "")))
