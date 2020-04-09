# -*- coding: utf-8 -*-

import inspect
import os

import tvb_data
from tvb.basic.profile import TvbProfile
from tvb.datatypes import cortex, connectivity
from tvb.simulator.plot.config import Config as ConfigBase

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

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
DEFAULT_SUBJECT = {"connectivity": connectivity.Connectivity.from_file(DEFAULT_CONNECTIVITY_ZIP),
                   "cortex": cortex.Cortex.from_file(os.path.join(DEFAULT_SUBJECT_PATH,
                                                                  DEFAULT_CORT_SURFACE_ZIP),
                                                     region_mapping_file=os.path.join(DEFAULT_SUBJECT_PATH,
                                                                                      DEFAULT_CORT_REGION_MAPPING_TXT))
                   }


class Config(ConfigBase):

    def __init__(self, head_folder=WORKING_DIR, raw_data_folder=DEFAULT_SUBJECT_PATH,
                 output_base=WORKING_DIR, separate_by_run=False):
        super(Config, self).__init__(head_folder, raw_data_folder, output_base, separate_by_run)
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
