# -*- coding: utf-8 -*-


import os
import inspect
import tvb_data
from tvb.simulator.plot.config import FiguresConfig as ConfigBase
from tvb.datatypes import cortex, connectivity
from tvb.basic.profile import TvbProfile
from tvb_scripts.utils.log_error_utils import initialize_logger

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)
initialize_logger('matplotlib')


class Config(ConfigBase):
    # NEST properties:
    NEST_MIN_DT = 0.001

    DEFAULT_MODEL = "iaf_cond_beta"  # "iaf_cond_deco2014"

    # Delays should be at least equal to NEST time resolution
    DEFAULT_CONNECTION = {"model": "static_synapse", "weight": 1.0, "delay": 0.0, 'receptor_type': 0,
                          "conn_spec": {"autapses": False, 'multapses': True, 'rule': "all_to_all",
                                        "indegree": None, "outdegree": None, "N": None, "p": 0.1}}

    DEFAULT_TVB_TO_NEST_INTERFACE = "poisson_generator"
    DEFAULT_NEST_TO_TVB_INTERFACE = "spike_detector"

    # TODO: confirm if the following is correct:
    # We assume that all quantities of
    # current, membrane voltage or spike rate
    # that are communicated between nodes
    # are mean field quantities, i.e., averages,
    # and therefore must have the same order of magnitude,
    # at all of the region-population-single neuron levels
    # So, the default behavior is for NO scaling
    # among TVB regions (1 to 1 coupling) and
    # between TVB and NEST regions (1 to all, and all to 1) coupling,
    # where average quantities will be computed in the respective interface functions
    # However, within and among NEST nodes' coupling
    # will need such a weight scaling, so that a neuron receives spikes
    # weighted by the possible spikes it could have received,
    # according to its total number of connections (indegree)
    def DEFAULT_NEST_SYNAPTIC_WEIGHT_SCALING(self, weight, number_of_connections=1):
        if isinstance(weight, dict):
            return weight
        return weight  # / number_of_connections

    # Available NEST output devices for the interface and their default properties
    NEST_OUTPUT_DEVICES_PARAMS_DEF = {"multimeter": {"withtime": True, "withgid": True, 'record_from': ["V_m"]},
                                      "voltimeter": {"withtime": True, "withgid": True},
                                      "spike_detector": {"withgid": True, "withtime": True, 'precise_times': True},
                                      "spike_multimeter": {"withtime": True, "withgid": True, 'record_from': ["spike"]}}

    # WORKING DIRECTORY
    TVB_NEST_DIR = os.path.abspath(__file__).split("tvb_nest")[0]
    MODULES_DIR = os.path.join(TVB_NEST_DIR, "tvb_nest/nest/modules")
    MODULES_BLDS_DIR = os.path.join(TVB_NEST_DIR, "tvb_nest/nest/modules_builds")
    WORKING_DIR = os.path.join(TVB_NEST_DIR, "tvb_nest/examples/outputs")
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

    def __init__(self, head_folder=WORKING_DIR, raw_data_folder=DEFAULT_SUBJECT_PATH,
                 output_base=WORKING_DIR, separate_by_run=False):
        super(Config, self).__init__(head_folder, raw_data_folder, output_base, separate_by_run)
        self.NEST_PATH = os.environ.get("NEST_INSTALL_DIR")


CONFIGURED = Config()
