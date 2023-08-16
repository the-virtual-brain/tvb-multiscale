# -*- coding: utf-8 -*-

from tvb_multiscale.core.config import Config as ConfigBase
from tvb_multiscale.core.utils.log_utils import initialize_logger as initialize_logger_base


class Config(ConfigBase):
    
    DEFAULT_SPIKING_MODEL = "default_model_placeholder"  # TODO: what's the default model?

    NETPYNE_INPUT_DEVICES_PARAMS_DEF = {
        "poisson_generator": {
            "record_generated_spikes": False,
        },
    }
    NETPYNE_OUTPUT_DEVICES_PARAMS_DEF = {
        "spike_recorder": {},
        "multimeter": {},
    }

    # Delays should be at least equal to NetPyNE time resolution
    # TODO: synapse_model_placeholder? and what weight should go here?
    def DEFAULT_CONNECTION(self):
        return {"synapse_model": "synapse_model_placeholder",
                "weight": 1.0, "delay": self.DEF_SPIKING_MIN_DELAY, 'receptor_type': 0,
                "source_inds": None, "target_inds": None, "params": {},
                "conn_spec": {"allow_autapses": True, 'allow_multapses': True, 'rule': "all_to_all",
                              "indegree": None, "outdegree": None, "N": None, "p": 0.1}}


CONFIGURED = Config(initialize_logger=False)


def initialize_logger(name="tvb_netpyne", target_folder=None, config=CONFIGURED):
    if target_folder is None:
        target_folder = Config().out.FOLDER_LOGS
    return initialize_logger_base(name, target_folder)