# -*- coding: utf-8 -*-

from tvb_nest.config import CONFIGURED, initialize_logger
from tvb_nest.nest_models.builders.nest_factory import load_nest
from tvb_nest.nest_models.brain import NESTBrain

from tvb_multiscale.spiking_models.network import SpikingNetwork
from tvb_multiscale.spiking_models.devices import DeviceSet


LOG = initialize_logger(__name__)


class NESTNetwork(SpikingNetwork):

    nest_instance = None

    def __init__(self, nest_instance=None,
                 brain_nodes=NESTBrain,
                 output_devices=DeviceSet(),
                 input_devices=DeviceSet(),
                 config=CONFIGURED):
        if nest_instance is None:
            nest_instance = load_nest(self.config, LOG)
        self.nest_instance = nest_instance
        super(NESTNetwork, self).__init__(brain_nodes, output_devices, input_devices, config)

    @property
    def spiking_simulator_module(self):
        return self.nest_instance

    @property
    def min_delay(self):
        return self.nest_instance.GetKernelStatus("min_delay")

    def configure(self, *args, **kwargs):
        self.nest_instance.Prepare(*args, **kwargs)

    def Run(self, *args, **kwargs):
        self.nest_instance.Run(*args, **kwargs)
