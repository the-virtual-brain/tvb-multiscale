# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tvb_multiscale.config import CONFIGURED
from tvb_multiscale.spiking_models.network import SpikingNetwork
from tvb_nest.spiking_models.builders.nest_factory import load_nest
from tvb_nest.spiking_models.region_node import NESTRegionNode
from tvb_nest.spiking_models.devices import NESTDeviceSet
from tvb_scripts.utils.log_error_utils import initialize_logger


LOG = initialize_logger(__name__)


class NESTNetwork(SpikingNetwork):

    def __init__(self, nest_instance=None,
                 region_nodes=pd.Series(),
                 output_devices=pd.Series(),
                 stimulation_devices=pd.Series(),
                 nodes_min_delay=0.0,
                 config=CONFIGURED):
        if nest_instance is None:
            nest_instance = load_nest(self.config.nest, LOG)
        self.nest_instance = nest_instance
        super(NESTNetwork, self).__init__(region_nodes, output_devices, stimulation_devices, nodes_min_delay, config)

        if nodes_min_delay <= 0.0:
            self.nodes_min_delay = self.nest_instance.GetKernelStatus("min_delay")
        else:
            self.nodes_min_delay = nodes_min_delay

        if isinstance(self.region_nodes, pd.Series):
            if len(self.region_nodes) > 0 and \
                    np.any([not isinstance(node, NESTRegionNode) for node in self.region_nodes]):
                raise ValueError("Input region_nodes is neither a SpikingRegionNode "
                                 "nor a pandas.Series of SpikingRegionNode objects!: \n %s" %
                                 str(self.region_nodes))

        if isinstance(self.output_devices, pd.Series):
            if len(self.output_devices) > 0 \
                    and np.any([not isinstance(dev, NESTDeviceSet) for dev in self.output_devices]):
                raise ValueError("Input output_devices is not a pandas.Series of output DeviceSet objects!:\n %s" %
                                 str(self.output_devices))

        if isinstance(self.stimulation_devices, pd.Series):
            if len(self.stimulation_devices) > 0 and \
                    np.any([not isinstance(dev, NESTDeviceSet) for dev in self.stimulation_devices]):
                raise ValueError("Input input_devices is not a pandas.Series of input DeviceSet objects!:\n %s" %
                                 str(self.stimulation_devices))

        LOG.info("%s created!" % self.__class__)

    @property
    def min_delay(self):
        return self.nest_instance.GetKernelStatus("min_delay")

    def configure(self, *args, **kwargs):
        self.nest_instance.Prepare(*args, **kwargs)

    def Run(self, *args, **kwargs):
        self.nest_instance.Run(*args, **kwargs)
