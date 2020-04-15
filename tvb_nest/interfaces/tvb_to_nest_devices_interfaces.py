# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.interfaces.tvb_to_spikeNet_device_interface import TVBtoSpikeNetDeviceInterface


# Each interface has its own set(values) method, depending on the underlying device:


class TVBtoNESTDeviceInterface(TVBtoSpikeNetDeviceInterface):

    @property
    def nest_instance(self):
        return self.spiking_network.nest_instance


class TVBtoNESTDCGeneratorInterface(TVBtoNESTDeviceInterface):

    def set(self, values):
        self.Set({"amplitude": values,
                  "origin": self.nest_instance.GetKernelStatus("time"),
                  "start": self.nest_instance.GetKernelStatus("min_delay"),
                  "stop": self.dt})


class TVBtoNESTPoissonGeneratorInterface(TVBtoNESTDeviceInterface):

    def set(self, values):
        self.Set({"rate": np.maximum([0], values),
                  "origin": self.nest_instance.GetKernelStatus("time"),
                  "start": self.nest_instance.GetKernelStatus("min_delay"),
                  "stop": self.dt})


class TVBtoNESTInhomogeneousPoissonGeneratorInterface(TVBtoNESTDeviceInterface):

    def set(self, values):
        values = np.maximum([0], values).tolist()
        for i_val, val in enumerate(values):
            values[i_val] = [val]
        self.Set({"rate_times": [[self.nest_instance.GetKernelStatus("time") +
                                  self.nest_instance.GetKernelStatus("resolution")]] * len(values),
                  "rate_values": values})


class TVBtoNESTSpikeGeneratorInterface(TVBtoNESTDeviceInterface):

    def set(self, values):
        # TODO: change this so that rate corresponds to number of spikes instead of spikes' weights
        self.Set({"spikes_times": np.ones((len(values),)) *
                                  self.nest_instance.GetKernelStatus("min_delay"),
                  "origin": self.nest_instance.GetKernelStatus("time"),
                  "spike_weights": values})


class TVBtoNESTMIPGeneratorInterface(TVBtoNESTDeviceInterface):

    def set(self, values):
        self.Set({"rate": np.maximum(0, values)})


INPUT_INTERFACES_DICT = {"dc_generator": TVBtoNESTDCGeneratorInterface,
                         "poisson_generator": TVBtoNESTPoissonGeneratorInterface,
                         "inhomogeneous_poisson_generator": TVBtoNESTInhomogeneousPoissonGeneratorInterface,
                         "spike_generator": TVBtoNESTSpikeGeneratorInterface,
                         "mip_generator": TVBtoNESTMIPGeneratorInterface}