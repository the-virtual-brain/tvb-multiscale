# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.interfaces.tvb_to_spikeNet_device_interface import TVBtoSpikeNetDeviceInterface


# Each interface has its own set(values) method, depending on the underlying device:


class TVBtoANNarchyDeviceInterface(TVBtoSpikeNetDeviceInterface):

    @property
    def annarchy_instance(self):
        return self.spiking_network.annarchy_instance


# class TVBtoANNarchyDCCurrentInjectorInterface(TVBtoANNarchyDeviceInterface):
#
#     def set(self, values):
#         self.Set({"amplitude": values})


class TVBtoANNarchyPoissonPopulationInterface(TVBtoANNarchyDeviceInterface):

    def set(self, values):
        self.Set({"rates": np.maximum([0], self._assert_input_size(values))})


class TVBtoANNarchyPoissonNeuronInterface(TVBtoANNarchyPoissonPopulationInterface):
   pass


class TVBtoANNarchyHomogeneousCorrelatedSpikeTrainsInterface(TVBtoANNarchyPoissonPopulationInterface):
    pass


# class TVBtoANNarchySpikeSourceArrayInterface(TVBtoANNarchyDeviceInterface):
#
#     def set(self, values):
#         self.do_for_all_devices("reset")  # Reset time to current time, for all devices
#         self.Set({"spikes_times": np.ones((len(values),))})


class TVBtoNESTMIPGeneratorInterface(TVBtoANNarchyDeviceInterface):

    def set(self, values):
        self.Set({"rate": np.maximum(0, values)})


INPUT_INTERFACES_DICT = {# "DCCurrentInjector": TVBtoANNarchyDCCurrentInjectorInterface,
                         "PoissonPopulation": TVBtoANNarchyPoissonPopulationInterface,
                         "Poisson_neuron": TVBtoANNarchyPoissonNeuronInterface,
                         "HomogeneousCorrelatedSpikeTrains": TVBtoANNarchyHomogeneousCorrelatedSpikeTrainsInterface,
                         # "SpikeSourceArray": TVBtoANNarchySpikeSourceArrayInterface
                        }