# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.interfaces.tvb_to_spikeNet_device_interface import TVBtoSpikeNetDeviceInterface


# Each interface has its own set(values) method, depending on the underlying device:


class TVBtoANNarchyDeviceInterface(TVBtoSpikeNetDeviceInterface):

    @property
    def annarchy_instance(self):
        return self.spiking_network.annarchy_instance


class TVBtoANNarchySpikeSourceArrayInterface(TVBtoANNarchyDeviceInterface):

    def set(self, values):
        self.do_for_all_devices("reset")  # Reset time to current time, for all devices
        self.Set({"spikes_times": np.ones((len(values),))})


class TVBtoANNarchyRateInterface(TVBtoANNarchyDeviceInterface):

    def set(self, values):
        self.Set({"rates": np.maximum([0], self._assert_input_size(values))})


class TVBtoANNarchyPoissonPopulationInterface(TVBtoANNarchyRateInterface):

    def set(self, values):
        self.Set({"rates": np.maximum([0], self._assert_input_size(values))})


class TVBtoANNarchyPoissonNeuronInterface(TVBtoANNarchyRateInterface):

    pass


class TVBtoANNarchyHomogeneousCorrelatedSpikeTrainsInterface(TVBtoANNarchyRateInterface):

    pass


class TVBtoANNarchyTimedArrayInterface(TVBtoANNarchyDeviceInterface):

    def set(self, values):
        vals = self._assert_input_size(values)[:, None, None]
        print(vals)
        self.Set({"rates": np.maximum(np.array([[0]]), vals)})


class TVBtoANNarchyTimedArrayPoissonPopulationInterface(TVBtoANNarchyTimedArrayInterface):
    pass


class TVBtoANNarchyTimedArrayPoissonNeuronInterface(TVBtoANNarchyTimedArrayInterface):
    pass


class TVBtoANNarchyTimedArrayHomogeneousCorrelatedSpikeTrainsInterface(TVBtoANNarchyTimedArrayInterface):
    pass


# class TVBtoANNarchyDCCurrentInjectorInterface(TVBtoANNarchyDeviceInterface):
#
#     def set(self, values):
#         self.Set({"amplitude": values})


INPUT_INTERFACES_DICT = {
    "SpikeSourceArray": TVBtoANNarchySpikeSourceArrayInterface,
    "PoissonPopulation": TVBtoANNarchyPoissonPopulationInterface,
    "Poisson_neuron": TVBtoANNarchyPoissonNeuronInterface,
    "HomogeneousCorrelatedSpikeTrains": TVBtoANNarchyHomogeneousCorrelatedSpikeTrainsInterface,
    "TimedArray": TVBtoANNarchyTimedArrayInterface,
    "TimedArrayPoissonPopulation": TVBtoANNarchyTimedArrayPoissonPopulationInterface,
    "TimedArrayPoisson_neuron": TVBtoANNarchyTimedArrayPoissonNeuronInterface,
    "TimedArrayHomogeneousCorrelatedSpikeTrains": TVBtoANNarchyTimedArrayHomogeneousCorrelatedSpikeTrainsInterface
    # "DCCurrentInjector": TVBtoANNarchyDCCurrentInjectorInterface,
}