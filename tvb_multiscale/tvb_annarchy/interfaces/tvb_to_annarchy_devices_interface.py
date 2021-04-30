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
        self.do_for_all("reset")  # Reset time to current time, for all devices
        self.Set({"spikes_times": np.ones((len(values),))})


class TVBtoANNarchyRateInterface(TVBtoANNarchyDeviceInterface):

    def set(self, values):
        self.Set({"rates": np.maximum([0], self._assert_input_size(values))})


class TVBtoANNarchyPoissonPopulationInterface(TVBtoANNarchyRateInterface):

    pass


class TVBtoANNarchyTimedArrayInterface(TVBtoANNarchyRateInterface):

    def set(self, values):
        values = self._assert_input_size(values)
        n_values = len(values)
        self.Set({"rates": np.maximum(np.array([[0]]), np.array(self._assert_input_size(values))[:, None, None]),
                  "schedule": np.array([self.annarchy_instance.get_time() + self.annarchy_instance.dt()]*n_values)})


class TVBtoANNarchyTimedPoissonPopulationInterface(TVBtoANNarchyTimedArrayInterface):

    pass


class TVBtoANNarchyHomogeneousCorrelatedSpikeTrainsInterface(TVBtoANNarchyTimedArrayInterface):

    pass


class TVBtoANNarchyDCCurrentInjectorInterface(TVBtoANNarchyDeviceInterface):

    def set(self, values):
        self.Set({"amplitude": values})


INPUT_INTERFACES_DICT = {
    "SpikeSourceArray": TVBtoANNarchySpikeSourceArrayInterface,
    "PoissonPopulation": TVBtoANNarchyPoissonPopulationInterface,
    "HomogeneousCorrelatedSpikeTrains": TVBtoANNarchyHomogeneousCorrelatedSpikeTrainsInterface,
    "TimedArray": TVBtoANNarchyTimedArrayInterface,
    "TimedPoissonPopulation": TVBtoANNarchyTimedPoissonPopulationInterface,
    "DCCurrentInjector": TVBtoANNarchyDCCurrentInjectorInterface,
}