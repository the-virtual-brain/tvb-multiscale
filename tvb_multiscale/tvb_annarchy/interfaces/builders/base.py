# -*- coding: utf-8 -*-
import numpy as np

from tvb_multiscale.tvb_annarchy.annarchy_models.devices import ANNarchyInputDeviceDict
from tvb_multiscale.tvb_annarchy.interfaces.builders.tvb_to_annarchy_devices_interface_builder import \
    TVBtoANNarchyDeviceInterfaceBuilder
from tvb_multiscale.tvb_annarchy.interfaces.builders.tvb_to_annarchy_parameter_interface_builder import \
    TVBtoANNarchyParameterInterfaceBuilder
from tvb_multiscale.tvb_annarchy.interfaces.builders.annarchy_to_tvb_interface_builder import \
    ANNarchytoTVBInterfaceBuilder

from tvb_multiscale.core.tvb.interfaces.builders import TVBSpikeNetInterfaceBuilder


class TVBANNarchyInterfaceBuilder(TVBSpikeNetInterfaceBuilder):
    _tvb_to_spikNet_device_interface_builder = TVBtoANNarchyDeviceInterfaceBuilder
    _tvb_to_spikeNet_parameter_interface_builder = TVBtoANNarchyParameterInterfaceBuilder
    _spikeNet_to_tvb_interface_builder = ANNarchytoTVBInterfaceBuilder
    _input_device_dict = ANNarchyInputDeviceDict

    # TVB <-> Spiking Network transformations' weights/funs
    # If set as weights, they will become a transformation function of
    # lambda state, regions_indices: w[regions_indices] * state[regions_indices]
    # If set as a function of lambda state: fun(state), it will become a vector function of:
    # lambda state, regions_indices: np.array([fun(state[index]) for index in regions_indices)])
    # TVB -> Spiking Network
    w_tvb_to_spike_rate = 1000.0  # (e.g., spike rate in NEST is in spikes/sec, assuming TVB rate is spikes/ms)
    w_tvb_to_current = 1000.0  # (1000.0 (nA -> pA), because I_e, and dc_generator amplitude in NEST are in pA)
    w_tvb_to_potential = 1.0  # assuming mV in both Spiking Network and TVB
    # TVB <- Spiking Network
    # We return from a Spiking Network spike_monitor the ratio number_of_population_spikes / number_of_population_neurons
    # for every TVB time step, which is usually a quantity in the range [0.0, 1.0],
    # as long as a neuron cannot fire twice during a TVB time step, i.e.,
    # as long as the TVB time step (usually 0.001 to 0.1 ms)
    # is smaller than the neurons' refractory time, t_ref (usually 1-2 ms)
    # For conversion to a rate, one has to do:
    # w_spikes_to_tvb = 1/tvb_dt, to get it in spikes/ms, and
    # w_spikes_to_tvb = 1000/tvb_dt, to get it in Hz
    w_spikes_to_tvb = 1.0
    w_spikes_var_to_tvb = 1.0
    # We return from a Spiking Network multimeter or voltmeter the membrane potential in mV
    w_potential_to_tvb = 1.0

    @property
    def annarchy_instance(self):
        return self.spiking_network.annarchy_instance

    @property
    def config(self):
        return self.spiking_network.config

    @property
    def spikeNet_min_delay(self):
        return self.annarchy_instance.dt()

    @property
    def annarchy_min_delay(self):
        return self.annarchy_instance.dt()

    def assert_delay(self, delay):
        return np.maximum(self.spikeNet_min_delay, delay)
