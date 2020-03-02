# -*- coding: utf-8 -*-

from pandas import Series
import numpy as np
from tvb_multiscale.interfaces.builders.tvb_to_spikeNet_device_interface_builder import \
    TVBtoSpikeNetDeviceInterfaceBuilder
from tvb_multiscale.interfaces.builders.tvb_to_spikeNet_parameter_interface_builder import \
    TVBtoSpikeNetParameterInterfaceBuilder
from tvb_multiscale.interfaces.builders.spikeNet_to_tvb_interface_builder import SpikeNetToTVBInterfaceBuilder
from tvb_multiscale.spiking_models.network import SpikingNetwork
from tvb_multiscale.spiking_models.devices import InputDeviceDict
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import ensure_list

from tvb.simulator.simulator import Simulator

LOG = initialize_logger(__name__)


class TVBSpikeNetInterfaceBuilder(object):

    _tvb_to_spikNet_device_interface_builder = TVBtoSpikeNetDeviceInterfaceBuilder
    _tvb_to_spikeNet_parameter_interface_builder = TVBtoSpikeNetParameterInterfaceBuilder
    _spikeNet_to_tvb_interface_builder = SpikeNetToTVBInterfaceBuilder
    _input_device_dict = InputDeviceDict

    integrator = None
    connectivity = None
    tvb_nodes_ids = []
    spiking_nodes_ids = []
    exclusive_nodes = False
    spiking_network = []

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
    # We return from a Spiking Network spike_detector the ratio number_of_population_spikes / number_of_population_neurons
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

    # The Spiking Network nodes where TVB input is directed
    tvb_to_spikeNet_interfaces = []

    # The Spiking Network nodes the activity of which is transformed to TVB state variables or parameters
    spikeNet_to_tvb_interfaces = []

    def __init__(self, tvb_simulator, spiking_network, spiking_nodes_ids, exclusive_nodes=False,
                 tvb_to_spiking_interfaces=None, spiking_to_tvb_interfaces=None):
        if isinstance(spiking_network, SpikingNetwork):
            self.spiking_network = spiking_network
        else:
            raise ValueError("Input spiking_network is not a SpikingNetwork object!\n%s" % str(spiking_network))
        self.exclusive_nodes = exclusive_nodes
        if isinstance(tvb_simulator, Simulator):
            self.tvb_simulator = tvb_simulator
            self.spiking_nodes_ids = np.array(ensure_list(spiking_nodes_ids))
            self.tvb_nodes_ids = list(range(self.tvb_connectivity.weights.shape[0]))
            if self.exclusive_nodes:
                try:
                    for i_n in self.spiking_nodes_ids:
                        self.tvb_nodes_ids.remove(i_n)
                except:
                    raise ValueError("Failed to compute tvb_nodes_ids from nest_nodes_ids %s "
                                     "and TVB connectivity of size %s!"
                                     % (str(self.spiking_nodes_ids), self.tvb_connectivity.number_of_regions))
            self.tvb_nodes_ids = np.array(self.tvb_nodes_ids)
            if self.tvb_simulator.stimulus is not None:
                if np.sum(self.tvb_simulator.stimulus[:, self.spiking_nodes_ids.tolist()]):
                    raise ValueError("TVB-Spiking Network interface does not implement TVB stimulus application to Spiking Network nodes!\n"
                                     "The user has to configure such stimulus as part of the Spiking Network model, "
                                     "via appropriate Spiking Network input devices!")
        else:
            raise ValueError("Input simulator_tvb is not a Simulator object!\n%s" % str(tvb_simulator))

            # TVB <-> Spiking Network transformations' weights/funs
            # If set as weights, they will become a transformation function of
            # lambda state, regions_indices: w[regions_indices] * state[regions_indices]
            # If set as a function of lambda state: fun(state), it will become a vector function of:
            # lambda state, regions_indices: np.array([fun(state[index]) for index in regions_indices)])
            # TVB -> Spiking Network
            self.w_tvb_to_spike_rate = 1000.0  # (e.g., spike rate in NEST is in spikes/sec, assuming TVB rate is spikes/ms)
            self.w_tvb_to_current = 1000.0  # (1000.0 (nA -> pA), because I_e, and dc_generator amplitude in NEST are in pA)
            self.w_tvb_to_potential = 1.0  # assuming mV in both NEST and TVB
            # TVB <- Spiking Network
            # We return from a Spiking Network spike_detector the ratio number_of_population_spikes / number_of_population_neurons
            # for every TVB time step, which is usually a quantity in the range [0.0, 1.0],
            # as long as a neuron cannot fire twice during a TVB time step, i.e.,
            # as long as the TVB time step (usually 0.001 to 0.1 ms)
            # is smaller than the neurons' refractory time, t_ref (usually 1-2 ms)
            # For conversion to a rate, one has to do:
            # w_spikes_to_tvb = 1/tvb_dt, to get it in spikes/ms, and
            # w_spikes_to_tvb = 1000/tvb_dt, to get it in Hz
            self.w_spikes_to_tvb = 1.0
            self.w_spikes_var_to_tvb = 1.0
            # We return from a Spiking Network multimeter or voltmeter the membrane potential in mV
            self.w_potential_to_tvb = 1.0

        if spiking_to_tvb_interfaces is not None:
            self.spikeNet_to_tvb_interfaces = ensure_list(spiking_to_tvb_interfaces)
        if tvb_to_spiking_interfaces is not None:
            self.tvb_to_spikeNet_interfaces = ensure_list(tvb_to_spiking_interfaces)

    @property
    def config(self):
        return self.spiking_network.config

    @property
    def logger(self):
        return self.spiking_network.logger

    @property
    def spiking_nodes(self):
        return self.spiking_network.region_nodes

    @property
    def spikeNet_min_delay(self):
        return self.spiking_network.min_delay

    @property
    def tvb_weights(self):
        return self.tvb_simulator.connectivity.tvb_weights

    @property
    def tvb_delays(self):
        return self.tvb_simulator.connectivity.delays

    @property
    def tvb_connectivity(self):
        return self.tvb_simulator.connectivity

    @property
    def tvb_integrator(self):
        return self.tvb_simulator.integrator

    @property
    def tvb_model(self):
        return self.tvb_simulator.model

    @property
    def tvb_dt(self):
        return self.tvb_simulator.integrator.dt

    @property
    def number_of_nodes(self):
        return self.tvb_connectivity.number_of_regions

    def assert_delay(self, delay):
        return np.maximum(0.0, delay)

    def _prepare_tvb_to_spikeNet_transform_fun(self, prop, dummy):
        # This method sets tranformations of TVB state
        # to be applied before communication towards Spiking Network
        # In the simplest case, nothing happens...
        transform_fun = prop.split("w_")[1]
        if hasattr(getattr(self, prop), "__call__"):
            # If the property is already set as a function:
            return {transform_fun:
                lambda state_variable, region_nodes_indices=None: \
                    getattr(self, prop)(state_variable[region_nodes_indices])}
        else:
            # If the property is set just as a weight:
            setattr(self, prop, dummy * getattr(self, prop))
            return {transform_fun:
                lambda state_variable, region_nodes_indices, weights=getattr(self, prop): \
                    state_variable[region_nodes_indices] * weights[region_nodes_indices]}

    def _prepare_spikeNet_to_tvb_transform_fun(self, prop, dummy):
        # This method sets tranformations of Spiking Network state
        # to be applied before communication towards TVB
        # In the simplest case, nothing happens...
        transform_fun = prop.split("w_")[1]
        if hasattr(getattr(self, prop), "__call__"):
            # If the property is already set as a function:
            return {transform_fun:
                lambda spikeNet_variable, region_nodes_indices=None:
                    getattr(self, prop)(spikeNet_variable)}
        else:
            # If the property is set just as a weight:
            setattr(self, prop, dummy * getattr(self, prop))
            return {transform_fun:
                lambda spikeNet_variable, region_nodes_indices, weights=getattr(self, prop): \
                    spikeNet_variable * weights[region_nodes_indices]}

    def generate_transforms(self):
        dummy = np.ones((self.number_of_nodes, ))
        # Confirm good shape for TVB-Spiking Network interface model parameters
        # TODO: find a possible way to differentiate scalings between
        #  receiver (as in _tvb_state_to_nest_current),
        #  and sender (as in all other cases below), node indexing
        #  Also, the size doesn't have to be in all cases equal to number_of_nodes,
        #  but sometimes equal to number_of_spiking_nodes or to number_of_tvb_nodes
        transforms = {}
        for prop in ["w_tvb_to_current",
                     "w_tvb_to_potential",
                     "w_tvb_to_spike_rate"]:
            transforms.update(self._prepare_tvb_to_spikeNet_transform_fun(prop, dummy))
        for prop in ["w_spikes_to_tvb",
                     "w_spikes_var_to_tvb",
                     "w_potential_to_tvb"]:
            transforms.update(self._prepare_spikeNet_to_tvb_transform_fun(prop, dummy))
        return transforms

    def build_interface(self, tvb_spikeNet_interface):
        """
        Configure the TVB Spiking Network interface of the fine scale as well other aspects of its interface with TVB
        :return: tvb_spikeNet_interface object
        """

        tvb_spikeNet_interface.config = self.config
        # TODO: find out why the model instance is different in simulator and interface...
        tvb_spikeNet_interface.tvb_model = self.tvb_model
        tvb_spikeNet_interface.dt = self.tvb_dt
        tvb_spikeNet_interface.tvb_nodes_ids = self.tvb_nodes_ids
        tvb_spikeNet_interface.spiking_nodes_ids = self.spiking_nodes_ids
        tvb_spikeNet_interface.exclusive_nodes = self.exclusive_nodes
        tvb_spikeNet_interface.spiking_network = self.spiking_network

        tvb_spikeNet_interface.transforms = self.generate_transforms()

        tvb_spikeNet_interface.tvb_to_spikeNet_interfaces = Series({})
        # Create a list of input devices for every TVB node inside Spiking Network and connect them to the target Spiking Network nodes:
        for interface in self.tvb_to_spikeNet_interfaces:
            model = interface.get("model", None)
            if model in self._input_device_dict.keys():
                tvb_spikeNet_interface.tvb_to_spikeNet_interfaces = \
                    tvb_spikeNet_interface.tvb_to_spikeNet_interfaces.append(
                        self._tvb_to_spikNet_device_interface_builder([],
                                                                      self.spiking_network, self.spiking_nodes,
                                                                      self.spiking_nodes_ids, self.tvb_nodes_ids,
                                                                      self.tvb_model, self.tvb_weights, self.tvb_delays,
                                                                      self.tvb_connectivity.region_labels, self.tvb_dt,
                                                                      self.exclusive_nodes,
                                                                      self.config).build_interface(interface)
                                                                            )
            else:
                tvb_spikeNet_interface.tvb_to_spikeNet_interfaces = \
                    tvb_spikeNet_interface.tvb_to_spikeNet_interfaces.append(
                            self._tvb_to_spikeNet_parameter_interface_builder([],
                                                                              self.spiking_network, self.spiking_nodes,
                                                                              self.spiking_nodes_ids, self.tvb_nodes_ids,
                                                                              self.tvb_model, self.exclusive_nodes,
                                                                              self.config).build_interface(interface)
                                                                            )

        tvb_spikeNet_interface.spikeNet_to_tvb_interfaces = \
            self._spikeNet_to_tvb_interface_builder(self.spikeNet_to_tvb_interfaces,
                                                    self.spiking_network, self.spiking_nodes,
                                                    self.spiking_nodes_ids, self.tvb_nodes_ids,
                                                    self.tvb_model, self.exclusive_nodes,
                                                    self.config).build_interfaces()

        return tvb_spikeNet_interface
