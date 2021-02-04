# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import os

import numpy as np

from tvb.basic.neotraits._attr import Attr
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.interfaces.builder import InterfaceBuilder
from tvb_multiscale.core.interfaces.io import RemoteSenders, RemoteReceivers
from tvb_multiscale.core.tvb.interfaces.transformers import \
    TVBTransformers, TVBRatesToSpikesTransformers, TVBSpikesToRatesElephantRate
from tvb_multiscale.core.tvb.interfaces.interfaces import TVBOutputInterface, TVBInputInterfaces, \
    TVBSenderInterface, TVBReceiverInterface, TVBTransformerSenderInterface, TVBReceiverTransformerInterface
from tvb_multiscale.core.tvb.cosimulator import CoSimulator
from tvb_multiscale.core.tvb.simulator_serialization import serialize_tvb_simulator, load_serial_tvb_simulator



LOG = initialize_logger(__name__)


class TVBInterfaceBuilder(InterfaceBuilder):
    __metaclass__ = ABCMeta

    tvb_simulator = Attr(label="TVB simulator",
                         doc="""The instance of TVB simulator""",
                         field_type=CoSimulator,
                         required=True)

    exclusive_nodes = Attr(label="TVB simulator",
                           doc="""The instance of TVB simulator""",
                           field_type=bool,
                           default=True,
                           required=True)

    @property
    def tvb_dt(self):
        return self.tvb_simulator.integrator.dt

    @property
    def tvb_model(self):
        return self.tvb_simulator.model

    @property
    def tvb_model_state_variables(self):
        return self.tvb_simulator.model.state_variables

    @property
    def tvb_model_cvar(self):
        return self.tvb_simulator.model.cvar

    @property
    def number_of_regions(self):
        return self.tvb_simulator.connectivity.number_of_regions

    @property
    def region_labels(self):
        return self.tvb_simulator.connectivity.region_labels

    @property
    def tvb_weights(self):
        return self.tvb_simulator.connectivity.weights

    @property
    def tvb_delays(self):
        return self.tvb_simulator.connectivity.delays

    def _proxy_inds(self, interfaces):
        return np.unique(self._only_inds(interfaces, "proxy", self.region_labels))

    @property
    def out_proxy_inds(self):
        return self._proxy_inds(self.output_interfaces)

    @property
    def in_proxy_inds(self):
        return self._proxy_inds(self.input_interfaces)

    def _voi_inds(self, interfaces):
        return np.unique(self._only_inds(interfaces, "voi", self.tvb_model_state_variables))

    @property
    def out_voi_inds(self):
        return self._voi_inds(self.output_interfaces)

    @property
    def in_voi_inds(self):
        return self._voi_inds(self.input_interfaces)

    @property
    def number_of_out_voi(self):
        return len(self.out_voi_inds)

    @property
    def number_of_in_voi(self):
        return len(self.in_voi_inds)

    def region_label_to_ind(self, labels):
        return self._label_to_ind(labels, self.region_labels)

    @property
    def out_proxy_labels(self):
        return self.region_labels[self.out_proxy_inds]

    @property
    def in_proxy_labels(self):
        return self.region_labels[self.in_proxy_inds]

    def voi_label_to_ind(self, voi):
        return self._label_to_ind(voi, self.tvb_simulator.model.state_variables)

    @property
    def out_voi_labels(self):
        return self.tvb_model_state_variables[self.out_voi_inds]

    @property
    def in_voi_labels(self):
        return self.tvb_model_state_variables[self.in_voi_inds]

    def configure(self):
        super(TVBInterfaceBuilder, self).configure()
        if len(self.output_interfaces):
            assert self.out_voi_labels in self.tvb_model_state_variables
            assert self.out_proxy_labels in self.region_labels
        if len(self.input_interfaces):
            assert self.in_voi_labels in self.tvb_model_state_variables
            assert self.in_proxy_labels in self.region_labels

    @abstractmethod
    def build(self):
        pass


class TVBRemoteInterfaceBuilder(TVBInterfaceBuilder):

    _remote_senders_types = tuple([val.value for val in RemoteSenders.__members__.values()])
    _remote_receivers_types = tuple([val.value for val in RemoteReceviers.__members__.values()])

    def configure(self):
        super(TVBRemoteInterfaceBuilder, self).configure()
        self._assert_output_interfaces_component_config(self._remote_senders_types, "sender")
        self._assert_input_interfaces_component_config(self._remote_receivers_types, "receiver")

    def build(self):
        pass


class TVBTransfomerInterfaceBuilder(TVBRemoteInterfaceBuilder):

    _tvb_transformers_types = tuple([val.value for val in TVBTransformers.__members__.values()])

    def configure(self):
        super(TVBTransfomerInterfaceBuilder, self).configure()
        self._assert_output_interfaces_component_config(self._tvb_transformers_types, "output_transformer")
        self._assert_input_interfaces_component_config(self._tvb_transformers_types, "input_transformer")

    def build(self):
        pass


class TVBOutputTransfomerInterfaceBuilder(TVBRemoteInterfaceBuilder):

    _tvb_transformers_types = tuple([val.value for val in TVBTransformers.__members__.values()])

    def configure(self):
        super(TVBTransfomerInterfaceBuilder, self).configure()
        self._assert_output_interfaces_component_config(self._tvb_transformers_types, "output_transformer")

    def build(self):
        pass


class TVBInputTransfomerInterfaceBuilder(TVBRemoteInterfaceBuilder):

    _tvb_transformers_types = tuple([val.value for val in TVBTransformers.__members__.values()])

    def configure(self):
        super(TVBTransfomerInterfaceBuilder, self).configure()
        self._assert_input_interfaces_component_config(self._tvb_transformers_types, "input_transformer")

    def build(self):
        pass


# class TVBInterfaceBuilder(InterfaceBuilder):
#
#     def __init__(self, tvb_serial_sim, spiking_network, spiking_nodes_ids, exclusive_nodes=False,
#                  tvb_to_spiking_interfaces=None, spiking_to_tvb_interfaces=None):
#         if isinstance(spiking_network, SpikingNetwork):
#             self.spiking_network = spiking_network
#         else:
#             raise ValueError("Input spiking_network is not a SpikingNetwork object!\n%s" % str(spiking_network))
#         self.exclusive_nodes = exclusive_nodes
#         if isinstance(tvb_serial_sim, os.PathLike):
#             self.tvb_serial_sim = load_serial_tvb_simulator(tvb_serial_sim)
#         elif not isinstance(tvb_serial_sim, dict):
#             self.tvb_serial_sim = serialize_tvb_simulator(tvb_serial_sim)
#         self.spiking_nodes_ids = np.array(ensure_list(spiking_nodes_ids))
#         self.tvb_nodes_ids = list(range(self.number_of_regions))
#         if self.exclusive_nodes:
#             try:
#                 for i_n in self.spiking_nodes_ids:
#                     self.tvb_nodes_ids.remove(i_n)
#             except:
#                 raise ValueError("Failed to compute tvb_nodes_ids from nest_nodes_ids %s "
#                                  "and TVB connectivity of size %s!"
#                                  % (str(self.spiking_nodes_ids), self.number_of_regions))
#             self.tvb_nodes_ids = np.array(self.tvb_nodes_ids)
#
#         # NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!
#         self.global_coupling_scaling = self.tvb_serial_sim["coupling.a"][0].item()
#
#         # TVB <-> Spiking Network transformations' weights/funs
#         # If set as weights, they will become a transformation function of
#         # lambda state, regions_indices: w[regions_indices] * state[regions_indices]
#         # If set as a function of lambda state: fun(state), it will become a vector function of:
#         # lambda state, regions_indices: np.array([fun(state[index]) for index in regions_indices)])
#         # TVB -> Spiking Network
#         self.w_tvb_to_spike_rate = 1000.0  # (e.g., spike rate in NEST is in spikes/sec, assuming TVB rate is spikes/ms)
#         self.w_tvb_to_current = 1000.0  # (1000.0 (nA -> pA), because I_e, and dc_generator amplitude in NEST are in pA)
#         self.w_tvb_to_potential = 1.0  # assuming mV in both NEST and TVB
#         # TVB <- Spiking Network
#         # We return from a Spiking Network spike_detector
#         # the ratio number_of_population_spikes / number_of_population_neurons
#         # for every TVB time step, which is usually a quantity in the range [0.0, 1.0],
#         # as long as a neuron cannot fire twice during a TVB time step, i.e.,
#         # as long as the TVB time step (usually 0.001 to 0.1 ms)
#         # is smaller than the neurons' refractory time, t_ref (usually 1-2 ms)
#         # For conversion to a rate, one has to do:
#         # w_spikes_to_tvb = 1/tvb_dt, to get it in spikes/ms, and
#         # w_spikes_to_tvb = 1000/tvb_dt, to get it in Hz
#         self.w_spikes_to_tvb = 1.0
#         self.w_spikes_var_to_tvb = 1.0
#         # We return from a Spiking Network multimeter or voltmeter the membrane potential in mV
#         self.w_potential_to_tvb = 1.0
#
#         if spiking_to_tvb_interfaces is not None:
#             self.spikeNet_to_tvb_interfaces = ensure_list(spiking_to_tvb_interfaces)
#         if tvb_to_spiking_interfaces is not None:
#             self.tvb_to_spikeNet_interfaces = ensure_list(tvb_to_spiking_interfaces)
#
#     @property
#     def config(self):
#         return self.spiking_network.config
#
#     @property
#     def tvb_dt(self):
#         return self.tvb_serial_sim["integrator.dt"]
#
#     @property
#     def tvb_model(self):
#         return self.tvb_serial_sim["model"]
#
#     @property
#     def tvb_model_state_variables(self):
#         return self.tvb_serial_sim["model.state_variables"]
#
#     @property
#     def tvb_model_cvar(self):
#         return self.tvb_serial_sim["model.cvar"]
#
#     @property
#     def number_of_regions(self):
#         return self.tvb_serial_sim["connectivity.number_of_regions"]
#
#     @property
#     def region_labels(self):
#         return self.tvb_serial_sim["connectivity.region_labels"]
#
#     @property
#     def tvb_weights(self):
#         return self.tvb_serial_sim["connectivity.weights"]
#
#     @property
#     def tvb_delays(self):
#         return self.tvb_serial_sim["connectivity.delays"]
#
#     @property
#     def spiking_nodes(self):
#         return self.spiking_network.brain_regions
#
#     @property
#     def spikeNet_min_delay(self):
#         return self.spiking_network.min_delay
#
#
#
#     def _prepare_tvb_to_spikeNet_transform_fun(self, prop, dummy):
#         # This method sets tranformations of TVB state
#         # to be applied before communication towards Spiking Network
#         # In the simplest case, nothing happens...
#         transform_fun = prop.split("w_")[1]
#         if hasattr(getattr(self, prop), "__call__"):
#             # If the property is already set as a function:
#             return {transform_fun:
#                         lambda state_variable, region_nodes_indices=None:
#                             getattr(self, prop)(state_variable[region_nodes_indices])}
#         else:
#             # If the property is set just as a weight:
#             setattr(self, prop, dummy * getattr(self, prop))
#             return {transform_fun:
#                         lambda state_variable, region_nodes_indices, weights=getattr(self, prop):
#                             state_variable[region_nodes_indices] * weights[region_nodes_indices]}
#
#     def _prepare_spikeNet_to_tvb_transform_fun(self, prop, dummy):
#         # This method sets tranformations of Spiking Network state
#         # to be applied before communication towards TVB
#         # In the simplest case, nothing happens...
#         transform_fun = prop.split("w_")[1]
#         if hasattr(getattr(self, prop), "__call__"):
#             # If the property is already set as a function:
#             return {transform_fun:
#                         lambda spikeNet_variable, region_nodes_indices=None:
#                             getattr(self, prop)(spikeNet_variable)}
#         else:
#             # If the property is set just as a weight:
#             setattr(self, prop, dummy * getattr(self, prop))
#             return {transform_fun:
#                         lambda spikeNet_variable, region_nodes_indices, weights=getattr(self, prop):
#                             spikeNet_variable * weights[region_nodes_indices]}
#
#     def generate_transforms(self):
#         dummy = np.ones((self.number_of_regions,))
#         # Confirm good shape for TVB-Spiking Network interface model parameters
#         # TODO: find a possible way to differentiate scalings between
#         #  receiver (as in _tvb_state_to_nest_current),
#         #  and sender (as in all other cases below), node indexing
#         #  Also, the size doesn't have to be in all cases equal to number_of_nodes,
#         #  but sometimes equal to number_of_spiking_nodes or to number_of_tvb_nodes
#         transforms = {}
#         for prop in ["w_tvb_to_current",
#                      "w_tvb_to_potential",
#                      "w_tvb_to_spike_rate"]:
#             transforms.update(self._prepare_tvb_to_spikeNet_transform_fun(prop, dummy))
#         for prop in ["w_spikes_to_tvb",
#                      "w_spikes_var_to_tvb",
#                      "w_potential_to_tvb"]:
#             transforms.update(self._prepare_spikeNet_to_tvb_transform_fun(prop, dummy))
#         return transforms
#
#     def build_interface(self, tvb_spikeNet_interface):
#         """
#         Configure the TVB Spiking Network interface of the fine scale as well other aspects of its interface with TVB
#         :return: tvb_spikeNet_interface object
#         """
#
#         tvb_spikeNet_interface.config = self.config
#         # TODO: find out why the model instance is different in simulator and interface...
#         tvb_spikeNet_interface.tvb_model_state_variables = self.tvb_model
#         tvb_spikeNet_interface.dt = self.tvb_dt
#         tvb_spikeNet_interface.tvb_nodes_ids = self.tvb_nodes_ids
#         tvb_spikeNet_interface.spiking_nodes_ids = self.spiking_nodes_ids
#         tvb_spikeNet_interface.exclusive_nodes = self.exclusive_nodes
#         tvb_spikeNet_interface.spiking_network = self.spiking_network
#
#         tvb_spikeNet_interface.transforms = self.generate_transforms()
#
#         tvb_spikeNet_interface.tvb_to_spikeNet_interfaces = Series({})
#         ids = [-1, -1]
#         for interface in self.tvb_to_spikeNet_interfaces:
#             model = interface.get("model", None)
#             if model in self._input_device_dict.keys():
#
#                 ids[0] += 1
#                 tvb_spikeNet_interface.tvb_to_spikeNet_interfaces = \
#                     tvb_spikeNet_interface.tvb_to_spikeNet_interfaces.append(
#                         self._tvb_to_spikNet_device_interface_builder([],
#                                                                       self.spiking_network,
#                                                                       self.spiking_nodes_ids, self.tvb_nodes_ids,
#                                                                       self.tvb_model_state_variables,
#                                                                       self.tvb_weights, self.tvb_delays,
#                                                                       self.region_labels, self.tvb_dt,
#                                                                       self.exclusive_nodes,
#                                                                       self.config).build_interface(interface, ids[0])
#                                                                             )
#             else:
#                 ids[1] += 1
#                 tvb_spikeNet_interface.tvb_to_spikeNet_interfaces = \
#                     tvb_spikeNet_interface.tvb_to_spikeNet_interfaces.append(
#                             self._tvb_to_spikeNet_parameter_interface_builder([],
#                                                                               self.spiking_network,
#                                                                               self.spiking_nodes_ids, self.tvb_nodes_ids,
#                                                                               self.tvb_model_state_variables,
#                                                                               self.tvb_model_cvar.tolist(),
#                                                                               self.exclusive_nodes,
#                                                                               self.config).build_interface(interface,
#                                                                                                            ids[1])
#                                                                             )
#
#         tvb_spikeNet_interface.spikeNet_to_tvb_interfaces = \
#             self._spikeNet_to_tvb_interface_builder(self.spikeNet_to_tvb_interfaces,
#                                                     self.spiking_network,
#                                                     self.spiking_nodes_ids, self.tvb_nodes_ids,
#                                                     self.tvb_model_state_variables,
#                                                     self.exclusive_nodes,
#                                                     self.config).build_interfaces()
#
#         return tvb_spikeNet_interface
