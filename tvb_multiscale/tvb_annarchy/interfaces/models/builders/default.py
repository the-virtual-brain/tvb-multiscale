# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np

from tvb_multiscale.tvb_annarchy.interfaces.builders.base import TVBANNarchyInterfaceBuilder
from tvb_multiscale.tvb_annarchy.interfaces.base import TVBANNarchyInterface
from tvb_multiscale.tvb_annarchy.interfaces.models.models import Linear

from tvb_multiscale.core.spiking_models.builders.templates import scale_tvb_weight, tvb_delay


class DefaultInterfaceBuilder(TVBANNarchyInterfaceBuilder):
    __metaclass__ = ABCMeta

    _tvb_annarchy_interface = Linear  # Set here the target interface model

    def __init__(self, tvb_simulator, annarchy_network, annarchy_nodes_ids, exclusive_nodes=False,
                 tvb_to_annarchy_interfaces=None, annarchy_to_tvb_interfaces=None, populations_sizes=[100, 100]):

        super(DefaultInterfaceBuilder, self).__init__(tvb_simulator, annarchy_network, annarchy_nodes_ids, exclusive_nodes,
                                                      tvb_to_annarchy_interfaces, annarchy_to_tvb_interfaces)
        self.N_E = populations_sizes[0]
        if len(populations_sizes) > 1:
            self.N_I = populations_sizes[1]

        # NOTE!!! TAKE CARE OF DEFAULT simulator.coupling.a!
        self.global_coupling_scaling = self.tvb_simulator.coupling.a[0].item()

    # By default we choose weights and delays with a random jitter around TVB ones!

    def tvb_weight_fun(self, source_node, target_node, scale=None):
        if scale is None:
            scale = self.global_coupling_scaling
        return scale_tvb_weight(source_node, target_node, self.tvb_weights, scale=scale)

    def tvb_delay_fun(self, source_node, target_node):
        return tvb_delay(source_node, target_node, self.tvb_delays)

    def receptor_fun(self, source_node, target_node, start=0):
        return "exc"

    # Spike rates are applied in parallelto neurons...

    def _build_default_rate_tvb_to_annarchy_interfaces(self, connections, **kwargs):
        # For spike transmission from TVB to ANNarchy devices as TVB proxy nodes with TVB delays:
        # Options:
        # "model": "poisson_generator", "params": {"allow_offgrid_times": False}
        # For spike trains with correlation probability p_copy set:
        # "model": "mip_generator", "params": {"p_copy": 0.5, "mother_seed": 0}
        # An alternative option to poisson_generator is:
        # "model": "inhomogeneous_poisson_generator", "params": {"allow_offgrid_times": False}
        interface = \
            {"model": "TimedPoissonPopulation",  # HomogeneousCorrelatedSpikeTrains, Poisson_neuron
             "params": {"period": self.tvb_dt},
        # -------Properties potentially set as function handles with args (tvb_node_id=None, annarchy_node_id=None)-----
              "interface_weights": 1.0*self.N_E,
        # Applied outside ANNarchy for each interface device
        #                                  Function of TVB connectivity weight:
              "weights": self.tvb_weight_fun,
        #                                  Function of TVB connectivity delay:
              "delays": self.tvb_delay_fun,
              "receptor_type": self.receptor_fun,
              "neurons_inds": lambda tvb_id, annarchy_id, neurons_inds:
                                tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
        # --------------------------------------------------------------------------------------------------------------
        #                           TVB sv or param -> ANNarchy population
              "connections": connections,
              "source_nodes": None, "target_nodes": None}  # None means all here
        interface.update(kwargs)
        self.tvb_to_spikeNet_interfaces.append(interface)

    # ...unlike currents that have to be distributed to neurons (e.g., N_E / N_E = 1.0)

    # def _build_default_current_tvb_to_annarchy_interfaces(self, connections, **kwargs):
    #     # For injecting current to ANNarchy neurons via dc generators acting as TVB proxy nodes with TVB delays:
    #     interface = \
    #         {"model": "DCCurrentInjector", "params": {},
    #     # -------Properties potentially set as function handles with args (tvb_node_id=None, annarchy_node_id=None)---
    #          # Applied outside ANNarchy for each interface device:
    #          "interface_weights": 1.0,
    #          #                                 Function of TVB connectivity weight:
    #          "weights": self.tvb_weight_fun,
    #          #                                 Function of TVB connectivity delay:
    #          "delays": self.tvb_delay_fun,
    #          "neurons_inds": lambda tvb_id, annarchy_id, neurons_inds:
    #                             tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
    #          # -------------------------------------------------------------------------------------------------------
    #          #                                                 TVB sv -> ANNarchy population
    #          "connections": connections,
    #          "source_nodes": None, "target_nodes": None}  # None means all here
    #     interface.update(kwargs)
    #     self.tvb_to_spikeNet_interfaces.append(interface)

    def _build_default_param_tvb_to_annarchy_interfaces(self, connections, **kwargs):
        # For directly setting an external current parameter in ANNarchy neurons instantaneously:
        interface = \
            {"model": "current", "parameter": "I",
            # ---------Properties potentially set as function handles with args (annarchy_node_id=None)----------------
            "interface_weights": 1.0,
            "neurons_inds": lambda node_id, neurons_inds:
                                tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
            # ----------------------------------------------------------------------------------------------------------
            #                                               TVB sv -> ANNarchy population
            "connections": connections,
            "nodes": None}  # None means all here
        interface.update(kwargs)
        self.tvb_to_spikeNet_interfaces.append(interface)

    def _build_default_annarchy_to_tvb_interfaces(self, connections, **kwargs):
        # ANNarchy -> TVB:
        interface = \
            {"model": "SpikeMonitor", "params": {},
             # ------------------Properties potentially set as function handles with args (annarchy_node_id=None)-------
             "interface_weights": 1.0, "delays": 0.0,
             "neurons_inds": lambda node_id, neurons_inds:
                                 tuple(np.array(neurons_inds)[:np.minimum(100, len(neurons_inds))]),
             # ---------------------------------------------------------------------------------------------------------
             "connections": connections, "nodes": None}  # None means all here
        interface.update(kwargs)
        self.spikeNet_to_tvb_interfaces.append(interface)

    def build_default_rate_tvb_to_annarchy_interfaces(self):
        self._build_default_rate_tvb_to_annarchy_interfaces({"R": ["E"]}, params={"geometry": self.N_E,
                                                                                  "period": self.tvb_dt})

    @abstractmethod
    def build_default_current_tvb_to_annarchy_interfaces(self):
        raise NotImplementedError

    @abstractmethod
    def build_default_param_tvb_to_annarchy_interfaces(self):
        raise NotImplementedError

    def build_default_annarchy_to_tvb_interfaces(self):
        self._build_default_annarchy_to_tvb_interfaces({"Rin": ["E"]})

    def default_build(self, tvb_to_annarchy_mode="rate", annarchy_to_tvb=True):
        if tvb_to_annarchy_mode and \
                (self.tvb_to_spikeNet_interfaces is None or len(self.tvb_to_spikeNet_interfaces) == 0):
            self.tvb_to_spikeNet_interfaces = []
            if tvb_to_annarchy_mode.lower() == "rate":
                # For spike transmission from TVB to ANNarchy devices as TVB proxy nodes with TVB delays:
                self.build_default_rate_tvb_to_annarchy_interfaces()
            # elif tvb_to_annarchy_mode.lower() == "current":
            #     # For injecting current to ANNarchy neurons via dc generators acting as TVB proxy nodes with TVB delays:
            #     self.build_default_current_tvb_to_annarchy_interfaces()
            elif tvb_to_annarchy_mode.lower() == "param":
                # For directly setting an external current parameter in ANNarchy neurons instantaneously:
                self.build_default_param_tvb_to_annarchy_interfaces()

        # The ANNarchy nodes the activity of which is transformed to TVB state variables or parameters
        if annarchy_to_tvb and \
                (self.spikeNet_to_tvb_interfaces is None or len(self.spikeNet_to_tvb_interfaces) == 0):
            self.spikeNet_to_tvb_interfaces = []
            self.build_default_annarchy_to_tvb_interfaces()

    def build_interface(self, tvb_annarchy_interface=None, tvb_to_annarchy_mode="rate", annarchy_to_tvb=True):
        self.default_build(tvb_to_annarchy_mode, annarchy_to_tvb)
        if not isinstance(tvb_annarchy_interface, TVBANNarchyInterface):
            tvb_annarchy_interface = self._tvb_annarchy_interface()
        return super(DefaultInterfaceBuilder, self).build_interface(tvb_annarchy_interface)
