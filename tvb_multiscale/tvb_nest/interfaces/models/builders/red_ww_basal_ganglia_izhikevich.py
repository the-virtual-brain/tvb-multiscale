# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.tvb_nest.interfaces.base import TVBNESTInterface
from tvb_multiscale.tvb_nest.interfaces.builders.base import TVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.interfaces.models.models import RedWWexcIO
from tvb_multiscale.core.spiking_models.builders.templates import scale_tvb_weight, tvb_delay

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


class RedWWexcIOBuilder(TVBNESTInterfaceBuilder):
    _tvb_nest_interface = RedWWexcIO

    def __init__(self, tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None, populations_sizes=[200]):
        super(RedWWexcIOBuilder, self).__init__(tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes,
                                                tvb_to_nest_interfaces, nest_to_tvb_interfaces)
        self.populations_sizes = populations_sizes
        self.G = self.tvb_simulator.model.G[0].item()
        self.global_coupling_scaling = self.tvb_simulator.coupling.a[0].item() * self.G

        self.Igpe_nodes_ids = [0, 1]
        self.Igpi_nodes_ids = [2, 3]
        self.I_nodes = self.Igpe_nodes_ids + self.Igpi_nodes_ids
        self.Estn_nodes_ids = [4, 5]
        self.Eth_nodes_ids = [8, 9]
        self.E_nodes = self.Estn_nodes_ids + self.Eth_nodes_ids
        self.Istr_nodes_ids = [6, 7]

        # WongWang model parameter r is in Hz, just like poisson_generator assumes in NEST:
        self.w_tvb_to_spike_rate = 1.0
        # We return from a NEST spike_detector the ratio number_of_population_spikes / number_of_population_neurons
        # for every TVB time step, which is usually a quantity in the range [0.0, 1.0],
        # as long as a neuron cannot fire twice during a TVB time step, i.e.,
        # as long as the TVB time step (usually 0.001 to 0.1 ms)
        # is smaller than the neurons' refractory time, t_ref (usually 1-2 ms)
        # For conversion to a rate, one has to do:
        # w_spikes_to_tvb = 1/tvb_dt, to get it in spikes/ms, and
        # w_spikes_to_tvb = 1000/tvb_dt, to get it in Hz
        # given WongWang model parameter r is in Hz but tvb dt is in ms:
        self.w_spikes_to_tvb = 1000.0 / self.tvb_dt

    # No random jitter to weights and delays by default for this model

    def tvb_weight_fun(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node,
                                tvb_weights=self.tvb_weights, scale=self.global_coupling_scaling)

    def tvb_delay_fun(self, source_node, target_node):
        return np.maximum(self.tvb_dt, tvb_delay(source_node, target_node, self.tvb_delays))

    def _build_default_rate_tvb_to_nest_interfaces(self, connections, **kwargs):
        # For spike transmission from TVB to NEST devices as TVB proxy nodes with TVB delays:
        # Options:
        # "model": "poisson_generator", "params": {"allow_offgrid_times": False}
        # For spike trains with correlation probability p_copy set:
        # "model": "mip_generator", "params": {"p_copy": 0.5, "mother_seed": 0}
        # An alternative option to poisson_generator is:
        # "model": "inhomogeneous_poisson_generator", "params": {"allow_offgrid_times": False}
        interface = \
            {"model": "inhomogeneous_poisson_generator",
             "params": {"allow_offgrid_times": False},
        # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)---------
              "interface_weights": 10.0,  # Assuming a connectome of ~120 cortical regions, for a total of ~600 neurons
        # Applied outside NEST for each interface device
        #                                  Function of TVB connectivity weight:
              "weights": self.tvb_weight_fun,
        #                                  Function of TVB connectivity delay:
              "delays": self.tvb_delay_fun,
              "receptor_type": 0,
        # --------------------------------------------------------------------------------------------------------------
        #                           TVB sv or param -> NEST population
              "connections": connections,
              "source_nodes": None, "target_nodes": None}  # None means all here
        interface.update(kwargs)
        self.tvb_to_spikeNet_interfaces.append(interface)

    def build_default_rate_tvb_to_nest_interfaces(self):
        for trg_pop, target_nodes in zip([["IdSN", "IiSN"],   "E"],
                                          [self.Istr_nodes_ids, self.E_nodes]):
            connections = {"R": ensure_list(trg_pop)}
            self._build_default_rate_tvb_to_nest_interfaces(connections, target_nodes=target_nodes)

    def _build_default_nest_to_tvb_interfaces(self, connections, **kwargs):
        # NEST -> TVB:
        interface = \
            {"model": "spike_recorder", "params": {},
             # ------------------Properties potentially set as function handles with args (nest_node_id=None)-----------
             "interface_weights": 1.0, "delays": 0.0,
             # ---------------------------------------------------------------------------------------------------------
             "connections": connections, "nodes": None}  # None means all here
        interface.update(kwargs)
        self.spikeNet_to_tvb_interfaces.append(interface)

    def build_default_nest_to_tvb_interfaces(self):
        for src_pop, nodes in zip(["I",          "E",          ["IdSN", "IiSN"]],
                                  [self.I_nodes, self.E_nodes, self.Istr_nodes_ids]):
            self._build_default_nest_to_tvb_interfaces({"Rin": ensure_list(src_pop)}, nodes=nodes)

    def default_build(self, tvb_to_spikeNet_mode="rate", spikeNet_to_tvb=True):
        if tvb_to_spikeNet_mode and \
                (self.tvb_to_spikeNet_interfaces is None or len(self.tvb_to_spikeNet_interfaces) == 0):
            self.tvb_to_spikeNet_interfaces = []
            if tvb_to_spikeNet_mode.lower() == "rate":
                # For spike transmission from TVB to NEST devices as TVB proxy nodes with TVB delays:
                self.build_default_rate_tvb_to_nest_interfaces()
            else:
                raise ValueError("No %s interface for this model! Only a 'rate' one is possible!")

        if spikeNet_to_tvb and \
                (self.spikeNet_to_tvb_interfaces is None or len(self.spikeNet_to_tvb_interfaces) == 0):
            self.spikeNet_to_tvb_interfaces = []
            self.build_default_nest_to_tvb_interfaces()

    def build_interface(self, tvb_spikeNet_interface=None, tvb_to_spikeNet_mode="rate", spikeNet_to_tvb=True):
        self.default_build(tvb_to_spikeNet_mode, spikeNet_to_tvb)
        if not isinstance(tvb_spikeNet_interface, TVBNESTInterface):
            tvb_spikeNet_interface = self._tvb_nest_interface()
        return super(RedWWexcIOBuilder, self).build_interface(tvb_spikeNet_interface)
