# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_nest.interfaces.builders.base import TVBNESTInterfaceBuilder
from tvb_nest.interfaces.models import RedWWexcIOinhI
from tvb_multiscale.spiking_models.builders.templates import scale_tvb_weight, tvb_delay


class RedWWexcIOinhIBuilder(TVBNESTInterfaceBuilder):

    def __init__(self, tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None, N_e=100):
        super(RedWWexcIOinhIBuilder, self).__init__(tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes,
                                                    tvb_to_nest_interfaces, nest_to_tvb_interfaces)
        if tvb_to_nest_interfaces is None:
    # # For directly setting an external current parameter in NEST neurons instantaneously:
    #         self.tvb_to_spikeNet_interfaces = [{"model": "current",  "parameter": "I_e",
    # # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
    #                                           "interface_weights": 5.0,
    # # ----------------------------------------------------------------------------------------------------------------
    # #                                               TVB sv -> NEST population
    #                                           "connections": {"S_e": ["E"]},
    #                                           "nodes": None}]  # None means all here
    #
    #         if self.tvb_model.lamda[0] > 0.0:
    #             #       Coupling towards the inhibitory population as well:
    #             self.tvb_to_spikeNet_interfaces.append({
    #                 "model": "current", "parameter": "I_e",
    #                 # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
    #                 "interface_weights": 5.0 * self.tvb_model.lamda[0],
    #                 # ----------------------------------------------------------------------------------------------------------------
    #                 #                                               TVB sv -> NEST population
    #                 "connections": {"S_e": ["I"]},
    #                 "nodes": None}
    #             )
    #

    # # # For injecting current to NEST neurons via dc generators acting as TVB proxy nodes with TVB delays:
    #
    #         self.tvb_to_spikeNet_interfaces = [{"model": "dc_generator", "params": {},
    # # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
    #                                           "interface_weights": N_e * 1.0,  # Applied outside NEST for each interface device
    # #                                 Function of TVB connectivity weight:
    #                                           "weights": self.G_scale_tvb_weight_exc,
    # #                                 Function of TVB connectivity delay:
    # #                                          "delays": self.tvb_delay,
    # # ----------------------------------------------------------------------------------------------------------------
    # #                                                 TVB sv -> NEST population
    #                                           "connections": {"S_e": ["E"]},
    #                                           "source_nodes": None, "target_nodes": None}]  # None means all here
    #
    #         if self.tvb_model.lamda[0] > 0.0:
    #             #       Coupling towards the inhibitory population as well:
    #             self.tvb_to_spikeNet_interfaces.append(
    #                 {"model": "dc_generator", "params": {},
    #                  # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
    #                  "interface_weights": N_e * 1.0,  # Applied outside NEST for each interface device
    #                  #                                 A function of TVB connectivity weight
    #                  "weights": self.G_scale_tvb_weight_inh,
    #                  #                                 A function of TVB connectivity delay
    #                  #                                           "delays": self.tvb_delay,
    #                  # ----------------------------------------------------------------------------------------------------------------
    #                  #                                                 TVB sv -> NEST population
    #                  "connections": {"S_e": ["I"]},
    #                  "source_nodes": None, "target_nodes": None}
    #             )


    # For spike transmission from TVB to NEST devices as TVB proxy nodes with TVB delays:

    # Options:
    # "model": "poisson_generator", "params": {"allow_offgrid_times": False}
    # For spike trains with correlation probability p_copy set:
    # "model": "mip_generator", "params": {"p_copy": 0.5, "mother_seed": 0}
    # An alternative option to poisson_generator is:
    # "model": "inhomogeneous_poisson_generator", "params": {"allow_offgrid_times": False}
            self.tvb_to_spikeNet_interfaces = [{"model": "inhomogeneous_poisson_generator",
                                            "params": {"allow_offgrid_times": False},
    # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
                                            "interface_weights": N_e * 1.0,  # Applied outside NEST for each interface device
    #                                  Function of TVB connectivity weight:
                                            "weights": self.G_scale_tvb_weight_exc,
    #                                  Function of TVB connectivity delay:
                                            "delays": self.tvb_delay,
                                            "receptor_types": 0,
    # ----------------------------------------------------------------------------------------------------------------
    #                                        TVB sv or param -> NEST population
                                            "connections": {"R_e": ["E"]},
                                            "source_nodes": None, "target_nodes": None}]  # None means all here

            if self.tvb_model.lamda[0] > 0.0:
                #       Coupling towards the inhibitory population as well:
                self.tvb_to_spikeNet_interfaces.append(
                    {"model": "inhomogeneous_poisson_generator",
                     "params": {"allow_offgrid_times": False},
                     # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
                     "interface_weights": N_e * 1.0,  # Applied outside NEST for each interface device
                     #                               A function of TVB connectivity weight
                     "weights": self.G_scale_tvb_weight_inh,
                     #                                 A function of TVB connectivity delay:
                     "delays": self.tvb_delay,
                     "receptor_types": 0,
                     # ----------------------------------------------------------------------------------------------------------------
                     #                                        TVB sv or param -> NEST population
                     "connections": {"R_e": ["I"]},
                     "source_nodes": None, "target_nodes": None}
                )

    # The NEST nodes the activity of which is transformed to TVB state variables or parameters
        if nest_to_tvb_interfaces is None:
            # NEST -> TVB:
            # Use S_e and S_i instead of r_e and r_i
            # for transmitting to the TVB state variables directly
            connections = OrderedDict()
            #            TVB <- NEST
            connections["R_e"] = ["E"]
            connections["R_i"] = ["I"]
            self.spikeNet_to_tvb_interfaces = \
                [{"model": "spike_detector", "params": {},
    # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
                  "weights": 1.0, "delays": 0.0,
    # ------------------------------------------------------------------------------------------------------------------
                  "connections": connections, "nodes": None}]  # None means all here


        self.w_tvb_to_current = 1000 * self.tvb_model.J_N[0]  # (nA of TVB -> pA of NEST)
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

    def G_scale_tvb_weight_exc(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node,
                                tvb_weights=self.tvb_weights,
                                scale=self.tvb_model.G[0])

    def G_scale_tvb_weight_inh(self, source_node, target_node):
        return scale_tvb_weight(source_node, target_node,
                                tvb_weights=self.tvb_weights,
                                scale=self.tvb_model.lamda[0] * self.tvb_model.G[0])

    def tvb_delay(self, source_node, target_node):
        return tvb_delay(source_node, target_node, self.tvb_delays)

    def build_interface(self, tvb_nest_interface=None):
        if not isinstance(tvb_nest_interface, RedWWexcIOinhI):
            tvb_nest_interface = RedWWexcIOinhI()
        return super(RedWWexcIOinhIBuilder, self).build_interface(tvb_nest_interface)
