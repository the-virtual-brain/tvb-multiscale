# -*- coding: utf-8 -*-

from collections import OrderedDict
from tvb_multiscale.config import CONFIGURED
from tvb_nest.interfaces.builders.base import TVBNESTInterfaceBuilder
from tvb_nest.interfaces.models import RedWWexcIOinhI
from tvb_nest.simulator_tvb.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI


class WWAMPANMDAGABABuilder(TVBNESTInterfaceBuilder):
    tvb_model = ReducedWongWangExcIOInhI()

    def __init__(self, tvb_simulator, nest_network, spiking_nodes_ids, N_e=100, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None, config=CONFIGURED):

        if tvb_to_nest_interfaces is None:
            # # For directly setting an external current parameter in NEST neurons instantaneously:
            self.tvb_to_nest_interfaces = [{"model": "current", "parameter": "I_e",
                                            # ---------Properties potentially set as function handles with args (nest_node_id=None)-----------------------------
                                            "interface_weights": 1.0,
                                            # ------------------------------------------------------------------------------------------------------------------
                                            #                TVB sv -> NEST population
                                            "connections": {"S_e": ["AMPA", "NMDA", "GABA"]},
                                            "nodes": None}]  # None means all here

    # For injecting current to NEST neurons via dc generators acting as TVB proxy nodes with TVB delays:
    # #         self.tvb_to_spikeNet_interfaces = [{"model": "dc_generator", "params": {},
    # # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
    #                                        "interface_weights": 1.0, # Applied outside NEST for each interface device
    #                                        "weights": 1.0,  # To multiply TVB connectivity weight
    # #                                      To add to TVB connectivity delay:
    #                                        "delays": spiking_network.nodes_min_delay,
    # --------------------------------------------------------------------------------------------------------------
    # #                                                TVB sv -> NEST population
    #                                        "connections": {"S_e": ["AMPA", "NMDA", "GABA"]},
    #                                        "source_nodes": None, "target_nodes": None}]  # None means all here
    #
    # #For spike transmission from TVB to NEST via poisson generators acting as TVB proxy nodes with TVB delays:
    #           self.tvb_to_spikeNet_interfaces =  [{"model": "poisson_generator", "params": {},
    # # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
    #                                           "interface_weights": 1.0*N_e,  # Applied outside NEST for each interface device
    #                                         "weights": tvb_simulator.model.G[0],  # To multiply TVB connectivity weight
    # #                                   To add to TVB connectivity delay:
    #                                           "delays": spiking_network.nodes_min_delay,
    #                                          "receptor_types": lambda tvb_node_id, nest_node_id: tvb_node_id + 1,
    # --------------------------------------------------------------------------------------------------------------
    # #                                                       TVB sv -> NEST population
    #                                         "connections": {"R_e": ["AMPA", "NMDA", "GABA"]},
    #                                         "source_nodes": None, "target_nodes": None}]  # None means all here
        # The NEST nodes the activity of which is transformed to TVB state variables or parameters
        if nest_to_tvb_interfaces is None:
            # NEST -> TVB:
            # Use S_e and S_i instead of r_e and r_i
            # for transmitting to the TVB state variables directly
            connections = OrderedDict()
            #            TVB <- NEST
            connections["R_e"] = ["AMPA", "NMDA"]
            connections["R_i"] = "GABA"
            nest_to_tvb_interfaces = \
                [{"model": "spike_detector", "params": {},
    # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
                  "weights": 1.0, "delays": 0.0,
    # -----------------------------------------------------------------------------------------------------------------
                  "connections": connections, "nodes": None}]  # None means all here

        super(WWAMPANMDAGABABuilder, self).__init__(tvb_simulator, nest_network, spiking_nodes_ids, exclusive_nodes,
                                                    tvb_to_nest_interfaces, nest_to_tvb_interfaces, config)
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

    def build_interface(self, tvb_nest_interface=None):
        if not isinstance(tvb_nest_interface, RedWWexcIOinhI):
            tvb_nest_interface = RedWWexcIOinhI()
        return super(WWAMPANMDAGABABuilder, self).build_interface(tvb_nest_interface)
