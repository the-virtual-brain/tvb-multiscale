# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from tvb_nest.interfaces.builders.base import TVBNESTInterfaceBuilder
from tvb_nest.interfaces.models import RedWWexcIOinhI
from tvb_nest.simulator_tvb.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI


# We need to "neutralize" TVB connectivity here, i.e., have all connections have a weight equal to 1.0.
# Therefore:
def weight_fun(tvb_node_id, nest_node_id, tvb_weights, G):
    if tvb_weights[tvb_node_id, nest_node_id] > 0.0:
        weight = G * tvb_weights[tvb_node_id, nest_node_id]
        return {"distribution": "normal", "mu": weight, "sigma": 0.1 * weight}
    else:
        return 0.0


class WWDeco2014Builder(TVBNESTInterfaceBuilder):
    tvb_model = ReducedWongWangExcIOInhI()

    def __init__(self, tvb_simulator, nest_network, nest_nodes_ids, N_E=100, exclusive_nodes=False,
                 tvb_to_nest_interfaces=None, nest_to_tvb_interfaces=None):

        if tvb_to_nest_interfaces is None:
    # TVB -> NEST
    # # # For directly setting an external current parameter in NEST neurons instantaneously:
    #         tvb_to_nest_interfaces = [{"model": "current",  "parameter": "I_e",
    # # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
    #                                    "interface_weights": 2.0,
    # # ----------------------------------------------------------------------------------------------------------------
    # #                                   TVB sv -> NEST population
    #                                     "connections": {"S_e": ["E", "I"]},
    #                                     "nodes": None}]  # None means all here

    # # For injecting current to NEST neurons via dc generators acting as TVB proxy nodes with TVB delays:
    #         tvb_to_nest_interfaces = [{"model": "dc_generator", "params": {},
    # # ---------Properties potentially set as function handles with args (nest_node_id=None)---------------------------
    # #                            Applied outside NEST for each interface device
    #                                     "interface_weights": 1.0 * N_E,
    # # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
    # #                                       To multiply TVB connectivity weight:
    #                                     "weights": lambda tvb_node_id, nest_node_id, weights:
    #                                                   weight_fun(tvb_node_id, nest_node_id,
    #                                                              weights, tvb_simulator.model.G[0]),
    # #                                   A function of TVB connectivity delay:
    #                                      "delays": lambda tvb_node_id, nest_node_id, delays:
    #                                                         delays[tvb_node_id, nest_node_id],
    # # ----------------------------------------------------------------------------------------------------------------
    # #                                                 TVB sv -> NEST population
    #                                      "connections": {"S_e": ["E", "I"]},
    #                                      "source_nodes": None, "target_nodes": None}]  # None means all here

            # For spike transmission from TVB to NEST devices acting as TVB proxy nodes with TVB delays:
            # Options:
            # "model": "poisson_generator", "params": {"allow_offgrid_times": False}
            # For spike trains with correlation probability p_copy set:
            # "model": "mip_generator", "params": {"p_copy": 0.5, "mother_seed": 0}
            # An alternative option to poisson_generator is:
            # "model": "inhomogeneous_poisson_generator", "params": {"allow_offgrid_times": False}
            tvb_to_nest_interfaces = [{"model": "inhomogeneous_poisson_generator",
                                        "params": {"allow_offgrid_times": False},
    # # ---------Properties potentially set as function handles with args (nest_node_id=None)-------------------------
                                         "interface_weights": 1.0 * N_E, # The number of excitatory neurons per region
    # Applied outside NEST for each interface device
    # -------Properties potentially set as function handles with args (tvb_node_id=None, nest_node_id=None)-----------
    #                                    To multiply TVB connectivity weight:
                                         "weights": 1.0,
    #                                     A function of TVB connectivity delay:
    #                                     "delays": lambda tvb_node_id, nest_node_id, delays:
    #                                                           delays[tvb_node_id, nest_node_id],
                                         "receptor_types": lambda tvb_node_id, nest_node_id: int(tvb_node_id + 1),
    # --------------------------------------------------------------------------------------------------------------
    #                                          TVB sv -> NEST population
                                         "connections": {"R_e": ["E", "I"]},
                                         "source_nodes": None, "target_nodes": None}]  # None means all here

    # The NEST nodes the activity of which is transformed to TVB state variables or parameters
        if nest_to_tvb_interfaces is None:
            # NEST -> TVB:
            # Use S_e and S_i instead of r_e and r_i
            # for transmitting to the TVB state variables directly
            connections = OrderedDict()
            #            TVB <- NEST
            connections["R_e"] = ["E"]
            connections["R_i"] = ["I"]
            nest_to_tvb_interfaces = \
                [{"model": "spike_detector", "params": {},
    # ------------------Properties potentially set as function handles with args (nest_node_id=None)--------------------
                  "weights": 1.0, "delays": 0.0,
    # -----------------------------------------------------------------------------------------------------------------
                  "connections": connections, "nodes": None}]  # None means all here

        super(WWDeco2014Builder, self).__init__(tvb_simulator, nest_network, nest_nodes_ids, exclusive_nodes,
                                                tvb_to_nest_interfaces, nest_to_tvb_interfaces)
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
        return super(WWDeco2014Builder, self).build_interface(tvb_nest_interface)
