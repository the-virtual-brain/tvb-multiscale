# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.models_builders.base import NESTModelBuilder
from tvb_nest.simulator_nest.nest_factory import compile_modules


class WWAMPANMDAGABABuilder(NESTModelBuilder):

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None, config=CONFIGURED,
                 w_ee=1.4, J_i=1.0):
        # Some properties for the default synapse to be used:
        self.default_connection["model"] = "scale_connection"
        super(WWAMPANMDAGABABuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config)
        # Spiking populations labels:
        self.populations_names = ["AMPA", "NMDA", "GABA"]
        self.populations_models = ["iaf_cond_ampa_gaba_deco2014",
                                   "iaf_cond_nmda_deco2014",
                                   "iaf_cond_ampa_gaba_deco2014"]
        nest_models = self.nest_instance.Models()
        for model in ["iaf_cond_ampa_gaba_deco2014", "iaf_cond_nmda_deco2014", "scale_connection"]:
            if model not in nest_models:
                # If the model is not installed into NEST already
                module = model + "module"
                try:
                    # Try to install it...
                    self.nest_instance.Install(module)
                except:
                    # ...unless we need to first compile it:
                    compile_modules(model, recompile=False, config=self.config)
                    # and now install it...
                    self.nest_instance.Install(module)
                nest_models = self.nest_instance.Models()
        self.populations_params = [{}, {},  # AMPA and NMDA get the default parameters
                                   {"tau_syn": 10.0}]  # decay synaptic time for GABA has to change
        # Common order of neurons' number per population:
        self.populations_order = 100
        # Spiking populations scalings for the number of neurons:
        rcptr_ampa_gaba = self.nest_instance.GetDefaults('iaf_cond_ampa_gaba_deco2014')['receptor_types']
        rcptr_nmda = self.nest_instance.GetDefaults('iaf_cond_nmda_deco2014')['receptor_types']
        self.populations_scales = [1.0, 1.0, 0.7]

        # Connection weights between the distinct populations:
        # Inhibition to excitation feedback inhibition
        # that could result from Feedback Inhibition Control
        # (see Deco, Ponce-Alvarez et al, J. of Neuroscience, 2014)
        self.J_i = J_i
        self.w_ee = w_ee
        w_ei = -self.J_i
        w_ie = 1.0
        w_ii = -1.0

        # Within region-node connections' weights
        self.population_connectivity_synapses_weights = \
            np.array([[w_ee, w_ee, w_ei],   # AMPA->AMPA, NMDA->AMPA, GABA->AMPA
                      [w_ee, w_ee, w_ei],   # AMPA->NMDA, NMDA->NMDA, GABA->NMDA
                      [w_ie, w_ie, w_ii]])  # AMPA->GABA, NMDA->GABA, GABA->GABA
        self.population_connectivity_synapses_delays = self.tvb_dt / 4
        self.population_connectivity_synapses_receptor_types = \
            np.array([[rcptr_ampa_gaba["SPIKESEXC_AMPA_REC"], rcptr_ampa_gaba["SPIKESEXC_NMDA"], rcptr_ampa_gaba["SPIKESINH_GABA"]],
                      [rcptr_nmda["SPIKESEXC_AMPA_REC"],      rcptr_nmda["SPIKESEXC_NMDA"],      rcptr_nmda["SPIKESINH_GABA"]],
                      [rcptr_ampa_gaba["SPIKESEXC_AMPA_REC"], rcptr_ampa_gaba["SPIKESEXC_NMDA"], rcptr_ampa_gaba["SPIKESINH_GABA"]]])

        # Among/Between region-node connections
        # Given that only the AMPA population of one region-node couples to
        # all populations of another region-node,
        # we need only one connection type
        self.node_connections = \
            [{"src_population": "AMPA", "trg_population": ["AMPA", "NMDA", "GABA"],
              "model": self.default_connection["model"],
              "params": self.default_connection["params"],
              "weight": 1.0,  # weight scaling the TVB connectivity weight
              "delay": 0.0,  # additional delay to the one of TVB connectivity
              "receptor_type": rcptr_ampa_gaba["SPIKESEXC_AMPA_EXT"]},
             ]

        # Creating  devices to be able to observe NEST activity:
        self.output_devices = []
        connections = OrderedDict({})
        #          label <- target population
        connections["AMPA"] = "AMPA"
        connections["NMDA"] = "NMDA"
        connections["GABA"] = "GABA"
        params = config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"]
        params['record_from'] = ["V_m", "s", "s_bound", "emmited_spike",
                                 "s_AMPA_ext", "s_AMPA_rec", "s_NMDA", "s_GABA",
                                 "I_AMPA_ext", "I_AMPA_rec", "I_NMDA", "I_GABA", "I_leak"]
        self.output_devices.append({"model": "multimeter", "params": params,
                                    "nodes": None, "connections": connections})
        connections = OrderedDict({})
        #          label <- target population
        connections["AMPA spikes"] = "AMPA"
        connections["NMDA spikes"] = "NMDA"
        connections["GABA spikes"] = "GABA"
        params = config.nest.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_detector"]
        self.output_devices.append({"model": "spike_detector", "params": params,
                                    "nodes": None, "connections": connections})
