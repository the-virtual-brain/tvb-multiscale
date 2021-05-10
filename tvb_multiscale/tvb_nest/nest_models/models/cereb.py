# -*- coding: utf-8 -*-
import h5py

from collections import OrderedDict

import numpy as np

from tvb_multiscale.tvb_nest.config import CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.builders.base import NESTNetworkBuilder


class NeuronsFun(object):
    conns = np.array([])
    start_id_scaffold = 0

    def __init__(self, start_id_scaffold, conns):
        self.start_id_scaffold = start_id_scaffold
        self.conns = conns

    def __call__(self, population):
        return population.nest_instance.NodeCollection[[int(x - self.start_id_scaffold + population.tolist()[0])
                                                        for x in self.conns]]


class CerebBuilder(NESTNetworkBuilder):

    output_devices_record_to = "ascii"

    path_to_network_source_file = ""

    # Synapse parameters: in E-GLIF, 3 synaptic receptors are present:
    # the first is always associated to exc, the second to inh, the third to remaining synapse type
    Erev_exc = 0.0  # [mV]	#[Cavallari et al, 2014]
    Erev_inh = -80.0  # [mV]
    # tau_exc for pc is for pf input; tau_exc for goc is for mf input; tau_exc for mli is for pf input
    tau_exc = {'golgi': 0.23, 'granule': 5.8, 'purkinje': 1.1, 'basket': 0.64, 'stellate': 0.64, 'dcn': 1.0,
               'dcnp': 3.64, 'dcnGlyI': 1.0,
               'io': 1.0}
    tau_inh = {'golgi': 10.0, 'granule': 13.61, 'purkinje': 2.8, 'basket': 2.0, 'stellate': 2.0, 'dcn': 0.7,
               'dcnp': 1.14, 'dcnGlyI': 1.0, 'io': 60.0}
    tau_exc_cfpc = 0.4
    tau_exc_pfgoc = 0.5
    tau_exc_cfmli = 1.2

    # Single neuron parameters:
    Zpos = False

    neuron_param = {
        'golgi_cell': {'t_ref': 2.0, 'C_m': 145.0, 'tau_m': 44.0, 'V_th': -55.0, 'V_reset': -75.0, 'Vinit': -62.0,
                       'E_L': -62.0, 'Vmin': -150.0,
                       'lambda_0': 1.0, 'tau_V': 0.4, 'I_e': 16.214, 'kadap': 0.217, 'k1': 0.031, 'k2': 0.023,
                       'A1': 259.988, 'A2': 178.01,
                       'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'E_rev4': Erev_exc,
                       'tau_syn1': tau_exc['golgi'], 'tau_syn2': tau_inh['golgi'], 'tau_syn3': tau_exc_pfgoc,
                       'tau_syn4': tau_exc['golgi']},
        'granule_cell': {'t_ref': 1.5, 'C_m': 7.0, 'tau_m': 24.15, 'V_th': -41.0, 'V_reset': -70.0, 'Vinit': -62.0,
                         'E_L': -62.0, 'Vmin': -150.0,
                         'lambda_0': 1.0, 'tau_V': 0.3, 'I_e': -0.888, 'kadap': 0.022, 'k1': 0.311, 'k2': 0.041,
                         'A1': 0.01, 'A2': -0.94,
                         'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['granule'],
                         'tau_syn2': tau_inh['granule'], 'tau_syn3': tau_exc['granule']},
        'purkinje_cell': {'t_ref': 0.5, 'C_m': 334.0, 'tau_m': 47.0, 'V_th': -43.0, 'V_reset': -69.0, 'Vinit': -59.0,
                          'E_L': -59.0,
                          'lambda_0': 4.0, 'tau_V': 3.5, 'I_e': Ie_pc, 'kadap': 1.492, 'k1': 0.1950, 'k2': 0.041,
                          'A1': 157.622, 'A2': 172.622,
                          'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['purkinje'],
                          'tau_syn2': tau_inh['purkinje'], 'tau_syn3': tau_exc_cfpc},
        'basket_cell': {'t_ref': 1.59, 'C_m': 14.6, 'tau_m': 9.125, 'V_th': -53.0, 'V_reset': -78.0, 'Vinit': -68.0,
                        'E_L': -68.0,
                        'lambda_0': 1.8, 'tau_V': 1.1, 'I_e': 3.711, 'kadap': 2.025, 'k1': 1.887, 'k2': 1.096,
                        'A1': 5.953, 'A2': 5.863,
                        'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['basket'],
                        'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
        'stellate_cell': {'t_ref': 1.59, 'C_m': 14.6, 'tau_m': 9.125, 'V_th': -53.0, 'V_reset': -78.0, 'Vinit': -68.0,
                          'E_L': -68.0,
                          'lambda_0': 1.8, 'tau_V': 1.1, 'I_e': 3.711, 'kadap': 2.025, 'k1': 1.887, 'k2': 1.096,
                          'A1': 5.953, 'A2': 5.863,
                          'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['basket'],
                          'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
        'dcn_cell_glut_large': {'t_ref': 1.5, 'C_m': 142.0, 'tau_m': 33.0, 'V_th': -36.0, 'V_reset': -55.0,
                                'Vinit': -45.0, 'E_L': -45.0,
                                'lambda_0': 3.5, 'tau_V': 3.0, 'I_e': 75.385, 'kadap': 0.408, 'k1': 0.697, 'k2': 0.047,
                                'A1': 13.857, 'A2': 3.477,
                                'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['dcn'],
                                'tau_syn2': tau_inh['dcn']},
        'dcn_cell_GABA': {'t_ref': 3.0, 'C_m': 56.0, 'tau_m': 56.0, 'V_th': -39.0, 'V_reset': -55.0, 'Vinit': -40.0,
                          'E_L': -40.0,
                          'lambda_0': 0.9, 'tau_V': 1.0, 'I_e': 2.384, 'kadap': 0.079, 'k1': 0.041, 'k2': 0.044,
                          'A1': 176.358, 'A2': 176.358,
                          'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['dcnp'],
                          'tau_syn2': tau_inh['dcnp']},
        'dcn_cell_Gly-I': {'t_ref': 3.0, 'C_m': 56.0, 'tau_m': 56.0, 'V_th': -39.0, 'V_reset': -55.0, 'Vinit': -40.0,
                           'E_L': -40.0,
                           'lambda_0': 0.9, 'tau_V': 1.0, 'I_e': 2.384, 'kadap': 0.079, 'k1': 0.041, 'k2': 0.044,
                           'A1': 176.358, 'A2': 176.358,
                           'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['dcnGlyI'],
                           'tau_syn2': tau_inh['dcnGlyI']},
        'io_cell': {'t_ref': 1.0, 'C_m': 189.0, 'tau_m': 11.0, 'V_th': -35.0, 'V_reset': -45.0, 'Vinit': -45.0,
                    'E_L': -45.0,
                    'lambda_0': 1.2, 'tau_V': 0.8, 'I_e': -18.101, 'kadap': 1.928, 'k1': 0.191, 'k2': 0.091,
                    'A1': 1810.93, 'A2': 1358.197,
                    'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc, 'tau_syn1': tau_exc['io'],
                    'tau_syn2': tau_inh['io']}}

    # Connection weights
    conn_weights = {'mossy_to_glomerulus': 1.0, 'ascending_axon_to_golgi': 1.5, 'ascending_axon_to_purkinje': 1.08,
                    'basket_to_purkinje': 0.436, 'basket_to_basket': 0.123,
                    'glomerulus_to_golgi': 1.5, 'glomerulus_to_granule': 0.2, 'golgi_to_granule': 2.5,
                    'golgi_to_golgi': 2.63,
                    'parallel_fiber_to_basket': 0.023, 'parallel_fiber_to_golgi': 0.083,
                    'parallel_fiber_to_purkinje': 0.034,
                    'parallel_fiber_to_stellate': 0.042, 'stellate_to_purkinje': 0.85, 'stellate_to_stellate': 0.109,
                    'dcn_GABA_to_io': 0.00000001, 'io_to_dcn_GABA': 0.2, 'io_to_dcn_glut_large': 0.1,
                    'io_to_purkinje': 350.0,
                    'mossy_to_dcn_Gly-I': 0.8, 'mossy_to_dcn_glut_large': 0.5, 'purkinje_to_dcn_GABA': 0.00418,
                    'purkinje_to_dcn_Gly-I': 0.2, 'purkinje_to_dcn_glut_large': 0.5}

    # Connection delays
    conn_delays = {'mossy_to_glomerulus': 1.0, 'ascending_axon_to_golgi': 2.0, 'ascending_axon_to_purkinje': 2.0,
                   'basket_to_purkinje': 4.0, 'basket_to_basket': 1.0,
                   'glomerulus_to_golgi': 4.0, 'glomerulus_to_granule': 4.0, 'golgi_to_granule': 2.0,
                   'golgi_to_golgi': 1.0,
                   'parallel_fiber_to_basket': 5.0, 'parallel_fiber_to_golgi': 5.0,
                   'parallel_fiber_to_purkinje': 5.0,
                   'parallel_fiber_to_stellate': 5.0, 'stellate_to_purkinje': 5.0, 'stellate_to_stellate': 1.0,
                   'NC_dcn_Gly-I_to_golgi': 1.0, 'NC_dcn_glut_large_to_golgi': 1.0,
                   'NC_dcn_glut_large_to_granule': 1.0, 'dcn_GABA_to_io': 20.0, 'io_to_dcn_GABA': 5.0,
                   'io_to_basket': {"mu": 70.0, "sigma": 10.0}, 'io_to_stellate': {"mu": 70.0, "sigma": 10.0},
                   'io_to_purkinje': 4.0, 'io_to_dcn_glut_large': 4.0,
                   'mossy_to_dcn_Gly-I': 4.0, 'mossy_to_dcn_glut_large': 4.0, 'purkinje_to_dcn_GABA': 4.0,
                   'purkinje_to_dcn_Gly-I': 4.0, 'purkinje_to_dcn_glut_large': 4.0}

    # Connection receptors
    conn_receptors = {'ascending_axon_to_golgi': 3, 'ascending_axon_to_purkinje': 1, 'basket_to_purkinje': 2,
                      'basket_to_basket': 2,
                      'glomerulus_to_golgi': 1, 'glomerulus_to_granule': 1, 'golgi_to_granule': 2, 'golgi_to_golgi': 2,
                      'parallel_fiber_to_basket': 1, 'parallel_fiber_to_golgi': 3, 'parallel_fiber_to_purkinje': 1,
                      'parallel_fiber_to_stellate': 1, 'stellate_to_purkinje': 2, 'stellate_to_stellate': 2,
                      'NC_dcn_Gly-I_to_golgi': 4, 'NC_dcn_glut_large_to_golgi': 4,
                      'NC_dcn_glut_large_to_granule': 3, 'dcn_GABA_to_io': 2, 'io_to_basket': 3,
                      'io_to_dcn_GABA': 1, 'io_to_dcn_glut_large': 1, 'io_to_purkinje': 3, 'io_to_stellate': 3,
                      'mossy_to_dcn_Gly-I': 1, 'mossy_to_dcn_glut_large': 1, 'purkinje_to_dcn_GABA': 2,
                      'purkinje_to_dcn_Gly-I': 2, 'purkinje_to_dcn_glut_large': 2}

    # Connection pre and post-synaptic neurons
    conn_pre_post = {'mossy_to_glomerulus': {'pre': 'mossy_fibers', 'post': 'glomerulus'},
                     'ascending_axon_to_golgi': {'pre': 'granule_cell', 'post': 'golgi_cell'},
                     'ascending_axon_to_purkinje': {'pre': 'granule_cell', 'post': 'purkinje_cell'},
                     'basket_to_purkinje': {'pre': 'basket_cell', 'post': 'purkinje_cell'},
                     'basket_to_basket': {'pre': 'basket_cell', 'post': 'basket_cell'},
                     'glomerulus_to_golgi': {'pre': 'glomerulus', 'post': 'golgi_cell'},
                     'glomerulus_to_granule': {'pre': 'glomerulus', 'post': 'granule_cell'},
                     'golgi_to_granule': {'pre': 'golgi_cell', 'post': 'granule_cell'},
                     'golgi_to_golgi': {'pre': 'golgi_cell', 'post': 'golgi_cell'},
                     'parallel_fiber_to_basket': {'pre': 'granule_cell', 'post': 'basket_cell'},
                     'parallel_fiber_to_golgi': {'pre': 'granule_cell', 'post': 'golgi_cell'},
                     'parallel_fiber_to_purkinje': {'pre': 'granule_cell', 'post': 'purkinje_cell'},
                     'parallel_fiber_to_stellate': {'pre': 'granule_cell', 'post': 'stellate_cell'},
                     'stellate_to_purkinje': {'pre': 'stellate_cell', 'post': 'purkinje_cell'},
                     'stellate_to_stellate': {'pre': 'stellate_cell', 'post': 'stellate_cell'},
                     'NC_dcn_Gly-I_to_golgi': {'pre': 'dcn_cell_Gly-I', 'post': 'golgi_cell'},
                     'NC_dcn_glut_large_to_golgi': {'pre': 'dcn_cell_glut_large', 'post': 'golgi_cell'},
                     'NC_dcn_glut_large_to_granule': {'pre': 'dcn_cell_glut_large', 'post': 'granule_cell'},
                     'dcn_GABA_to_io': {'pre': 'dcn_cell_GABA', 'post': 'io_cell'},
                     'io_to_basket': {'pre': 'io_cell', 'post': 'basket_cell'},
                     'io_to_dcn_GABA': {'pre': 'io_cell', 'post': 'dcn_cell_GABA'},
                     'io_to_dcn_glut_large': {'pre': 'io_cell', 'post': 'dcn_cell_glut_large'},
                     'io_to_purkinje': {'pre': 'io_cell', 'post': 'purkinje_cell'},
                     'io_to_stellate': {'pre': 'io_cell', 'post': 'stellate_cell'},
                     'mossy_to_dcn_Gly-I': {'pre': 'mossy_fibers', 'post': 'dcn_cell_Gly-I'},
                     'mossy_to_dcn_glut_large': {'pre': 'mossy_fibers', 'post': 'dcn_cell_glut_large'},
                     'purkinje_to_dcn_GABA': {'pre': 'purkinje_cell', 'post': 'dcn_cell_GABA'},
                     'purkinje_to_dcn_Gly-I': {'pre': 'purkinje_cell', 'post': 'dcn_cell_Gly-I'},
                     'purkinje_to_dcn_glut_large': {'pre': 'purkinje_cell', 'post': 'dcn_cell_glut_large'}
                     }

    RECORD_VM = True
    STIMULUS = True
    TOT_DURATION = 400.  # mseconds
    STIM_MF_START = 140.  # beginning of stimulation to MFs
    STIM_MF_END = 300.  # end of stimulation to MFs
    STIM_MF_FREQ = 40.  # MF Frequency in Hz
    STIM_IO_START = 290.  # beginning of stimulation to IO
    STIM_IO_END = 300.  # end of stimulation to IO
    STIM_IO_FREQ = 500.  # IO Frequency in Hz
    BACKGROUND_FREQ = 4.

    ordered_neuron_types = ['mossy_fibers', 'io_cell', 'glomerulus', "granule_cell", "golgi_cell",
                            "basket_cell", "stellate_cell", "purkinje_cell",
                            'dcn_cell_GABA', 'dcn_cell_Gly-I', 'dcn_cell_glut_large']
    neuron_types = []
    start_id_scaffold = []

    def __init__(self, tvb_simulator, nest_nodes_ids, nest_instance=None,
                 config=CONFIGURED, logger=None, path_to_network_source_file=""):
        super(CerebBuilder, self).__init__(tvb_simulator, nest_nodes_ids, nest_instance, config, logger)
        self.nest_nodes_ids = nest_nodes_ids
        self.path_to_network_source_file = path_to_network_source_file
        # Common order of neurons' number per population:
        self.population_order = 1  # we want scale to define exactly the number of neurons of each population
        self.modules_to_install = ["cereb"]
        if self.Zpos:
            self.Ie_pc = 176.26
        else:
            self.Ie_pc = 742.54

    def set_populations(self):
        # Populations' configurations
        self.neuron_types = list(self.net_src_file['cells/placement'].keys())
        ordered_neuron_types = []
        for neuron_type in self.ordered_neuron_types:
            ordered_neuron_types.append(self.neuron_types.pop(self.neuron_types.index(neuron_type)))
        ordered_neuron_types += self.neuron_types
        self.neuron_types = ordered_neuron_types
        self.populations = []
        self.start_id_scaffold = {}
        # All cells are modelled as E-GLIF models;
        # with the only exception of Glomeruli and Mossy Fibers
        # (not cells, just modeled as relays; i.e., parrot neurons)
        for neuron_name in self.neuron_types:
            if neuron_name != 'glomerulus' and neuron_name != 'mossy_fibers':
                model = 'eglif_cond_alpha_multisyn'
            else:
                model = 'parrot_neuron'
            n_neurons = np.array(self.net_src_file['cells/placement/' + neuron_name + '/identifiers'])[1]
            self.populations.append(
                {"label": neuron_name, "model": model,
                 "params": self.neuron_param.get(neuron_name, {}),
                 "scale": n_neurons,
                 "nodes": None})
            self.start_id_scaffold[neuron_name] = \
                np.array(self.net_src_file['cells/placement/' + neuron_name + '/identifiers'])[0]

    def set_populations_connections(self):
        self.default_populations_connection["conn_spec"]["rule"] = "one_to_one"
        self.populations_connections = []
        for conn_name in self.conn_weights.keys():
            conn = np.array(self.net_src_file['cells/connections/'+conn_name])
            pre = self.conn_pre_post[conn_name]["pre"]
            pre_dummy = np.ones(len(pre))
            if conn_name == "mossy_to_glomerulus":
                weights = pre_dummy * self.conn_weights[conn_name]
                delays = pre_dummy * self.conn_delays[conn_name]
            elif conn_name == "io_to_basket" or conn_name == "io_to_stellate":
                weights = pre_dummy * self.conn_weights[conn_name],
                delays = pre_dummy * self.nest_instance.random.normal(mean=self.conn_delays[conn_name]["mu"],
                                                                      std=self.conn_delays[conn_name]["sigma"])
            else:
                weights = pre_dummy * self.conn_weights[conn_name]
                delays = pre_dummy * self.conn_delays[conn_name]
            self.populations_connections.append(
                {"source": pre,
                 "target": self.conn_pre_post[conn_name]["post"],
                 "source_neurons": NeuronsFun(self.start_id_scaffold[self.conn_pre_post[conn_name]["pre"]],
                                                  conn[:, 0].flatten()),

                 "target_neurons": NeuronsFun(self.start_id_scaffold[self.conn_pre_post[conn_name]["post"]],
                                                  conn[:, 1].flatten()),
                 "synapse_model": 'static_synapse',
                 "conn_spec": self.default_populations_connection["conn_spec"],
                 "weight": weights,
                 "delay": delays,
                 "receptor_type": self.conn_receptors.get(conn_name, 0),
                 "nodes": None
                 }
            )

    def neurons_fun(self, population, total_neurons=100):
        # We use this in order to measure up to n_neurons neurons from every population
        n_neurons = len(population)
        if n_neurons > total_neurons:
            return population[0:-1:int(np.ceil(1.0 * n_neurons / total_neurons))]
        else:
            return population

    def set_spike_recorder(self):
        connections = OrderedDict()
        #          label <- target population
        for pop in self.populations:
            connections[pop["label"] + "_spikes"] = pop["label"]
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder"])
        params["record_to"] = self.output_devices_record_to
        device = {"model": "spike_recorder", "params": params,
                  # "neurons_fun": lambda node, population: self.neurons_fun(population),
                  "connections": connections, "nodes": None}  # None means all here
        return device

    def set_multimeter(self):
        connections = OrderedDict()
        #               label    <- target population
        for pop in self.populations:
            if pop["label"] != 'glomerulus' and pop["label"] != 'mossy_fibers':
                connections[pop["label"]] = pop["label"]
        params = dict(self.config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"])
        params["record_to"] = self.output_devices_record_to
        params["interval"] = self.monitor_period
        device = {"model": "multimeter", "params": params,
                  "neurons_fun": lambda node, population: self.neurons_fun(population),
                  "connections": connections, "nodes": None}  # None means all here
        return device

    def set_output_devices(self):
        # Creating  devices to be able to observe NEST activity:
        # Labels have to be different
        self.output_devices = [self.set_spike_recorder()]
        if self.RECORD_VM:
            self.output_devices.append(self.set_multimeter())

    def set_spike_stimulus(self):
        connections = OrderedDict()
        #             label <- target population
        connections["Stimulus"] = ['mossy_fibers']
        device = \
            {"model": "poisson_generator",
             "params": {"rate": self.STIM_FREQ, "origin": 0.0, "start": self.STIM_START, "stop": self.STIM_END},
             "connections": connections, "nodes": None,
             "weights": 1.0,
             "delays": 0.0,
             "receptor_type": 0}
        return device

    def set_spike_stimulus_background(self):
        connections = OrderedDict()
        #             label <- target population
        connections["Background"] = ['mossy_fibers']
        device = \
            {"model": "poisson_generator",
             "params": {"rate": self.BACKGROUND_FREQ, "origin": 0.0, "start": 0.0}, # not necessary: "stop": self.TOT_DURATION
             "connections": connections, "nodes": None,
             "weights": 1.0,
             "delays": 0.0,
             "receptor_type": 0}
        return device

    def set_input_devices(self):
        if self.STIMULUS:
            self.input_devices = [self.set_spike_stimulus(), self.set_spike_stimulus_background()]

    def set_defaults(self):
        if self.Zpos:
            self.Ie_pc = 176.26
        else:
            self.Ie_pc = 742.54
        self.net_src_file = h5py.File(self.path_to_network_source_file, 'r+')
        self.set_populations()
        self.set_populations_connections()
        self.set_output_devices()
        self.set_input_devices()
        self.net_src_file.close()

    def build(self, set_defaults=True):
        if set_defaults:
            self.set_defaults()
        return super(CerebBuilder, self).build()