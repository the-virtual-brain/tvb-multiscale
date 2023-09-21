# -*- coding: utf-8 -*-

import time
import os
import glob
from shutil import copyfile
import h5py

import numpy as np
from pandas import Series

from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import configure_nest_kernel
from tvb_multiscale.tvb_nest.nest_models.builders.base import NESTNetworkBuilder
from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
from tvb_multiscale.tvb_nest.nest_models.brain import NESTBrain
from tvb_multiscale.tvb_nest.nest_models.region_node import NESTRegionNode
from tvb_multiscale.tvb_nest.nest_models.population import NESTPopulation
from tvb_multiscale.tvb_nest.nest_models.devices import NESTPoissonGenerator, NESTSpikeRecorder, NESTMultimeter

from tvb_multiscale.core.spiking_models.devices import DeviceSet

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


def copy_network_source_file(path_to_network_source_file):
    copypath, extension = os.path.splitext(path_to_network_source_file)
    while len(glob.glob(copypath+"*")):
        copypath += "_copy"
    copypath += extension
    copyfile(path_to_network_source_file, copypath)
    return copypath


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

    if Zpos:
        Ie_pc = 176.26
    else:
        Ie_pc = 742.54

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
    BACKGROUND = True
    STIMULUS = True
    TOT_DURATION = 400.   # ms
    STIM_MF_START = 140.  # beginning of stimulation to MFs
    STIM_MF_END = 300.    # end of stimulation to MFs
    STIM_MF_FREQ = 40.    # MF Frequency in Hz
    STIM_IO_START = 290.  # beginning of stimulation to IO
    STIM_IO_END = 300.    # end of stimulation to IO
    STIM_IO_FREQ = 500.   # IO Frequency in Hz
    BACKGROUND_FREQ = 4.  # Background Frequency in Hz

    ordered_populations_labels = ['mossy_fibers', 'glomerulus', "granule_cell", "golgi_cell",
                                  'io_cell', "basket_cell", "stellate_cell", "purkinje_cell",
                                  'dcn_cell_GABA', 'dcn_cell_Gly-I', 'dcn_cell_glut_large']
    start_id_scaffold = []

    modules_to_install = ["cereb"]

    def __init__(self, tvb_simulator={}, spiking_simulator=None, config=None, logger=None,
                 pops_to_nodes_inds={}, regions_inds_to_regions_labels={}, path_to_network_source_file=""):
        self.pops_to_nodes_inds = pops_to_nodes_inds
        self.regions_inds_to_regions_labels = regions_inds_to_regions_labels
        super(CerebBuilder, self).__init__(tvb_simulator, np.unique(list(self.pops_to_nodes_inds.values())),
                                           spiking_simulator, config, logger)
        self.path_to_network_source_file = path_to_network_source_file

    def _initialize(self):
        self.spiking_nodes_inds = np.unique(list(self.pops_to_nodes_inds.values()))
        if not os.path.isfile(self.path_to_network_source_file):
            package_folder = __file__.split("tvb_multiscale")[0]
            self.path_to_network_source_file = \
                os.path.join(package_folder, "examples/data/cerebellum/300x_200z_DCN_IO.hdf5")
        # Common order of neurons' number per population:
        self.population_order = 1  # we want scale to define exactly the number of neurons of each population
        self.modules_to_install = ["cereb"]
        if self.Zpos:
            self.Ie_pc = 176.26
        else:
            self.Ie_pc = 742.54
        self.neuron_param["purkinje_cell"]["I_e"] = self.Ie_pc
        self.nest_network = NESTNetwork(self.nest_instance)
        self.nest_network.brain_regions = NESTBrain()
        self.nest_network.input_devices = Series()
        self.nest_network.output_devices = Series()

    def get_populations_labels_from_file(self):
        return list(self.net_src_file['cells/placement'].keys())

    def get_n_neurons_from_file(self, pop_name):
        return np.array(self.net_src_file['cells/placement/' + pop_name + '/identifiers'])[1]

    def get_start_id_scaffold_from_file(self, pop_name):
        return np.array(self.net_src_file['cells/placement/' + pop_name + '/identifiers'])[0]

    def get_connection_from_file(self, conn_name):
        return np.array(self.net_src_file['cells/connections/'+conn_name])

    def assert_ordered_populations_labels(self):
        pop_labels = self.get_populations_labels_from_file()
        assert np.all([pop_label in self.ordered_populations_labels for pop_label in pop_labels])

    def get_pop_model(self, pop_name):
        if pop_name != 'glomerulus' and pop_name != 'mossy_fibers':
            return 'eglif_cond_alpha_multisyn'
        else:
            return 'parrot_neuron'

    def _populations_to_regions(self, pop_name):
        return [self.regions_inds_to_regions_labels[reg_ind]
                for reg_ind in ensure_list(self.pops_to_nodes_inds[pop_name])]

    def build_populations(self):
        # Populations' configurations
        self.assert_ordered_populations_labels()
        self.populations = {}
        self.start_id_scaffold = {}
        # All cells are modelled as E-GLIF models;
        # with the only exception of Glomeruli and Mossy Fibers
        # (not cells, just modeled as relays; i.e., parrot neurons)
        self.logger.info("Creating populations:\n%s..." % str(self.ordered_populations_labels))
        self.start_id_scaffold = {}
        for node_id, node_label in zip(self.spiking_nodes_inds, self.spiking_nodes_labels):
            self.nest_network.brain_regions[node_label] = NESTRegionNode(label=node_label)
            self.start_id_scaffold[node_label] = {}
            for pop_name in self.ordered_populations_labels:
                if self.pops_to_nodes_inds[pop_name] == node_id:
                    model = self.get_pop_model(pop_name)
                    n_neurons = self.get_n_neurons_from_file(pop_name)
                    self.nest_network.brain_regions[node_label][pop_name] = \
                        NESTPopulation(self.nest_instance.Create(model, n_neurons,
                                                                 params=self.neuron_param.get(pop_name, {}),),
                                       self.nest_instance, label=pop_name, brain_region=node_label)
                    self.logger.info("...created %d neurons of model %s for population %s in brain region %s..." %
                                     (n_neurons, model, pop_name, node_label))
                    self.start_id_scaffold[node_label][pop_name] = self.get_start_id_scaffold_from_file(pop_name)

    def _get_scaffold(self, conn, pop_name, reg_name):
        return tuple(np.array(conn
                              - self.start_id_scaffold[reg_name][pop_name]
                              + self.nest_network.brain_regions[reg_name][pop_name].gids[0]).astype('i').tolist())

    def build_populations_connections(self):
        self.default_populations_connection["conn_spec"]["rule"] = "one_to_one"
        self.populations_connections = []
        self.logger.info("Connecting within brain regions populations...")
        tic1 = time.time()
        for conn_name in self.conn_weights.keys():
            tic = time.time()
            conn = self.get_connection_from_file(conn_name)
            source = self.conn_pre_post[conn_name]["pre"]
            target = self.conn_pre_post[conn_name]["post"]
            reg_name = self.regions_inds_to_regions_labels[self.pops_to_nodes_inds[source]]
            if reg_name == self.regions_inds_to_regions_labels[self.pops_to_nodes_inds[target]]:
                pre = self._get_scaffold(conn[:, 0], source, reg_name)
                n_pre = len(pre)
                pre_dummy = np.ones((n_pre,))
                post = self._get_scaffold(conn[:, 1], target, reg_name)
                if conn_name == "mossy_to_glomerulus":
                    weights = pre_dummy * self.conn_weights[conn_name]
                    delays = pre_dummy * self.conn_delays[conn_name]
                else:
                    weights = pre_dummy * self.conn_weights[conn_name]
                    delays = pre_dummy * self.conn_delays[conn_name]
                receptor = self.conn_receptors.get(conn_name, 0)
                self.logger.info("...connecting populations %s (%s) -> %s (%s) (%s connections) at receptor %d..." %
                                 (source, reg_name, target, reg_name, n_pre, receptor))
                self.nest_instance.Connect(pre, post,
                                           conn_spec={"rule": self.default_populations_connection["conn_spec"]["rule"]},
                                           syn_spec={"synapse_model": "static_synapse",
                                                     "weight": weights, "delay": delays,
                                                     "receptor_type": receptor})

                self.logger.info("...done in %g sec..." % (time.time() - tic))
        self.logger.info("...done in %g sec..." % (time.time() - tic1))

    def build_nodes_connections(self):
        self.default_nodes_connection["conn_spec"]["rule"] = "one_to_one"
        self.nodes_connections = []
        self.logger.info("Connecting among brain regions populations...")
        tic1 = time.time()
        for conn_name in self.conn_weights.keys():
            tic = time.time()
            conn = self.get_connection_from_file(conn_name)
            source = self.conn_pre_post[conn_name]["pre"]
            target = self.conn_pre_post[conn_name]["post"]
            reg_source = self.regions_inds_to_regions_labels[self.pops_to_nodes_inds[source]]
            reg_target = self.regions_inds_to_regions_labels[self.pops_to_nodes_inds[target]]
            if reg_source != reg_target:
                pre = self._get_scaffold(conn[:, 0], source, reg_source)
                n_pre = len(pre)
                pre_dummy = np.ones((n_pre,))
                post = self._get_scaffold(conn[:, 1], target, reg_target)
                self.logger.info("...connecting populations %s (%s) -> %s (%s) (%s connections)..." %
                                 (source, reg_source, target, reg_target, n_pre))
                if conn_name == "io_to_basket" or conn_name == "io_to_stellate":
                    weights = pre_dummy * self.conn_weights[conn_name],
                    delays = self.nest_instance.random.normal(mean=self.conn_delays[conn_name]["mu"],
                                                              std=self.conn_delays[conn_name]["sigma"])
                else:
                    weights = pre_dummy * self.conn_weights[conn_name]
                    delays = pre_dummy * self.conn_delays[conn_name]
                receptor = self.conn_receptors.get(conn_name, 0)
                self.logger.info("...connecting populations %s (%s) -> %s (%s) (%s connections) at receptor %d..." %
                                 (source, reg_source, target, reg_target, n_pre, receptor))
                self.nest_instance.Connect(pre, post,
                                           conn_spec={"rule": self.default_nodes_connection["conn_spec"]["rule"]},
                                           syn_spec={"synapse_model": "static_synapse",
                                                     "weight": weights, "delay": delays,
                                                     "receptor_type": receptor})
                self.logger.info("...done in %g sec..." % (time.time() - tic))
        self.logger.info("...done in %g sec..." % (time.time() - tic1))

    def neurons_fun(self, population, total_neurons=100):
        # We use this in order to measure up to n_neurons neurons from every population
        n_neurons = population.number_of_neurons
        if n_neurons > total_neurons:
            return population._nodes[0:-1:np.maximum(1, int(np.round(1.0 * n_neurons / total_neurons)))]
        else:
            return population._nodes

    def build_spike_recorder(self):
        self.logger.info("...building and connecting spike recorders...")
        for pop_name in self.ordered_populations_labels:
            self.logger.info("...to %s..." % pop_name)
            self.nest_network.output_devices[pop_name] = DeviceSet(label=pop_name, model="spike_recorder")
            for reg_name in self._populations_to_regions(pop_name):
                self.logger.info("...at region %s..." % reg_name)
                label = "%s_%s" % (pop_name, reg_name)
                self.nest_network.output_devices[pop_name][reg_name] = \
                    NESTSpikeRecorder(self.nest_instance.Create('spike_recorder',
                                                                params={"record_to": self.output_devices_record_to,
                                                                        "label": label}),
                                      nest_instance=self.nest_instance,
                                      label=label, brain_region=reg_name)
                self.nest_instance.Connect(self.nest_network.brain_regions[reg_name][pop_name]._nodes,  # self.neurons_fun()
                                           self.nest_network.output_devices[pop_name][reg_name].device)
            self.nest_network.output_devices[pop_name].update()

    def build_multimeter(self):
        self.logger.info("...building and connecting multimeters...")
        for pop_name in self.ordered_populations_labels:
            if pop_name != 'glomerulus' and pop_name != 'mossy_fibers':
                dev_name = pop_name + "_ts"
                self.logger.info("...to %s..." % pop_name)
                self.nest_network.output_devices[dev_name] = DeviceSet(label=dev_name, model="multimeter")
                for reg_name in self._populations_to_regions(pop_name):
                    self.logger.info("...at region %s..." % reg_name)
                    label = "%s_%s" % (dev_name, reg_name)
                    self.nest_network.output_devices[dev_name][reg_name] = \
                        NESTMultimeter(self.nest_instance.Create('multimeter',
                                                                 params={"record_from": ["V_m"],
                                                                         "record_to": self.output_devices_record_to,
                                                                         "label": label}),
                                       nest_instance=self.nest_instance,
                                       label=label, brain_region=reg_name)
                    self.nest_instance.Connect(self.nest_network.output_devices[dev_name][reg_name].device,
                                               self.nest_network.brain_regions[reg_name][pop_name]._nodes) #self.neurons_fun()
                self.nest_network.output_devices[dev_name].update()

    def build_output_devices(self):
        self.logger.info("Building output devices...")
        # Creating  devices to be able to observe NEST activity:
        # Labels have to be different
        self.build_spike_recorder()
        if self.RECORD_VM:
            self.build_multimeter()

    def build_spike_stimulus_mf(self):
        self.logger.info("...building and connecting mossy fibers stimulus...")
        self.nest_network.input_devices["STIM_MF"] = DeviceSet(label="STIM_MF", model="poisson_generator")
        for reg_name in self._populations_to_regions("mossy_fibers"):
            self.logger.info("...at region %s..." % reg_name)
            self.nest_network.input_devices["STIM_MF"][reg_name] = \
                NESTPoissonGenerator(self.nest_instance.Create('poisson_generator',
                                                               params={'rate': self.STIM_MF_FREQ,
                                                                       'start': self.STIM_MF_START,
                                                                       'stop': self.STIM_MF_END}),
                                     nest_instance=self.nest_instance,
                                     label="STIM_MF", brain_region=reg_name)
            self.nest_instance.Connect(self.nest_network.input_devices["STIM_MF"][reg_name].device,
                                       self.nest_network.brain_regions[reg_name]["mossy_fibers"]._nodes,
                                       conn_spec={"rule": "all_to_all"},
                                       syn_spec={"synapse_model": "static_synapse", "weight": 1.0,
                                                 "delay": self.default_min_delay, "receptor_type": 0}
                                       )
        self.nest_network.input_devices["STIM_MF"].update()

    def select_microzone_negative(self, neurons):
        # the current NEST IO belongs to the Z- microzone
        io_neurons = []
        for io in neurons:
            if io - neurons + self.start_id_scaffold[self._populations_to_regions['io_cell'][0]]['io_cell'] in \
                    self.net_src_file['labels/placement/microzone-negative']:  # the current NEST IO belongs to the Z- microzone
                io_neurons.append(io)
        return self.nest_instance.NodeCollection(io_neurons)

    def build_spike_stimulus_io(self):
        self.logger.info("...building and connecting inferior olive stimulus...")
        self.nest_network.input_devices["STIM_IO"] = DeviceSet(label="STIM_IO", model="poisson_generator")
        for reg_name in self._populations_to_regions("io_cell"):
            self.logger.info("...at region %s..." % reg_name)
            self.nest_network.input_devices["STIM_IO"][reg_name] = \
                NESTPoissonGenerator(self.nest_instance.Create('poisson_generator',
                                                               params={'rate': self.STIM_IO_FREQ,
                                                                       'start': self.STIM_IO_START,
                                                                       'stop': self.STIM_IO_END}),
                                     nest_instance=self.nest_instance,
                                     label="STIM_IO", brain_region=reg_name)
            self.nest_instance.Connect(self.nest_network.input_devices["STIM_IO"][reg_name].device,
                                  # self.select_microzone_negative(self.nest_network.brain_regions[reg_name]["io_cell"])
                                       self.nest_network.brain_regions[reg_name]["io_cell"]._nodes,
                                       conn_spec={"rule": "all_to_all"},
                                       syn_spec={"synapse_model": "static_synapse", "weight": 25.0,
                                                 "delay": self.default_min_delay, "receptor_type": 1})
        self.nest_network.input_devices["STIM_IO"].update()

    def build_spike_stimulus_background(self):
        self.logger.info("...building and connecting background stimulus...")
        self.nest_network.input_devices["BACKGROUND"] = DeviceSet(label="BACKGROUND", model="poisson_generator")
        for reg_name in self._populations_to_regions("mossy_fibers"):
            self.logger.info("...at region %s..." % reg_name)
            self.nest_network.input_devices["BACKGROUND"][reg_name] = \
                NESTPoissonGenerator(self.nest_instance.Create('poisson_generator',
                                                               params={'rate': self.BACKGROUND_FREQ,
                                                                       'start': self.spiking_dt,
                                                                       'stop': self.TOT_DURATION}),
                                     nest_instance=self.nest_instance,
                                     label="BACKGROUND", brain_region=reg_name)
            self.nest_instance.Connect(self.nest_network.input_devices["BACKGROUND"][reg_name].device,
                                       self.nest_network.brain_regions[reg_name]["mossy_fibers"]._nodes,
                                       conn_spec={"rule": "all_to_all"},
                                       syn_spec={"synapse_model": "static_synapse", "weight": 1.0,
                                                 "delay": self.default_min_delay, "receptor_type": 0}
                                       )
        self.nest_network.input_devices["BACKGROUND"].update()

    def build_input_devices(self):
        self.logger.info("Building input devices...")
        if self.BACKGROUND:
            self.build_spike_stimulus_background()
        if self.STIMULUS:
            self.build_spike_stimulus_mf()
            self.build_spike_stimulus_io()

    def configure(self):
        self.nest_instance = configure_nest_kernel(self.nest_instance, self.config)
        self._initialize()
        super(CerebBuilder, self).configure()
        self.compile_install_nest_modules(self.modules_to_install)

    def build(self, set_defaults=True):
        # We create a copy so that the original can be read by another process while it is open here:
        if set_defaults:
            copypath = copy_network_source_file(self.path_to_network_source_file)
            self.net_src_file = h5py.File(copypath, 'r+')
            self.build_populations()
            self.build_populations_connections()
            self.build_nodes_connections()
            self.build_output_devices()
            self.build_input_devices()
            self.net_src_file.close()
            # Remove copy:
            os.remove(copypath)
        return self.nest_network
