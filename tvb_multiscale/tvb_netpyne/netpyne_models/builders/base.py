# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np

from tvb_multiscale.core.spiking_models.builders.base import SpikingNetworkBuilder
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices

from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory import load_netpyne
from tvb_multiscale.tvb_netpyne.netpyne_models.network import NetpyneNetwork
from tvb_multiscale.tvb_netpyne.netpyne_models.population import NetpynePopulation
from tvb_multiscale.tvb_netpyne.netpyne_models.region_node import NetpyneRegionNode
from tvb_multiscale.tvb_netpyne.netpyne_models.brain import NetpyneBrain
from tvb_multiscale.tvb_netpyne.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory import create_device, connect_device


class NetpyneNetworkBuilder(SpikingNetworkBuilder):

    config = CONFIGURED
    _spiking_simulator_name = "netpyne_instance"
    modules_to_install = list()
    _spiking_brain = NetpyneBrain()

    def __init__(self, tvb_simulator=dict(), spiking_nodes_inds=list(),
                 spiking_simulator=None, config=CONFIGURED, logger=None):
        self.config = config
        if self.config is None:
            self.config = CONFIGURED
        # Beware: this method can be called multiple times (first - when creating default object)
        super(NetpyneNetworkBuilder, self).__init__(tvb_simulator, spiking_nodes_inds,
                                                    spiking_simulator, self.config, logger)

        self._spiking_brain = NetpyneBrain()
        self.modules_to_install = list()

    @property
    def netpyne_instance(self):
        return self.spiking_simulator

    def configure(self, netParams, simConfig, autoCreateSpikingNodes=True):
        if self.netpyne_instance is None:
            self.spiking_simulator = load_netpyne(self.config)

        if self.netpyne_instance is None:
            self.spiking_simulator = load_netpyne(self.config)

        super(NetpyneNetworkBuilder, self).configure()
        self.netpyne_instance.autoCreateSpikingNodes = autoCreateSpikingNodes
        self.netpyne_instance.importModel(netParams, simConfig, self.spiking_dt, self.config)

    @abstractmethod
    def proxy_node_synaptic_model_funcs(self):
        pass

    def set_synapse(self, syn_model, weight, delay, receptor_type, params=dict()):
        """Method to set the synaptic model, the weight, the delay,
           the synaptic receptor type, and other possible synapse parameters
           to a synapse_params dictionary.
           Arguments:
            - syn_model: the name (string) of the synapse model
            - weight: the weight of the synapse
            - delay: the delay of the connection,
            - receptor_type: the receptor type
            - params: a dict of possible synapse parameters
           Returns:
            a dictionary of the whole synapse configuration
        """
        # populate these values to be read in `connect_two_populations` below
        syn_spec = {'synapse_model': syn_model, 'weight': weight, 'delay': delay, 'receptor_type': receptor_type}
        syn_spec.update(params)
        return syn_spec

    def build_spiking_population(self, label, model, brain_region, size, params):
        """This methods builds a NetpynePopulation instance,
           which represents a population of spiking neurons of the same neural model,
           and residing at a particular brain region node.
           Arguments:
            label: name (string) of the population
            model: name (string) of the neural model
            size: number (integer) of the neurons of this population
            params: dictionary of parameters of the neural model to be set upon creation
           Returns:
            a NetpynePopulation class instance
        """
        size = int(np.round(size))

        global_label = params.get("global_label")
        population = NetpynePopulation(None, self.netpyne_instance, label, global_label, brain_region)

        if self.netpyne_instance.autoCreateSpikingNodes:
            print(f"Netpyne:: Creating spiking population '{population.global_label}' of {size} neurons of type '{model}'.")
            self.netpyne_instance.registerPopulation(population.global_label, model, size)
        return population

    def connect_two_populations(self, pop_src, src_inds_fun, pop_trg, trg_inds_fun, conn_spec, syn_spec):
        """Method to connect two NetpynePopulation instances in the SpikingNetwork.
           Arguments:
            source: the source NetpynePopulation of the connection
            src_inds_fun: a function that selects a subset of the souce population neurons
            target: the target NetpynePopulation of the connection
            trg_inds_fun: a function that selects a subset of the target population neurons
            conn_params: a dict of parameters of the connectivity pattern among the neurons of the two populations,
                         excluding weight and delay ones
            synapse_params: a dict of parameters of the synapses among the neurons of the two populations,
                            including weight, delay and synaptic receptor type ones
        """
        # TODO: Should we also use src_inds_fun, trg_inds_fun? (see NEST for reference)
        src = pop_src.global_label
        trg = pop_trg.global_label

        rule = conn_spec["rule"]
        if rule == "all_to_all":
            prob = 1.0
        else:
            prob = rule["prob"]
        self.netpyne_instance.interconnectSpikingPopulations(src, trg,
                                                             syn_spec["receptor_type"], syn_spec["weight"],
                                                             syn_spec["delay"], prob)

    def build_spiking_region_node(self, label="", input_node=None, *args, **kwargs):
        """This methods builds a NetpyneRegionNode instance,
           which consists of a pandas.Series of all SpikingPopulation instances,
           residing at a particular brain region node.
           Arguments:
            label: name (string) of the region node. Default = ""
            input_node: an already created SpikingRegionNode() class. Default = None.
            *args, **kwargs: other optional positional or keyword arguments
           Returns:
            a SpikingRegionNode class instance
        """
        return NetpyneRegionNode(input_node, label=label)

    def build_and_connect_devices(self, devices):
        """Method to build and connect input or output devices, organized by
           - the variable they measure or stimulate (pandas.Series), and the
           - population(s) (pandas.Series), and
           - brain region nodes (pandas.Series) they target.
           See tvb_multiscale.core.spiking_models.builders.factory
           and tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory"""
        return build_and_connect_devices(devices, create_device, connect_device,
                                         self._spiking_brain, self.config, netpyne_instance=self.netpyne_instance)

    def build_spiking_network(self):
        """A method to build the final NetpyneNetwork class based on the already created constituents."""
        return NetpyneNetwork(self.netpyne_instance, brain_regions=self._spiking_brain,
                              output_devices=self._output_devices, input_devices=self._input_devices,
                              config=self.config)
