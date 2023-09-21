# -*- coding: utf-8 -*-

from copy import deepcopy

from pandas import concat

from tvb_multiscale.core.utils.data_structures_utils import safe_dict_copy
from tvb_multiscale.core.spiking_models.builders.factory import build_and_connect_devices
from tvb_multiscale.core.spiking_models.builders.base import SpikingNetworkBuilder
from tvb_multiscale.core.spiking_models.devices import DeviceSets

from tvb_multiscale.tvb_annarchy.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_annarchy.annarchy_models.population import ANNarchyPopulation
from tvb_multiscale.tvb_annarchy.annarchy_models.region_node import ANNarchyRegionNode
from tvb_multiscale.tvb_annarchy.annarchy_models.brain import ANNarchyBrain
from tvb_multiscale.tvb_annarchy.annarchy_models.network import ANNarchyNetwork
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.annarchy_factory import \
    load_annarchy, assert_model, create_population, connect_two_populations, create_device, connect_device


LOG = initialize_logger(__name__, config=CONFIGURED)


class ANNarchyNetworkBuilder(SpikingNetworkBuilder):

    """This is the base class of a ANNarchyNetworkBuilder,
       which builds a ANNarchyNetwork from user configuration inputs.
       The builder is half way opinionated.
    """

    config = CONFIGURED
    _spiking_simulator_name = "annarchy_instance"
    modules_to_install = []
    _spiking_brain = ANNarchyBrain()
    _models_import_path = CONFIGURED.MYMODELS_IMPORT_PATH

    _input_proxies = DeviceSets()
    # input_proxies['Inhibitory']['rh-insula']

    def __init__(self, tvb_simulator, spiking_nodes_inds, spiking_simulator=None, config=None, logger=None):
        super(ANNarchyNetworkBuilder, self).__init__(tvb_simulator, spiking_nodes_inds, spiking_simulator,
                                                     config, logger)
        self._spiking_brain = ANNarchyBrain()

    @property
    def annarchy_instance(self):
        return self.spiking_simulator

    def __str__(self):
        return super(ANNarchyNetworkBuilder, self).__str__() + "\nannarchy simulator: %s" % str(self.annarchy_instance)

    def _configure_annarchy(self, **kwargs):
        if self.annarchy_instance is None:
            self.spiking_simulator = load_annarchy(self.config, self.logger)
            self.annarchy_instance.clear()  # This will restart ANNarchy!
            self.update_spiking_dt()
            self.update_default_min_delay()
            kwargs["dt"] = self.spiking_dt
            kwargs["seed"] = kwargs.pop("seed", self.config.ANNARCHY_SEED)
            kwargs["verbose"] = kwargs.pop("verbose", self.config.VERBOSE)
            self.annarchy_instance.setup(**kwargs)

    def configure(self, **kwargs):
        if self.config is None:
            self.config = CONFIGURED
        if self.logger is None:
            self.logger = initialize_logger(__name__, config=self.config)
        self._configure_annarchy()
        super(ANNarchyNetworkBuilder, self).configure()

    @property
    def min_delay(self):
        if self.annarchy_instance:
            return self.annarchy_instance.dt()
        else:
            return self.config.MIN_SPIKING_DT

    def set_synapse(self, syn_model, weights, delays, target, params=dict()):
        """Method to set the synaptic model, the weight, the delay,
           the synaptic target, and other possible synapse parameters
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
        return {'synapse_model': syn_model, 'weights': weights,
                'delays': delays, 'target': target, 'params': safe_dict_copy(params)}

    def _assert_model(self, model):
        return assert_model(model, self.annarchy_instance, self._models_import_path)

    def build_spiking_population(self, label, model, brain_region, size, params):
        """This methods builds an  ANNarchyPopulation instance,
           which represents a population of spiking neurons of the same neural model,
           and residing at a particular brain region node.
           Arguments:
            label: name (string) of the population
            model: name (string) of the neural model
            brain_region: name (string) of the brain reegion the population will reside
            size: number (integer) of the neurons of this population
            params: dictionary of parameters of the neural model to be set upon creation
           Returns:
            a ANNarchyPopulation class instance
        """
        params["name"] = label
        annarchy_population = create_population(model, self.annarchy_instance, size=size, params=params,
                                                import_path=self._models_import_path, config=self.config)
        return ANNarchyPopulation(annarchy_population, self.annarchy_instance,
                                  label=label, model=annarchy_population.neuron_type.name, brain_region=brain_region)

    def connect_two_populations(self, pop_src, src_inds_fun, pop_trg, trg_inds_fun, conn_spec, syn_spec):
        """Method to connect two ANNarchyPopulation instances in the SpikingNetwork.
           Arguments:
            source: the source ANNarchyPopulation of the connection
            src_inds_fun: a function that selects a subset of the souce population neurons
            target: the target ANNarchyPopulation of the connection
            trg_inds_fun: a function that selects a subset of the target population neurons
            conn_spec: a dict of parameters of the connectivity pattern among the neurons of the two populations,
                       excluding weight and delay ones
            syn_spec: a dict of parameters of the synapses among the neurons of the two populations,
                      including weight, delay and synaptic target ones
        """
        # Prepare the synaptic model:
        syn_spec = safe_dict_copy(syn_spec)
        syn_spec["synapse_model"] = \
            self._assert_model(
                syn_spec.pop("synapse_model",
                             syn_spec.pop("model",
                                          syn_spec.pop("synapse", None))))
        # Get connection arguments by copying conn_spec. Make sure to pop out the "method" entry:
        this_syn_spec = safe_dict_copy(syn_spec)
        this_conn_spec = safe_dict_copy(conn_spec)
        proj = connect_two_populations(pop_src, pop_trg, this_syn_spec.pop("weights"),
                                       this_syn_spec.pop("delays"), this_syn_spec.pop("target"),
                                       syn_spec=this_syn_spec, conn_spec=this_conn_spec,
                                       source_view_fun=src_inds_fun, target_view_fun=trg_inds_fun,
                                       name="%s -> %s" % (pop_src.label, pop_trg.label),
                                       annarchy_instance=self.annarchy_instance)
        # Add this projection to the source and target population inventories:
        pop_src.projections_pre.append(proj)
        pop_trg.projections_post.append(proj)

    def build_spiking_region_node(self, label="", input_node=None, *args, **kwargs):
        """This methods builds a ANNarchyRegionNode instance,
           which consists of all ANNarchyPopulation instances, residing at a particular brain region node.
           Arguments:
            label: name (string) of the region node. Default = ""
            input_node: an already created SpikingRegionNode() class. Default = None.
            *args, **kwargs: other optional positional or keyword arguments
           Returns:
            a ANNarchyRegionNode class instance
        """
        return ANNarchyRegionNode(input_node, label=label)

    def build_and_connect_input_devices(self):
        """Method to build and connect input devices, organized by
           - the variable they stimulate, and the
           - population(s), and
           - brain region nodes they target."""
        _devices = []
        for device in self._input_devices:
            device["input_proxies"] = self._input_proxies
            LOG.info("Generating and connecting %s -> %s device set of model %s\n"
                     "for nodes %s..." % (str(list(device["connections"].keys())),
                                          str(list(device["connections"].values())),
                                          device["model"], str(device["nodes"])))
            _devices.append(self.build_and_connect_devices(device))
        if len(_devices):
            return DeviceSets(concat(_devices), name="input_devices")
        else:
            return DeviceSets(name="input_devices")

    def build_and_connect_devices(self, devices):
        """Method to build and connect input or output devices, organized by
           - the variable they measure or stimulate, and the
           - population(s), and
           - brain region nodes they target.
           See tvb_multiscale.core.spiking_models.builders.factory,
           and tvb_multiscale.tvb_annarchy.annarchy_models.builders.annarchy_factory.
        """
        return build_and_connect_devices(devices, create_device, connect_device,
                                         self._spiking_brain, self.config, annarchy_instance=self.annarchy_instance,
                                         import_path=self._models_import_path)

    def build_spiking_network(self):
        """A method to build the final ANNarchyNetwork class based on the already created constituents."""
        return ANNarchyNetwork(annarchy_instance=self.annarchy_instance, brain_regions=self._spiking_brain,
                               output_devices=self._output_devices, input_devices=self._input_devices,
                               input_proxies=self._input_proxies, config=self.config)
