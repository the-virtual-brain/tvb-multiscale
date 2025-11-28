# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

import ANNarchy

from tvb.basic.neotraits.api import Attr, Int, List

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list, is_integer, extract_integer_intervals

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.spiking_models.population import SpikingPopulation


class _ANNarchyPopulation(HasTraits):

    """ANNarchyPopulation class
       Wraps around an ANNarchy.core.Population class and
       represents a population of neurons of the same neural model,
       residing at the same brain region.
    """

    annarchy_network = Attr(field_type=ANNarchy.Network, required=False,
                            label="ANNarchy.Network", doc="""Instance of ANNarchy.Network""")

    _nodes = Attr(field_type=Population, default=None, required=False,
                  label="ANNarchy.core.Population",
                  doc="""Instance of ANNarchy.core.Population""")

    label = Attr(field_type=str, default="", required=True,
                 label="Population label", doc="""Label of ANNarchy.core.Population""")

    model = Attr(field_type=str, default="", required=True, label="Population model",
                 doc="""Label of neuronal model of ANNarchy.core.Population's neurons""")

    brain_region = Attr(field_type=str, default="", required=True, label="Brain region",
                        doc="""Label of the brain region the ANNarchy.core.Population resides""")

    projections_pre = []
    # List(of=Projection, default=(), label="Outgoing projections",
    #                        doc="""A list of population's outgoing ANNarchy.Projection instances""")

    projections_post = []
    # List(of=Projection, default=(), label="Incoming projections",
    #                         doc="""A list of population's incoming ANNarchy.Projection instances""")

    _source_conns_attr = "pre"
    _target_conns_attr = "post"
    _weight_attr = "w"
    _delay_attr = "delay"
    _receptor_attr = "target"
    _default_connection_attrs = [_weight_attr, _delay_attr, _receptor_attr]

    def __init__(self, nodes=None, annarchy_network=None, **kwargs):
        if isinstance(annarchy_network, ANNarchy.Network):
            self.annarchy_network = annarchy_network
        self._nodes = nodes
        label = kwargs.get("label", "")
        if self._nodes is not None:
            if len(label):
                self._nodes.name = label
            else:
                label = self._nodes.name
        self.label = label
        kwargs["label"] = label
        self.model = str(kwargs.get("model", self.annarchy_model))
        self.brain_region = str(kwargs.get("brain_region", ""))
        self.projections_pre = []
        self.projections_post = []
        HasTraits.__init__(self)

    @property
    def spiking_simulator_module(self):
        return ANNarchy

    def _assert_annarchy_network(self):
        assert isinstance(self.annarchy_network, ANNarchy.Network)

    @property
    def annarchy_model(self):
        if isinstance(self._nodes, ANNarchy.core.Population):
            return str(self._nodes.neuron_type.name)
        else:
            return str(self.__class__.__name__)

    @property
    def geometry(self):
        return self._nodes.geometry

    @property
    def population_ind(self):
        if isinstance(self._nodes, ANNarchy.core.Population):
            return self._nodes.id

    @property
    def gids(self):
        """Method to get a sequence (list, tuple, array) of the individual gids of nodes's elements"""
        return self._nodes.ranks

    @property
    def neurons(self):  # tuple of populations' neurons
        """Method to get all neurons' indices of this population.
           Returns:
            array of neurons' global indices.
            In ANNarchy: So far we get only local indices.
            We form global indices by zipping local indices with the global population indice.
        """
        local_inds = self._nodes.ranks
        return np.array(list(zip([self.population_ind] * len(local_inds), local_inds)))

    def _assert_nodes(self, nodes=None):
        """Method to assert an input set of neurons either as:
            - the present instance of ANNarchy.core.Population class
            - a ANNarchy.core.PopulationView.PopulationView instance
              of the present instance of ANNarchy.core.Population class
            - a collection (tuple, list, numpy.ndarray)
              of global indices (i.e., tuples of (population_inds, neuron_ind),
              of the present instance of ANNarchy.core.Population class, or of local indices thereof,
            Default input=None, which corresponds to the present instance of ANNarchy.core.Population class.
        """
        if nodes is None:
            nodes = self._nodes
        else:
            if isinstance(nodes, ANNarchy.core.Population):
                # Assert that we refer to this object's Population
                assert self._nodes == nodes
            elif isinstance(nodes, ANNarchy.core.Population.PopulationView):
                # Assert that we refer to a view of this object's Population
                assert self._nodes == nodes.population
            else:
                # Let's check if these are global or local indices of neurons...
                local_inds = []
                for node in ensure_list(nodes):
                    if isinstance(node, (tuple, list)):
                        # If neurons are global_ids formed as tuples of (population_ind, neuron_ind)...
                        if node[0] == self.population_ind:
                            # ... confirm that the population_ind is correct and get the neuron_ind
                            local_inds.append(node[1])
                            # If neurons are just local inds, gather them...
                        elif is_integer(node):
                            local_inds.append(node)
                        else:
                            raise ValueError(
                                "neurons %s\nis neither an instance of ANNarchy.core.Population.Population, "
                                "nor of  ANNarchy.core.PopulationView.PopulationView,\n"
                                "nor is it a collection (tuple, list, or numpy.ndarray) "
                                "of global (tuple of (population_inds, neuron_ind) or local indices of neurons!")
                        # Return a Population View:
                        nodes = self._nodes[local_inds]
        return nodes

    def _Set(self, values_dict, neurons=None):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            neurons: instance of a ANNarchy.core.Population.Population
                     or ANNarchy.core.Population.PopulationView class,
                     or a collection (tuple, list, numpy.ndarray) of global indices
                     (i.e., tuples of (population_inds, neuron_ind),
                     of the present instance of ANNarchy.core.Population.Population class, or of local indices thereof,
        """
        self._assert_nodes(neurons).set(values_dict)

    def _Get(self, attrs=None, neurons=None):
        """Method to get attributes of the SpikingPopulation's neurons.
           Arguments:
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all attributes
            neurons: instance of a ANNarchy.core.Population.Population
                     or ANNarchy.core.PopulationView.PopulationView class,
                     or a collection (tuple, list, numpy.ndarray) of global indices
                     (i.e., tuples of (population_inds, neuron_ind),
                     of the present instance of ANNarchy.core.Population.Population class, or of local indices thereof,
           Returns:
            Dictionary of numpy.arrays of neurons' attributes.
        """
        dictionary = {}
        neurons = self._assert_nodes(neurons)
        if attrs is None:
            if self.model.lower().find("timed") > -1:
                attrs = ["rates", "schedule", "period"]
            elif self.model.lower().find("PoissonPopulation".lower()) > -1:
                attrs = ["rates", "target"]
            elif self.model.lower().find("HomogeneousCorrelatedSpikeTrains".lower()) > -1:
                attrs = ["rates", "corr", "tau", "schedule", "period", "mu", "sigma", "refractory"]
            elif self.model.lower().find("SpikeSourceArray".lower()) > -1:
                attrs = ["spike_times"]
            else:
                attrs = neurons.attributes
        for attribute in attrs:
            try:
                dictionary[attribute] = neurons.get(attribute)
            except:
                pass
        return dictionary

    def _get_projections(self, pre_or_post, neurons):
        """Get the projections of this populations.
           Arguments:
            pre_or_post: "pre" or "post" to choose the corresponding connections
            neurons: an ANNarchy.core.Population.Population
                    or ANNarchy.core.PopulationView.PopulationView to filter the connections returned
           Return:
            a list of ANNarchy.Projection instances
        """
        projections = []
        if neurons is not None:
            for proj in getattr(self, "projections_%s" % pre_or_post):
                if getattr(proj, pre_or_post) == neurons:
                    projections.append(proj)
        else:
            projections = getattr(self, "projections_%s" % pre_or_post)
        return projections

    def _GetConnections(self, neurons=None, source_or_target=None):
        """Method to get all the connections from/to a SpikingPopulation neuron.
        Arguments:
            neurons: ANNarchy.core.Population.Population
                     or ANNarchy.core.PopulationView.PopulationView or sequence (tuple, list, array) of neurons
                     the connections of which should be included in the output.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            A list of Projections' objects.
        """
        neurons = self._assert_nodes(neurons)
        if source_or_target not in ["source", "target"]:
            return self._get_projections("pre", neurons), self._get_projections("post", neurons)
        elif source_or_target == "source":
            return self._get_projections("pre", neurons)
        elif source_or_target == "target":
            return self._get_projections("post", neurons)

    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
             values_dict: dictionary of attributes names' and values.
             connections: a Projection object or a collection (list, tuple, array) thereof.
                          Default = None, corresponding to all connections to/from the present population.
        """
        if connections is None:
            connections = self._GetConnections()
        if isinstance(connections, tuple):
            if len(connections) == 1:
                connections = connections[0]
            else:
                # In case we deal with both pre and post connections, treat them separately:
                for connection in connections:
                    self._SetToConnections(values_dict, connection)
                return
        for connection in ensure_list(connections):
            if connection in self.projections_pre or connection in self.projections_post:
                # connection.set(values_dict) <- this would be straightforward, but can generate
                # arbitrary attributes that get ignored by the projection but are readable with get()
                # TODO: figure out why this is bad, if it doesn't cause an error/Exception!
                for attr, value in values_dict.items():
                    if hasattr(connection, attr):
                        connection.set({attr: value})
                    else:
                        raise AttributeError("Projection %s has no attribute named %s." %(connection.name, attr))
            else:
                raise AttributeError("No incoming projection %s associated to this %s of model %s with label %s." %
                                 (connection.name, self.__class__.__name__, self.model, self.label))

    def _set_attributes_of_connection_to_dict(self, dictionary, connection, attribute):
        if attribute in dictionary.keys():
            dictionary[attribute].append(connection.get(attribute))
        else:
            dictionary[attribute] = [connection.get(attribute)]

    def _GetFromConnections(self, attrs=None, connections=None):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
            Arguments:
             connections: a Projection object or a collection (list, tuple, array) thereof.
                          Default = None, corresponding to all connections to/from the present population.
             attrs: collection (list, tuple, array) of the attributes to be included in the output.
                    Default = None, corresponds to all attributes
            Returns:
             Dictionary of lists (for the possible different Projection objects) of arrays of connections' attributes.
        """
        if connections is None:
            connections = self._GetConnections()
        if isinstance(connections, tuple):
            if len(connections) == 1:
                connections = connections[0]
            else:
                # In case we deal with both pre and post connections, treat them separately:
                outputs = []
                for connection in connections:
                    outputs.append(self._GetFromConnections(attrs, connection))
                return tuple(outputs)
        dictionary = {}
        for connection in ensure_list(connections):
            dictionary = {}
            if connection in self.projections_pre or connection in self.projections_post:
                if attrs is None:
                    attrs = np.union1d(self._default_connection_attrs, connection.attributes)
                else:
                    attrs = np.union1d(attrs, self._default_connection_attrs)
                for attribute in attrs:
                    self._set_attributes_of_connection_to_dict(dictionary, connection, attribute)
        return dictionary

    def info_neurons(self):
        neurons = self.neurons
        info = OrderedDict()
        if neurons.size:
            populations_inds = np.unique(neurons[:, 0])
            for pop_ind in populations_inds:
                info["nodes_gids_of_population_%d" % pop_ind] = np.array(neurons[neurons[:, 0] == pop_ind, 1])
        return info


class ANNarchyPopulation(_ANNarchyPopulation, SpikingPopulation):

    """ANNarchyPopulation class
       Wraps around an ANNarchy.core.Population.Population class and
       represents a population of neurons of the same neural model,
       residing at the same brain region.
    """

    def __init__(self, nodes=None, annarchy_network=None, **kwargs):
        _ANNarchyPopulation.__init__(self, nodes, annarchy_network, **kwargs)
        SpikingPopulation.__init__(self, nodes, **kwargs)

    def info(self, recursive=0):
        return SpikingPopulation.info(self, recursive=recursive)

    def info_details(self, recursive=0, **kwargs):
        return SpikingPopulation.info_details(self, recursive=recursive, **kwargs)
