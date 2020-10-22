# -*- coding: utf-8 -*-

from tvb_multiscale.core.spiking_models.population import SpikingPopulation

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list, extract_integer_intervals


class NESTPopulation(SpikingPopulation):

    """NESTPopulation class
       Wraps around a nest.NodeCollection and
       represents a population of neurons of the same neural model,
       residing at the same brain region.
    """

    nest_instance = None
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    def __init__(self, node_collection, label="", model="", nest_instance=None, **kwargs):
        self.nest_instance = nest_instance
        super(NESTPopulation, self).__init__(node_collection, label, model, **kwargs)

    @property
    def spiking_simulator_module(self):
        return self.nest_instance

    def _assert_nest(self):
        if self.nest_instance is None:
            raise ValueError("No NEST instance associated to this %s of model %s with label %s!" %
                             (self.__class__.__name__, self.model, self.label))

    @property
    def node_collection(self):
        return self._population

    @property
    def population(self):
        return self._population

    @property
    def neurons(self):
        return self._population.global_id

    def _assert_neurons(self, neurons=None):
        if neurons is None:
            neurons = self._population
        else:
            self._assert_nest()
            if not isinstance(neurons, self.nest_instance.NodeCollection):
                neurons = self.nest_instance.NodeCollection(neurons)
        return neurons

    def summarize_neurons_indices(self, print=False):
        """Method to summarize neurons' indices' intervals.
        Arguments:
         print: if True, a string is returned, Default = False
        Returns:
         a list of intervals' limits, or of single indices, or a string of the list if print = True"""
        return extract_integer_intervals(self.neurons, print=print)

    def _print_neurons(self):
        return "\n%d neurons: %s" % (self.number_of_neurons, self.summarize_neurons_indices(print=True))

    def _Set(self, values_dict, neurons=None):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
        """
        self._assert_neurons(neurons).set(values_dict)

    def _Get(self, attrs=None, neurons=None):
        """Method to get attributes of the SpikingPopulation's neurons.
           Arguments:
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all attributes
            neurons: instance of a NodeCollection class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
           Returns:
            Dictionary of tuples of neurons' attributes.
        """
        if attrs is None:
            return self._assert_neurons(neurons).get()
        else:
            return self._assert_neurons(neurons).get(ensure_list(attrs))

    def _GetConnections(self, neurons=None, source_or_target=None):
        """Method to get all the connections from/to a SpikingPopulation neuron.
        Arguments:
            neurons: nest.NodeCollection or sequence (tuple, list, array) of neurons
                     the connections of which should be included in the output.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            nest.SynapseCollection.
        """
        self._assert_nest()
        neurons = self._assert_neurons(neurons)
        if source_or_target not in ["source", "target"]:
            return self.nest_instance.GetConnections(source=neurons), \
                   self.nest_instance.GetConnections(target=neurons)
        else:
            kwargs = {source_or_target: neurons}
            return self.nest_instance.GetConnections(**kwargs)

    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
             values_dict: dictionary of attributes names' and values.
             connections: nest.SynapseCollection, or a tuple of outgoing and incoming nest.SynapseCollection instances
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
        connections.set(values_dict)

    def _GetFromConnections(self, attrs=None, connections=None):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
            Arguments:
             attrs: collection (list, tuple, array) of the attributes to be included in the output.
                    Default = None, corresponds to all attributes
             connections: nest.SynapseCollection, or a tuple of outgoing and incoming nest.SynapseCollection instances
                          Default = None, corresponding to all connections to/from the present population.
            Returns:
             Dictionary of tuples of connections' attributes.

        """
        if connections is None:
            connections = self._GetConnections()
        if isinstance(connections, tuple):
            if len(connections) == 1:
                connections = connections[0]
            else:
                # In case we deal with both source and target connections, treat them separately:
                outputs = []
                for connection in connections:
                    outputs.append(self._GetFromConnections(attrs, connection))
                return tuple(outputs)
        if attrs is None:
            return connections.get()
        else:
            return connections.get(ensure_list(attrs))
