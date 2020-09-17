# -*- coding: utf-8 -*-

from tvb_multiscale.core.spiking_models.population import SpikingPopulation

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list, extract_integer_intervals


class NESTPopulation(SpikingPopulation):

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
    def neurons(self):
        return self._population.global_id

    def summarize_neurons_indices(self, print=False):
        """Method to summarize neurons' indices' intervals.
        Arguments:
         print: if True, a string is returned, Default = False
        Returns:
         a list of intervals' limits, or of single indices, or a string of the list if print = True"""
        return extract_integer_intervals(self.neurons, print=print)

    def _print_neurons(self):
        return "\n%d neurons: %s" % (self.number_of_neurons, self.summarize_neurons_indices(print=True))

    def _assert_neurons(self, neurons=None):
        if neurons is None:
            neurons = self._population
        elif not isinstance(neurons, self.nest_instance.NodeCollection):
            neurons = self.nest_instance.NodeCollection(neurons)
        return neurons

    def _Set(self, neurons, values_dict):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            neurons: tuple of neurons the attributes of which should be set.
            values_dict: dictionary of attributes names' and values.
        """
        self._assert_nest()
        self._assert_neurons(neurons).set(values_dict)

    def _Get(self, neurons, attrs=None):
        """Method to get attributes of the SpikingPopulation's neurons.
           Arguments:
            neurons: tuple of neurons which should be included in the output.
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
           Returns:
            Dictionary of tuples of neurons' attributes.
        """
        self._assert_nest()
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

    def _SetToConnections(self, connections, values_dict):
        """Method to set attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
             connections: nest.SynapseCollection.
             values_dict: dictionary of attributes names' and values.
        """
        self._assert_nest()
        connections.set(values_dict)

    def _GetFromConnections(self, connections, attrs=None):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
            Arguments:
             connections: nest.SynapseCollection.
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
            Returns:
             Dictionary of tuples of connections' attributes.

        """
        self._assert_nest()
        if attrs is None:
            return connections.get()
        else:
            return connections.get(attrs)
