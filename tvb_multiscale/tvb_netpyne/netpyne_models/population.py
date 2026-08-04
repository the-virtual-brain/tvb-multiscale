# -*- coding: utf-8 -*-

from tvb_multiscale.core.spiking_models.population import SpikingPopulation


class NetpynePopulation(SpikingPopulation):

    netpyne_instance = None

    def __init__(self, nodes, netpyne_instance, label, global_label, brain_region, **kwargs):

        self.netpyne_instance = netpyne_instance

        if global_label is None:
            self.global_label = brain_region + '.' + label
        else:
            self.global_label = global_label

        # to be used below in super.init
        kwargs["label"] = label
        kwargs["brain_region"] = brain_region

        super(NetpynePopulation, self).__init__(nodes, **kwargs)

    def population(self):
        """Overrides correpondent method of SpikingPopulation"""
        return self.gids

    def get_size(self):
        """Overrides correpondent method of SpikingNodeCollection"""
        return len(self.gids)

    @property
    def gids(self):
        """Method to get a sequence (list, tuple, array) of the individual gids of populations' neurons"""
        gids = self.netpyne_instance.cellGidsForPop(self.global_label)
        return gids

    def _Set(self, values_dict, neurons=None):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
        """
        pass

    def _Get(self, attr=None, neurons=None):
        """Method to get attributes of the SpikingPopulation's neurons.
           Arguments:
            attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all attributes
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of neurons' attributes.
        """
        pass

    def _GetConnections(self, neurons=None, source_or_target=None):
        """Method to get all the connections from/to a SpikingPopulation neuron.
           Arguments:
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
            source_or_target: Direction of connections relative to the populations' neurons
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            connections' objects.
        """
        pass

    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from/to the SpikingPopulation's neurons.
           Arguments:
             values_dict: dictionary of attributes names' and values.
             connections: connections' objects.
                          Default = None, corresponding to all connections to/from the present population.
        """
        pass

    def _GetFromConnections(self, attr=None, connections=None):
        """Method to get attributes of the connections from/to the SpikingPopulation's neurons.
            Arguments:
             attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                    Default = None, corresponding to all attributes
             connections: connections' objects.
                          Default = None, corresponding to all connections to/from the present population.
            Returns:
             Dictionary of sequences (lists, tuples, or arrays) of connections' attributes.

        """
        pass

    @property
    def spiking_simulator_module(self):
        return self.netpyne_instance

    @property
    def _assert_spiking_simulator(self):
        if self.netpyne_instance is None:
            raise ValueError("No NetPyNE instance associated to this %s of model %s with label %s!" %
                             (self.__class__.__name__, self.model, self.label))

    @property
    def _assert_nodes(self, nodes=None):
        pass