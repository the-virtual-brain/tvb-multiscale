# -*- coding: utf-8 -*-

from tvb_multiscale.core.spiking_models.population import SpikingPopulation
from tvb_multiscale.tvb_nest.nest_models.node import NESTNodeCollection

from tvb.basic.neotraits.api import Attr


class NESTPopulation(NESTNodeCollection, SpikingPopulation):

    """NESTPopulation class
       Wraps around a nest.NodeCollection and
       represents a population of neurons of the same neural model,
       residing at the same brain region.
    """

    from nest import NodeCollection

    _nodes = Attr(field_type=NodeCollection, default=NodeCollection(), required=False,
                  label="Population", doc="""NEST population NodeCollection instance""")

    def __init__(self, nodes=None, nest_instance=None, **kwargs):
        self.nest_instance = nest_instance
        NESTNodeCollection.__init__(self, nodes, nest_instance, **kwargs)
        SpikingPopulation.__init__(self, nodes, **kwargs)

    def _assert_neurons(self, neurons=None):
        return self._assert_nodes(neurons)

    def _Set(self, values_dict, neurons=None):
        """Method to set attributes of the SpikingPopulation's neurons.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            neurons: instance of a population class,
                     or sequence (list, tuple, array) of neurons the attributes of which should be set.
                     Default = None, corresponds to all neurons of the population.
        """
        NESTNodeCollection._Set(self, values_dict, neurons)

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
        return NESTNodeCollection._Get(self, attrs, neurons)

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
        return NESTNodeCollection._GetConnections(self, neurons, source_or_target)


class NESTParrotPopulation(NESTPopulation):

    """NESTParrotPopulation class to wrap around a NEST parrot_neuron population"""

    def __init__(self, nodes=None, nest_instance=None, **kwargs):
        model = kwargs.get("model", "parrot_neuron")
        if len(model) == 0:
            model = "parrot_neuron"
        kwargs["model"] = model
        NESTPopulation.__init__(self, nodes, nest_instance, **kwargs)
