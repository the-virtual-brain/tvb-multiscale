# -*- coding: utf-8 -*-

from tvb_multiscale.core.spiking_models.population import SpikingPopulation
from tvb_multiscale.tvb_nest.nest_models.node import _NESTNodeCollection

from tvb.basic.neotraits.api import Attr


class NESTPopulation(_NESTNodeCollection, SpikingPopulation):

    """NESTPopulation class
       Wraps around a nest.NodeCollection and
       represents a population of neurons of the same neural model,
       residing at the same brain region.
    """

    from nest import NodeCollection

    # _nodes = Attr(field_type=NodeCollection, default=NodeCollection(), required=False,
    #               label="Population", doc="""NEST population NodeCollection instance""")

    def __init__(self, nodes=NodeCollection(), nest_instance=None, **kwargs):
        self.nest_instance = nest_instance
        _NESTNodeCollection.__init__(self, nodes, nest_instance, **kwargs)
        SpikingPopulation.__init__(self, nodes, **kwargs)

    def __getstate__(self):
        d = SpikingPopulation.__getstate__(self)
        d.update(_NESTNodeCollection.__getstate__(self))
        return d

    def __setstate__(self, d):
        SpikingPopulation.__setstate__(self, d)
        _NESTNodeCollection.__setstate__(self, d)

    def __str__(self):
        return SpikingPopulation.__str__(self)

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
        _NESTNodeCollection._Set(self, values_dict, neurons)

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
        return _NESTNodeCollection._Get(self, attrs, neurons)

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
        return _NESTNodeCollection._GetConnections(self, neurons, source_or_target)


class NESTParrotPopulation(NESTPopulation):

    """NESTParrotPopulation class to wrap around a NEST parrot_neuron population"""

    from nest import NodeCollection

    def __init__(self, nodes=NodeCollection(), nest_instance=None, **kwargs):
        model = kwargs.get("model", "parrot_neuron")
        if len(model) == 0:
            model = "parrot_neuron"
        kwargs["model"] = model
        NESTPopulation.__init__(self, nodes, nest_instance, **kwargs)
