# -*- coding: utf-8 -*-

import uuid

from nest import NodeCollection

from tvb.basic.neotraits.api import Int

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.neotraits import HasTraits, Attr
from tvb_multiscale.core.spiking_models.node import SpikingNodeCollection

from tvb_multiscale.tvb_nest.nest_models.server_client.node_collection import NodeCollection as RemoteNodeCollection


LOG = initialize_logger(__name__)


class _NESTNodeCollection(HasTraits):

    """NESTNodeCollection is a class that
       represents a nodes collection of the spiking network of the same neural model,
       residing at the same brain region.
    """

    _nodes = Attr(field_type=(NodeCollection, RemoteNodeCollection), default=NodeCollection(), required=False,
                  label="NEST NodeCollection ", doc="""NESTNodeCollection instance""")

    label = Attr(field_type=str, default="", required=True,
                 label="Node label", doc="""Label of NESTNodeCollection""")

    model = Attr(field_type=str, default="", required=True, label="Node model",
                 doc="""Label of model of NESTNodeCollection's nodes""")

    brain_region = Attr(field_type=str, default="", required=True, label="Brain region",
                        doc="""Label of the brain region the NESTNodeCollection resides""")


    nest_instance = None
    _source_conns_attr = "source"
    _target_conns_attr = "target"
    _weight_attr = "weight"
    _delay_attr = "delay"
    _receptor_attr = "receptor"

    def __init__(self, nodes=NodeCollection(), nest_instance=None, **kwargs):
        """Constructor of a collection class.
           Arguments:
            nodes: Class instance of a sequence of spiking network elements,
                  that depends on each spiking simulator. Default=None.
            nest_instance: pyNEST module
            **kwargs that may contain:
                label: a string with the label of the node
                model: a string with the name of the model of the node
                brain_region: a string with the name of the brain_region where the node resides
        """
        self._nodes = nodes
        self.nest_instance = nest_instance
        self.label = str(kwargs.get("label", self.__class__.__name__))
        self.model = str(kwargs.get("model", self.__class__.__name__))
        self.brain_region = str(kwargs.get("brain_region", ""))
        HasTraits.__init__(self)
        self.configure()

    @property
    def spiking_simulator_module(self):
        return self.nest_instance

    def __getstate__(self):
        return {"nest_instance": self.nest_instance,
                "_nodes": self._nodes,
                "label": self.label,
                "model": self.model,
                "brain_region": self.brain_region,
                "_weight_attr": self._weight_attr,
                "_delay_attr": self._delay_attr,
                "_receptor_attr": self._receptor_attr,
                "gid": self.gid,
                "title": self.title,
                "tags": self.tags}

    def __setstate__(self, d):
        self.nest_instance = d.get("nest_instance", None)
        self._nodes = d.get("_nodes", None)
        self.label = d.get("label", "")
        self.model = d.get("model", "")
        self.brain_region = d.get("brain_region", "")
        self._weight_attr = d.get("_weight_attr", "")
        self._delay_attr = d.get("_delay_attr", "")
        self._receptor_attr = d.get("_receptor_attr", "")
        self.gid = d.get("gid", uuid.uuid4())
        self.title = d.get("title",
                           '{} gid: {}'.format(self.__class__.__name__, self.gid))
        self.tags = d.get("tags", {})

    def _assert_spiking_simulator(self):
        if self.nest_instance is None:
            raise ValueError("No NEST instance associated to this %s of model %s with label %s!" %
                             (self.__class__.__name__, self.model, self.label))

    def _assert_nest(self):
        return self._assert_spiking_simulator()

    def _assert_nodes(self, nodes=None):
        if nodes is None:
            return self._nodes
        if not isinstance(nodes, (NodeCollection, RemoteNodeCollection)):
            if self._nodes:
                try:
                    return self._nodes[nodes]
                except:
                    pass
            return self.nest_instance.NodeCollection(ensure_list(nodes))
        return nodes

    @property
    def gids(self):
        """Method to get a sequence (list, tuple, array) of the individual gids of nodes's elements"""
        if self._nodes:
            return tuple(ensure_list(self._nodes.global_id))
        else:
            return ()

    @property
    def global_id(self):
        return self.gids

    @property
    def nest_model(self):
        if self._nodes:
            return str(self._nodes.get("model"))
        else:
            return ""

    def _Set(self, values_dict, nodes=None):
        """Method to set attributes of the Spikingcollection's nodes.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            nodes: instance of a collection class,
                   or sequence (list, tuple, array) of nodes the attributes of which should be set.
                   Default = None, corresponds to all nodes of the collection.
        """
        self._assert_nodes(nodes).set(values_dict)

    def _Get(self, attrs=None, nodes=None):
        """Method to get attributes of the Spikingcollection's nodes.
           Arguments:
            attrs: collection (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all attributes
            nodes: instance of a NodeCollection class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes of the collection.
           Returns:
            Dictionary of tuples of nodes' attributes.
        """
        if attrs is None:
            return self._assert_nodes(nodes).get()
        else:
            return self._assert_nodes(nodes).get(ensure_list(attrs))

    def _GetConnections(self, nodes=None, source_or_target=None):
        """Method to get all the connections from/to a NESTNodeCollection neuron.
        Arguments:
            nodes: nest.NodeCollection or sequence (tuple, list, array) of nodes
                     the connections of which should be included in the output.
            source_or_target: Direction of connections relative to the collection's nodes
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            nest.SynapseCollection.
        """
        self._assert_spiking_simulator()
        nodes = self._assert_nodes(nodes)
        if source_or_target not in ["source", "target"]:
            return self.nest_instance.GetConnections(source=nodes), \
                   self.nest_instance.GetConnections(target=nodes)
        else:
            kwargs = {source_or_target: nodes}
            return self.nest_instance.GetConnections(**kwargs)

    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from/to the NESTNodeCollection's nodes.
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
        """Method to get attributes of the connections from/to the NESTNodeCollection's nodes.
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
                    outputs.append(self._GetFromConnections(attrs=attrs, connections=connection))
                return tuple(outputs)
        if attrs is None:
            return connections.get()
        else:
            return connections.get(ensure_list(attrs))


class NESTNodeCollection(_NESTNodeCollection, SpikingNodeCollection):

    """NESTNodeCollection is a class that 
       represents a nodes collection of the spiking network of the same neural model, 
       residing at the same brain region.
    """

    def info(self, recursive=0):
        return SpikingNodeCollection.info(self, recursive=recursive)

    def info_details(self, recursive=0, **kwargs):
        return SpikingNodeCollection.info_details(self, recursive=recursive, **kwargs)
