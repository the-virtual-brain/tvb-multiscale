# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import uuid
from collections import OrderedDict

import numpy as np

from tvb.basic.neotraits.api import Attr, Int
from tvb.contrib.scripts.utils.data_structures_utils import list_of_dicts_to_dicts_of_ndarrays

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.utils.data_structures_utils import summarize, extract_integer_intervals, summary_info


LOG = initialize_logger(__name__)


class SpikingNodeCollection(HasTraits):
    __metaclass__ = ABCMeta

    """SpikingNodeCollection is a class that 
       represents a nodes collection of the spiking network of the same neural model, 
       residing at the same brain region.
       The abstract methods have to be implemented by 
       spiking simulator specific classes that will inherit this class.
    """

    _nodes = None  # Class instance of a sequence of nodes, that depends on its spiking simulator

    label = Attr(field_type=str, default="", required=True,
                 label="Node label", doc="""Label of SpikingNodeCollection""")

    model = Attr(field_type=str, default="", required=True, label="Node model",
                 doc="""Label of model of SpikingNodeCollection's nodes""")

    brain_region = Attr(field_type=str, default="", required=True, label="Brain region",
                        doc="""Label of the brain region the SpikingNodeCollection resides""")

    _size = Int(field_type=int, default=0, required=True, label="Size",
                doc="""The number of elements of SpikingNodeCollection """)

    _source_conns_attr = ""
    _target_conns_attr = ""
    _weight_attr = ""
    _delay_attr = ""
    _receptor_attr = ""

    def __init__(self, nodes=None, **kwargs):
        """Constructor of a population class.
           Arguments:
            nodes: Class instance of a sequence of spiking network elements, 
                  that depends on each spiking simulator. Default=None.
            **kwargs that may contain:
                label: a string with the label of the node
                model: a string with the name of the model of the node
                brain_region: a string with the name of the brain_region where the node resides
        """
        self._nodes = nodes
        self.label = str(kwargs.get("label", self.__class__.__name__))
        self.model = str(kwargs.get("model", self.__class__.__name__))
        self.brain_region = str(kwargs.get("brain_region", ""))
        self._size = self.get_size()
        HasTraits.__init__(self)
        self.configure()

    def __getstate__(self):
        return {"_nodes": self._nodes,
                "label": self.label,
                "model": self.model,
                "brain_region": self.brain_region,
                "_size": self._size,
                "_weight_attr": self._weight_attr,
                "_delay_attr": self._delay_attr,
                "_receptor_attr": self._receptor_attr,
                "gid": self.gid,
                "title": self.title,
                "tags": self.tags}

    def __setstate__(self, d):
        self._nodes = d.get("_nodes", None)
        self.label = d.get("label", "")
        self.model = d.get("model", "")
        self._size = d.get("_size", self.get_size())
        self.brain_region = d.get("brain_region", "")
        self._weight_attr = d.get("_weight_attr", "")
        self._delay_attr = d.get("_delay_attr", "")
        self._receptor_attr = d.get("_receptor_attr", "")
        self.gid = d.get("gid", uuid.uuid4())
        self.title = d.get("title",
                           '{} gid: {}'.format(self.__class__.__name__, self.gid))
        self.tags = d.get("tags", {})
        self.configure()

    def __getitem__(self, keys):
        """Slice specific nodes (keys) of this SpikingNodeCollection.
           Argument:
            keys: sequence of target populations' keys.
           Returns:
            Sub-collection of SpikingNodeCollection nodes.
        """
        return self._nodes[keys]

    @property
    def node_collection(self):
        return self._nodes

    @property
    @abstractmethod
    def spiking_simulator_module(self):
        pass

    @abstractmethod
    def _assert_spiking_simulator(self):
        """Method to assert that the node of the network is valid"""
        pass

    @abstractmethod
    def _assert_nodes(self, nodes=None):
        """Method to assert that the node of the network is valid"""
        pass

    @property
    @abstractmethod
    def gids(self):
        """Method to get a sequence (list, tuple, array) of the individual gids of nodes's elements"""
        pass

    @property
    def nodes(self):
        return self._nodes

    # Methods to get or set attributes for nodes and/or their connections:

    @abstractmethod
    def _Set(self, values_dict, nodes=None):
        """Method to set attributes of the SpikingNodeCollection's nodes.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            nodes: instance of a nodes class,
                   or sequence (list, tuple, array) of nodes the attributes of which should be set.
                   Default = None, corresponds to all nodes.
        """
        pass

    @abstractmethod
    def _Get(self, attr=None, nodes=None):
        """Method to get attributes of the SpikingNodeCollection's nodes.
           Arguments:
            attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                   Default = None, corresponding to all attributes
            nodes: instance of a nodes class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes.
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of nodes' attributes.
        """
        pass

    @abstractmethod
    def _GetConnections(self, nodes=None, source_or_target=None):
        """Method to get all the connections from/to a SpikingNodeCollection node.
           Arguments:
            nodes: instance of a nodes class,
                   or sequence (list, tuple, array) of nodes the attributes of which should be set.
                   Default = None, corresponds to all nodes.
            source_or_target: Direction of connections relative to nodes
                              "source", "target" or None (Default; corresponds to both source and target)
           Returns:
            connections' objects.
        """
        pass

    @abstractmethod
    def _SetToConnections(self, values_dict, connections=None):
        """Method to set attributes of the connections from/to the SpikingNodeCollection's nodes.
           Arguments:
             values_dict: dictionary of attributes names' and values.
             connections: connections' objects.
                          Default = None, corresponding to all connections to/from the present nodes.
        """
        pass

    @abstractmethod
    def _GetFromConnections(self, attrs=None, connections=None):
        """Method to get attributes of the connections from/to the SpikingNodeCollection's nodes.
            Arguments:
             attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                    Default = None, corresponding to all attributes
             connections: connections' objects.
                          Default = None, corresponding to all connections to/from the present nodes.
            Returns:
             Dictionary of sequences (lists, tuples, or arrays) of connections' attributes.

        """
        pass

    def get_size(self):
        """Method to compute the total number of SpikingNodeCollection's nodes.
            Returns:
                int: number of nodes.
        """
        if self._nodes:
            return len(self._nodes)
        else:
            return 0

    def Set(self, values_dict, nodes=None):
        """Method to set attributes of the SpikingNodeCollection's nodes.
        Arguments:
            values_dict: dictionary of attributes names' and values.
            nodes: instance of a nodes class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes.
        """
        self._Set(values_dict, nodes)

    def Get(self, attrs=None, nodes=None, summary=None):
        """Method to get attributes of the SpikingNodeCollection's nodes.
           Arguments:
            attrs: names of attributes to be returned. Default = None, corresponds to all nodes' attributes.
            nodes: instance of a nodes class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of nodes' attributes.
        """
        attributes = self._Get(attrs, nodes)
        if isinstance(attributes, (tuple, list)):
            attributes = list_of_dicts_to_dicts_of_ndarrays(attributes)
        if summary:
            return summarize(attributes, summary)
        else:
            return attributes

    def get_attributes(self, nodes=None, summary=False):
        """Method to get all attributes of the SpikingNodeCollection's nodes.
           Arguments:
            nodes: instance of a nodes class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of nodes' attributes.
        """
        return self.Get(nodes=nodes, summary=summary)

    def GetConnections(self, nodes=None,  source_or_target=None):
        """Method to get all connections of the device to/from nodes.
           Arguments:
            nodes: instance of a nodes class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes.
            Returns:
                connections' objects.
        """
        return self._GetConnections(nodes, source_or_target)

    def SetToConnections(self, values_dict, nodes=None, source_or_target=None, connections=None):
        """Method to set attributes of the connections from/to the SpikingNodeCollection's nodes.
           Arguments:
            values_dict: dictionary of attributes names' and values.
            nodes: instance of a nodes class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes.
            source_or_target: Direction of connections relative to the populations' nodes
                              "source", "target" or None (Default; corresponds to both source and target)
            connections: connections' objects, identical to the output of the GetConnections() method
                         Default = None, in which the arguments above are taken into consideration.
        """
        if connections is None:
            if source_or_target is None:
                # In case we deal with both source and target connections, treat them separately:
                for source_or_target in ["source", "target"]:
                    self.SetToConnections(values_dict, nodes, source_or_target)
            else:
                connections = self.GetConnections(nodes, source_or_target)
                self._SetToConnections(values_dict, connections)
        self._SetToConnections(values_dict, connections)

    def GetFromConnections(self, attrs=None, nodes=None, source_or_target=None, connections=None, summary=None):
        """Method to get attributes of the connections from/to the SpikingNodeCollection's nodes.
           Arguments:
            attrs: sequence (list, tuple, array) of the attributes to be included in the output.
                   Default = None, correspondingn to all attributes
            nodes: instance of a nodes class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes.
            source_or_target: Direction of connections relative to the populations' nodes
                              "source", "target" or None (Default; corresponds to both source and target)
            connections: connections' objects, identical to the output of the GetConnections() method
                         Default = None, in which the arguments above are taken into consideration.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Dictionary of lists of connections' attributes,
            or tuple of two such dictionaries for source and target connections
        """
        if connections is None:
            if source_or_target is None:
                # In case we deal with both source and target connections, treat them separately:
                output = []
                for source_or_target in ["source", "target"]:
                    output.append(self.GetFromConnections(attrs=attrs, nodes=nodes,
                                                          source_or_target=source_or_target, summary=summary))
                if len(output) == 0:
                    return {}
                if len(output) == 1:
                    return output[0]
                return tuple(output)
            else:
                outputs = self._GetFromConnections(attrs,
                                                   self.GetConnections(nodes=nodes, source_or_target=source_or_target))
        else:
            outputs = self._GetFromConnections(attrs, connections)
        if summary is not None:
            outputs = summarize(outputs, summary)
        return outputs

    def _get_connection_attribute(self, attr, nodes=None, source_or_target=None, connections=None, summary=None):
        """Method to get a single connections' attribute of the SpikingNodeCollections's nodes.
                   Arguments:
                    attr: the attribute to be returned
                    nodes: instance of a nodes class,
                             or sequence (list, tuple, array) of nodes the attributes of which should be returned.
                             Default = None, corresponds to all nodes.
                    source_or_target: Direction of connections relative to the populations' nodes
                                      "source", "target" or None (Default; corresponds to both source and target)
                    connections: connections' objects, identical to the output of the GetConnections() method
                                 Default = None, in which the arguments above are taken into consideration.
                    summary: if integer, return a summary of unique output values
                                         within accuracy of the specified number of decimal digits
                             otherwise, if it is not None or False return
                             either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                             or a list of unique string entries for all other attributes,
                             Default = None, corresponds to returning all values
                   Returns:
                    Sequence (list, tuple, or array) of nodes's connections' attribute.
                """
        if connections is None:
            if source_or_target is None:
                # In case we deal with both source and target connections, treat them separately:
                outputs = []
                for source_or_target in ["source", "target"]:
                    outputs.append(self._get_connection_attribute(attr, nodes=nodes, source_or_target=source_or_target,
                                                                  summary=summary))
                return tuple(outputs)
            return self.GetFromConnections(attr, nodes=nodes, source_or_target=source_or_target,
                                           summary=summary).get(attr, [])
        else:
            return self.GetFromConnections(attr, connections=connections, summary=summary).get(attr, [])

    def get_connected_nodes(self, nodes=None, source_or_target=None, connections=None, summary=None):
        """Method to get the connected nodes of the SpikingNodeCollections's nodes.
            Arguments:
                nodes: instance of a nodes class,
                       or sequence (list, tuple, array) of nodes the attributes of which should be set.
                       Default = None, corresponds to all nodes.
                source_or_target: Direction of connections relative to the populations' nodes
                                  "source", "target" or None (Default; corresponds to both source and target)
                connections: connections' objects, identical to the output of the GetConnections() method
                             Default = None, in which the arguments above are taken into consideration.
                summary: if integer, return a summary of unique output values
                         within accuracy of the specified number of decimal digits
                         otherwise, if it is not None or False return
                         either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                         or a list of unique string entries for all other attributes,
                        Default = None, corresponds to returning all values
                Returns:
                    Sequence (list, tuple, or array) of connected nodes' gids.
                """
        if connections is None:
            if source_or_target is None:
                # In case we deal with both source and target connections, treat them separately:
                outputs = []
                for source_or_target in ["source", "target"]:
                    outputs.append(self.get_connected_nodes(nodes=nodes, source_or_target=source_or_target,
                                                            ummary=summary))
                return tuple(outputs)
            # In this case the connections are found based on source_or_target,
            # and we need to reverse source_or_target to determines the nodes to return:
            if source_or_target == "target":
                attr = self._source_conns_attr
            else:
                attr = self._target_conns_attr
            return self.GetFromConnections(attr, nodes=nodes, source_or_target=source_or_target,
                                           summary=summary).get(attr, [])
        else:
            # In this case the connections have already been found,
            # and the source_or_target determines if we want the sources or targets of those connections.
            attr = getattr(self, "_%s_conns_attr" % source_or_target)
            return self.GetFromConnections(attr, connections=connections, summary=summary).get(attr, [])

    def get_weights(self, nodes=None, source_or_target=None, connections=None, summary=None):
        """Method to get the connections' weights of the SpikingNodeCollections's nodes.
           Arguments:
            nodes: instance of a nodes class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes.
            source_or_target: Direction of connections relative to the populations' nodes
                              "source", "target" or None (Default; corresponds to both source and target)
            connections: connections' objects, identical to the output of the GetConnections() method
                         Default = None, in which the arguments above are taken into consideration.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Sequence (list, tuple, or array) of nodes's connections' weights.
        """
        return self._get_connection_attribute(self._weight_attr, nodes, source_or_target, connections, summary)

    def get_delays(self, nodes=None, source_or_target=None, connections=None, summary=None):
        """Method to get the connections' delays of the SpikingNodeCollections's nodes.
           Arguments:
            nodes: instance of a nodes class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes.
            source_or_target: Direction of connections relative to the populations' nodes
                              "source", "target" or None (Default; corresponds to both source and target)
            connections: connections' objects, identical to the output of the GetConnections() method
                         Default = None, in which the arguments above are taken into consideration.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Sequence (list, tuple, or array) of nodes's connections' delays.
        """
        return self._get_connection_attribute(self._delay_attr, nodes, source_or_target, connections, summary)

    def get_receptors(self, nodes=None, source_or_target=None, connections=None, summary=None):
        """Method to get the connections' receptors of the SpikingNodeCollections's nodes.
            nodes: instance of a nodes class,
                     or sequence (list, tuple, array) of nodes the attributes of which should be set.
                     Default = None, corresponds to all nodes.
            source_or_target: Direction of connections relative to the populations' nodes
                              "source", "target" or None (Default; corresponds to both source and target)
            connections: connections' objects, identical to the output of the GetConnections() method
                         Default = None, in which the arguments above are taken into consideration.
            summary: if integer, return a summary of unique output values
                                 within accuracy of the specified number of decimal digits
                     otherwise, if it is not None or False return
                     either a dictionary of a statistical summary of mean, minmax, and variance for numerical attributes,
                     or a list of unique string entries for all other attributes,
                     Default = None, corresponds to returning all values
           Returns:
            Sequence (list, tuple, or array) of nodes's connections' receptors.
        """
        return self._get_connection_attribute(self._receptor_attr, nodes, source_or_target, connections, summary)

    @property
    def number_of_nodes(self):
        """Method to get the total number of SpikingNodeCollection's nodes and set the respective protected property.
            Returns:
             int: number of nodes.
        """
        if self._size == 0 or self._size is None:
            self._size = self.get_size()
        return self._size

    @property
    def attributes(self):
        """Method to get the attributes of the SpikingNodeCollection's nodes.
           Returns:
            Dictionary of sequences (lists, tuples, or arrays)  of nodes's nodes' attributes.
        """
        return self.get_attributes()

    @property
    def connections(self):
        """Method to get the connections of the SpikingNodeCollection's nodes.
           Returns:
            connections' objects.
        """
        return self.GetConnections()

    @property
    def weights(self):
        """Method to get the connections' weights' statistical summary of the SpikingNodeCollections's nodes.
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of nodes's connections' weights.
        """
        return self.get_weights()

    @property
    def delays(self):
        """Method to get the connections' delays of the SpikingNodeCollections's nodes.
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of nodes's connections' delays.
        """
        return self.get_delays()

    @property
    def receptors(self):
        """Method to get the connections' receptors of the SpikingNodeCollections's nodes.
           Returns:
            Dictionary of sequences (lists, tuples, or arrays) of nodes's connections' receptors.
        """
        return self.get_receptors()

    def info_nodes(self):
        info = OrderedDict()
        info['number_of_nodes'] = self.number_of_nodes
        info["nodes"] = self.nodes
        return info

    def info_neurons(self):
        return {"gids": np.array(self.gids)}

    def _info_connectivity(self, source_or_target, attributes=True):
        source_or_target = source_or_target.lower()
        info = OrderedDict()
        conns = self.GetConnections(source_or_target=source_or_target)
        if source_or_target == "target":
            source_or_target = "in"
            source_or_target_reverse = "source"
        else:
            source_or_target = "out"
            source_or_target_reverse = "target"
        if attributes:
            if attributes == True:
                attributes = [getattr(self, "_%s_conns_attr" % source_or_target_reverse),
                              self._weight_attr, self._delay_attr, self._receptor_attr]
            conn_attrs = self.GetFromConnections(attrs=attributes, connections=conns)
            for attr in attributes:
                info["%s_%s" % (attr, source_or_target)] = conn_attrs.get(attr, np.array([]))
        return info

    def _info_connections(self, source_or_target):
        return self._info_connectivity(source_or_target, False)

    def info_connectivity(self, source_or_target=None):
        info = OrderedDict()
        if source_or_target is None or source_or_target.lower() == "source":
            info.update(self._info_connectivity("source"))
        if source_or_target is None or source_or_target.lower() == "target":
            info.update(self._info_connectivity("target"))
        return info

    def info_connections(self, source_or_target=None):
        info = OrderedDict()
        if source_or_target is None or source_or_target.lower() == "source":
            info.update(self._info_connections("source"))
        if source_or_target is None or source_or_target.lower() == "target":
            info.update(self._info_connections("target"))
        return info

    def info(self, recursive=0):
        info = super(SpikingNodeCollection, self).info(recursive=recursive)
        info.update(self.info_nodes())
        return info

    def info_details(self, recursive=0, connectivity=False, source_or_target=None):
        info = super(SpikingNodeCollection, self).info_details(recursive=recursive)
        info.update(self.info_nodes())
        info.update(self.info_neurons())
        if self._nodes is not None:
            info["parameters"] = self.get_attributes(summary=False)
            if connectivity:
                info["connectivity"] = self.info_connectivity(source_or_target)
        return info
