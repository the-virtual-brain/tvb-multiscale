# -*- coding: utf-8 -*-

import numpy
import ray

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.tvb_nest.config import CONFIGURED


class RayNESTServer(object):

    def __init__(self, config=CONFIGURED):
        from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import load_nest, configure_nest_kernel
        self.nest_instance = configure_nest_kernel(load_nest(config=config), config=config)

    def _node_collection_to_gids(self, node_collection):
        return node_collection.global_id

    def NodeCollection(self, node_gids, *args, **kwargs):
        output = self.nest_instance.NodeCollection(tuple(ensure_list(node_gids)), *args, **kwargs)
        return output

    def _synapse_collection_to_dict(self, synapse_collection):
        return synapse_collection.get()

    def SynapseCollection(self, elements):
        synapse_model = numpy.unique(elements["synapse_model"])
        if numpy.all(synapse_model == synapse_model[0]):
            synapse_model = synapse_model[0]
        else:
            synapse_model = None
        return self.nest_instance.GetConnections(source=self.NodeCollection(numpy.unique(elements["source"])),
                                                 target=self.NodeCollection(numpy.unique(elements["target"])),
                                                 synapse_model=synapse_model)

    @ray.method(num_returns=1)
    def nest(self, attr, *args, **kwargs):
        # For all calls that do not have NodeCollection or SynapseCollection as inputs or outputs
        return getattr(self.nest_instance, attr)(*args, **kwargs)

    def _nodes_or_synapses(self, elements):
        if isinstance(elements, dict):
            # assuming these are connections:
            return self.SynapseCollection(elements)
        elif isinstance(elements, (tuple, list)):
            # assuming these are nodes:
            return self.NodeCollection(elements)
        else:
            return elements

    @ray.method(num_returns=1)
    def get(self, nodes, *params, **kwargs):
        if len(nodes):
            return self._nodes_or_synapses(nodes).get(*params, **kwargs)
        else:
            return ()

    @ray.method(num_returns=1)
    def set(self, nodes, params=None, **kwargs):
        if len(nodes):
            return self._nodes_or_synapses(nodes).set(params=params, **kwargs)
        else:
            return None

    @ray.method(num_returns=1)
    def GetStatus(self, nodes, keys=None, output=''):
        if len(nodes):
            return self.nest_instance.GetStatus(self._nodes_or_synapses(nodes), keys=keys, output=output)
        else:
            return ()

    @ray.method(num_returns=1)
    def SetStatus(self, nodes, params, val=None):
        if len(nodes):
            return self.nest_instance.SetStatus(self._nodes_or_synapses(nodes), params, val=val)
        else:
            return None

    @ray.method(num_returns=1)
    def install_nest(self, *args, **kwargs):
        from pynestml.frontend.pynestml_frontend import install_nest
        return install_nest(*args, **kwargs)

    @ray.method(num_returns=1)
    def help(self, obj=None, return_text=True):
        # TODO: a warning for when return_text = False
        return self.nest_instance.help(obj=self._nodes_or_synapses(obj), return_text=True)

    @ray.method(num_returns=1)
    def Create(self, model, n=1, params=None, positions=None):
        return self._node_collection_to_gids(
            self.nest_instance.Create(model, n=n, params=params, positions=positions))

    @ray.method(num_returns=1)
    def GetNodes(self, properties={}, local_only=False):
        return self._node_collection_to_gids(
            self.nest_instance.GetNodes(properties=properties, local_only=local_only))

    @ray.method(num_returns=1)
    def GetLocalNodeCollection(self, node_inds):
        if len(node_inds):
            return self._node_collection_to_gids(
                self.nest_instance.GetLocalNodeCollection(self.NodeCollection(node_inds)))
        else:
            return ()

    @ray.method(num_returns=1)
    def Connect(self, pre, post, conn_spec=None, syn_spec=None):
        # return self._synapse_collection_to_dict(
        return self.nest_instance.Connect(self.NodeCollection(pre),
                                           self.NodeCollection(post),
                                           conn_spec=conn_spec, syn_spec=syn_spec)  # )

    @ray.method(num_returns=1)
    def Disconnect(self, pre, post, conn_spec='one_to_one', syn_spec='static_synapse'):
        return self.nest_instance.Disconnect(self.NodeCollection(pre),
                                             self.NodeCollection(post),
                                             conn_spec=conn_spec, syn_spec=syn_spec)

    @ray.method(num_returns=1)
    def GetConnections(self, source=None, target=None, synapse_model=None, synapse_label=None):
        if source is not None:
            source = self.NodeCollection(source)
        if target is not None:
            target = self.NodeCollection(target)
        return self._synapse_collection_to_dict(
            self.nest_instance.GetConnections(source=source, target=target,
                                              synapse_model=synapse_model, synapse_label=synapse_label))

    @ray.method(num_returns=1)
    def CopyModel(self, existing, new, params=None):
        self.nest_instance.CopyModel(existing, new, params)
        return new
