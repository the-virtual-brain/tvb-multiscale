# -*- coding: utf-8 -*-

from six import string_types
import json
import ray

import numpy
import pandas

from tvb.contrib.scripts.utils.data_structures_utils import \
    ensure_list, dicts_of_lists_to_lists_of_dicts, list_of_dicts_to_dict_of_tuples

from tvb_multiscale.core.utils.data_structures_utils import is_iterable
from tvb_multiscale.core.spiking_models.ray import RaySpikingNetwork

from tvb_multiscale.tvb_nest.config import CONFIGURED
from tvb_multiscale.tvb_nest.nest_models.ray.node_collection import RayNodeCollection
from tvb_multiscale.tvb_nest.nest_models.ray.synapse_collection import RaySynapseCollection


def serializable(data):
    """Make data serializable for JSON.
       Modified from pynest utils.

    Parameters
    ----------
    data : any

    Returns
    -------
    data_serialized : str, int, float, list, dict
        Data can be encoded to JSON
    """

    if isinstance(data, (numpy.ndarray, RayNodeCollection)):
        return data.tolist()
    if isinstance(data, RaySynapseCollection):
        # Get full information from SynapseCollection
        return serializable(data.todict())
    if isinstance(data, (list, tuple)):
        return [serializable(d) for d in data]
    if isinstance(data, dict):
        return dict([(key, serializable(value)) for key, value in data.items()])
    return data


def to_json(data, **kwargs):
    """Serialize data to JSON.
       Modified from pynest utils.

    Parameters
    ----------
    data : any
    kwargs : keyword argument pairs
        Named arguments of parameters for `json.dumps` function.

    Returns
    -------
    data_json : str
        JSON format of the data
    """

    data_serialized = serializable(data)
    data_json = json.dumps(data_serialized, **kwargs)
    return data_json


class RayNESTClient(object):

    run_task_ref_obj = None

    def __init__(self, nest_server):
        self.nest_server = nest_server

    def __getstate__(self):
        return {"nest_server": self.nest_server, "_run_ref": self.run_task_ref_obj}

    def __setstate__(self, d):
        self.nest_server = d.get("nest_server", None)
        self.run_task_ref_obj = d.get("_run_ref", None)

    def _gids(self, gids):
        return tuple(ensure_list(gids))

    def _node_collection_to_gids(self, node_collection):
        return node_collection.gids

    def NodeCollection(self, gids):
        return RayNodeCollection(self, self._gids(gids))

    def SynapseCollection(self, conns_dict):
        return RaySynapseCollection(self, conns_dict)

    def _synapse_collection_to_source_target_dict(self, synapse_collection):
        return {"source": synapse_collection.source, "target": synapse_collection.target,
                "synapse_model": synapse_collection.synapse_model}

    def get(self, nodes,  *params, **kwargs):
        if isinstance(nodes, RayNodeCollection):
            return ray.get(
                self.nest_server.get.remote(
                    self._node_collection_to_gids(nodes),  *params, **kwargs))
        else:
            return ray.get(
                self.nest_server.get.remote(
                    self._synapse_collection_to_source_target_dict(nodes),  *params, **kwargs))

    def set(self, nodes, params=None, **kwargs):
        if isinstance(nodes, RayNodeCollection):
            return ray.get(
                self.nest_server.set.remote(
                    self._node_collection_to_gids(nodes), params, **kwargs))
        else:
            return ray.get(
                self.nest_server.set.remote(
                    self._synapse_collection_to_source_target_dict(nodes), params, **kwargs))

    def GetStatus(self, nodes, attrs=None, output=None):
        if isinstance(nodes, RayNodeCollection):
            return ray.get(
                self.nest_server.GetStatus.remote(
                    self._node_collection_to_gids(nodes), attrs, output))
        else:
            return ray.get(
                self.nest_server.GetStatus.remote(
                    self._synapse_collection_to_source_target_dict(nodes), attrs, output))

    def SetStatus(self, nodes, params, val=None):
        if isinstance(nodes, RayNodeCollection):
            return ray.get(
                self.nest_server.SetStatus.remote(
                    self._node_collection_to_gids(nodes), params, val))
        else:
            return ray.get(
                self.nest_server.SetStatus.remote(
                    self._synapse_collection_to_source_target_dict(nodes), params, val))

    def Create(self, model, n=1, params=None, positions=None):
        return self.NodeCollection(
            ray.get(self.nest_server.Create.remote(model, n=n, params=params, positions=positions)))

    def Connect(self, pre, post, conn_spec=None, syn_spec=None, return_synapsecollection=False):
        if return_synapsecollection:
            return self.SynapseCollection(ray.get(
                self.nest_server.Connect.remote(self._node_collection_to_gids(pre),
                                                self._node_collection_to_gids(post),
                                                conn_spec=conn_spec, syn_spec=syn_spec,
                                                return_synapsecollection=True)))
        else:
            return ray.get(
                self.nest_server.Connect.remote(self._node_collection_to_gids(pre),
                                                self._node_collection_to_gids(post),
                                                conn_spec=conn_spec, syn_spec=syn_spec,
                                                return_synapsecollection=False))

    def Disconnect(self, pre, post, conn_spec='one_to_one', syn_spec='static_synapse'):
        return ray.get(
            self.nest_server.Disconnect.remote(self._node_collection_to_gids(pre),
                                               self._node_collection_to_gids(post),
                                               conn_spec=conn_spec, syn_spec=syn_spec))

    def GetLocalNodeCollection(self, node_collection):
        if len(node_collection):
            return self.NodeCollection(
                self.nest_server.GetLocalNodeCollection.remote(
                    self._node_collection_to_gids(node_collection)))
        else:
            return self.NodeCollection(())

    def GetConnections(self, source=None, target=None, synapse_model=None, synapse_label=None):
        if source is not None:
            source = self._node_collection_to_gids(source)
        if target is not None:
            target = self._node_collection_to_gids(target)
        return self.SynapseCollection(
            ray.get(self.nest_server.GetConnections.remote(source=source, target=target,
                                                           synapse_model=synapse_model,
                                                           synapse_label=synapse_label)))

    def GetNodes(self, properties={}, local_only=False):
        return self.NodeCollection(
            self.nest_server.GetNodes.remote(properties=properties, local_only=local_only))

    def Models(self):
        return ray.get(self.nest_server.nest.remote("Models"))

    def GetDefaults(self, model, keys=None, output=''):
        return ray.get(self.nest_server.nest.remote("GetDefaults", model, keys=keys, output=output))

    def SetDefaults(self, model, params, val=None):
        return ray.get(self.nest_server.nest.remote("SetDefaults", model, params, val=val))

    def CopyModel(self, existing, new, params=None):
        return ray.get(self.nest_server.CopyModel.remote((existing, new, params)))

    def ConnectionRules(self):
        return ray.get(self.nest_server.nest.remote("ConnectionRules"))

    def DisableStructuralPlasticity(self):
        return ray.get(self.nest_server.nest.remote("DisableStructuralPlasticity"))

    def EnableStructuralPlasticity(self):
        return ray.get(self.nest_server.nest.remote("EnableStructuralPlasticity"))

    def ResetKernel(self):
        return ray.get(self.nest_server.nest.remote("ResetKernel"))

    def GetKernelStatus(self, *args):
        return ray.get(self.nest_server.nest.remote("GetKernelStatus", *args))

    def SetKernelStatus(self, values_dict):
        return ray.get(self.nest_server.nest.remote("SetKernelStatus", values_dict))

    def set_verbosity(self, level):
        return ray.get(self.nest_server.nest.remote("set_verbosity", level))

    def get_verbosity(self):
        return ray.get(self.nest_server.nest.remote("get_verbosity"))

    def sysinfo(self):
        return ray.get(self.nest_server.nest.remote("sysinfo"))

    def help(self, obj=None, return_text=True):
        # TODO: a warning for when return_text = False
        if isinstance(obj, RayNodeCollection):
            return ray.get(self.nest_server.help(obj=self._node_collection_to_gids(obj), return_text=True))
        elif isinstance(obj, RaySynapseCollection):
            return ray.get(self.nest_server.help(obj=self._synapse_collection_to_source_target_dict(obj),
                                                 return_text=True))
        else:
            return ray.get(self.nest_server.help(obj=obj, return_text=True))

    def Install(self, module_name):
        return ray.get(self.nest_server.nest.remote("Install", module_name))

    def PrintNodes(self):
        return ray.get(self.nest_server.nest.remote("PrintNodes"))

    def authors(self):
        return ray.get(self.nest_server.nest.remote("authors"))

    def get_argv(self):
        return ray.get(self.nest_server.nest.remote("get_argv"))

    @property
    def is_running(self):
        if self.run_task_ref_obj is None:
            return False
        else:
            done, running = ray.wait([self.run_task_ref_obj], timeout=0)
            if len(running):
                return True
            else:
                return False

    @property
    def block_run(self):
        if self.is_running:
            ray.get(self.run_task_ref_obj)
        self.run_task_ref_obj = None
        return self.run_task_ref_obj

    def _run(self, method, time):
        if not self.is_running:
            if method.lower() == "simulate":
                method = "Simulate"
            else:
                method = "Run"
            run_task_ref_obj = None
            while run_task_ref_obj is None:
                run_task_ref_obj = self.nest_server.nest.remote(method, time)
            self.run_task_ref_obj = run_task_ref_obj
        return self.run_task_ref_obj

    def Prepare(self):
        return ray.get(self.nest_server.nest.remote("Prepare"))

    def Run(self, time):
        return self._run("Run", time)

    def RunLock(self, time, ref_objs=[]):
        if len(ref_objs):
            ray.get(ref_objs)
        return self._run("Run", time)

    def Simulate(self, time):
        return self._run("Simulate", time)

    def Cleanup(self):
        return ray.get(self.nest_server.nest.remote("Cleanup"))
