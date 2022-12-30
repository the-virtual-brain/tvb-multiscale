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


class RayNESTClientBase(object):

    obj_ref = list()

    def __init__(self, *args, **kwargs):
        self.obj_ref = list()

    def __getstate__(self):
        return {"obj_refs": list(self.obj_refs)}

    def __setstate__(self, d):
        self.obj_refs = d.get("obj_refs", self.obj_refs)

    def request(self, call, *args, **kwargs):
        pass

    def async_request(self, call, *args, **kwargs):
        pass

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

    def _nodes(self, nodes):
        if isinstance(nodes, RayNodeCollection):
            return self._node_collection_to_gids(nodes)
        elif isinstance(nodes, RaySynapseCollection):
            return self._synapse_collection_to_source_target_dict(nodes)
        else:
            return nodes

    def _block(self, kwargs):
        return kwargs.pop("block", True)

    def get(self, nodes, *params, **kwargs):
        if self._block(kwargs):
            return self.request("get", self._nodes(nodes), *params, **kwargs)
        else:
            return self.async_request("get", self._nodes(nodes), *params, **kwargs)

    def set(self, nodes, params=None, **kwargs):
        if self._block(kwargs):
            return self.request("set", self._nodes(nodes), *params, **kwargs)
        else:
            return self.async_request("set", self._nodes(nodes), *params, **kwargs)

    def GetStatus(self, nodes, attrs=None, output=None, block=True):
        if block:
            return self.request("GetStatus", self._nodes(nodes), attrs, output)
        else:
            return self.async_request("GetStatus", self._nodes(nodes), attrs, output)

    def SetStatus(self, nodes, params, val=None, block=True):
        if block:
            return self.request("SetStatus", self._nodes(nodes), params, val)
        else:
            return self.async_request("SetStatus", self._nodes(nodes), params, val)

    def Create(self, model, n=1, params=None, positions=None, block=True):
        if block:
           return self.NodeCollection(self.request("Create", model, n=n, params=params, positions=positions))
        else:
            return self.async_request("Create", model, n=n, params=params, positions=positions)

    def Connect(self, pre, post, conn_spec=None, syn_spec=None, **kwargs):
        if self._block(kwargs):
            #return self.SynapseCollection(
            return self.request("Connect",
                             self._node_collection_to_gids(pre), self._node_collection_to_gids(post),
                             conn_spec=conn_spec, syn_spec=syn_spec)  # )
        else:
            return self.async_request("Connect",
                                      self._node_collection_to_gids(pre), self._node_collection_to_gids(post),
                                      conn_spec=conn_spec, syn_spec=syn_spec)

    def Disconnect(self, pre, post, conn_spec='one_to_one', syn_spec='static_synapse', block=True):
        if block:
            return self.SynapseCollection(
                self.request("Disconnect",
                             self._node_collection_to_gids(pre), self._node_collection_to_gids(post),
                             conn_spec=conn_spec, syn_spec=syn_spec))
        else:
            return self.async_request("Disconnect",
                                      self._node_collection_to_gids(pre), self._node_collection_to_gids(post),
                                      conn_spec=conn_spec, syn_spec=syn_spec)

    def GetLocalNodeCollection(self, node_collection, block=True):
        if len(node_collection):
            if block:
                return self.NodeCollection(
                    self.request("GetLocalNodeCollection", self._node_collection_to_gids(node_collection)))
            else:
                return self.async_request("GetLocalNodeCollection", self._node_collection_to_gids(node_collection))
        else:
            return self.NodeCollection(())

    def GetConnections(self, source=None, target=None, synapse_model=None, synapse_label=None, block=True):
        if source is not None:
            source = self._node_collection_to_gids(source)
        if target is not None:
            target = self._node_collection_to_gids(target)
        if block:
            return self.SynapseCollection(
                    self.request("GetConnections",
                                 source=source, target=target,
                                 synapse_model=synapse_model, synapse_label=synapse_label))
        else:
            return self.async_request("GetConnections",
                                      source=source, target=target,
                                      synapse_model=synapse_model, synapse_label=synapse_label)

    def GetNodes(self, properties={}, local_only=False, block=True):
        if block:
            return self.NodeCollection(
                self.request("GetNodes", properties=properties, local_only=local_only))
        else:
            return self.async_request("GetNodes", properties=properties, local_only=local_only)

    def Models(self, block=True):
        if block:
            return self.request("Models")
        else:
            return self.async_request("Models")

    def GetDefaults(self, model, keys=None, output='', block=True):
        if block:
            return self.request("GetDefaults", model, keys=keys, output=output)
        else:
            return self.async_request("GetDefaults", model, keys=keys, output=output)

    def SetDefaults(self, model, params, val=None, block=True):
        if block:
            return self.request("SetDefaults", model,params, val=val)
        else:
            return self.async_request("SetDefaults", model, params, val=val)

    def CopyModel(self, existing, new, params=None, block=True):
        if block:
            return self.request("CopyModel", existing, new, params=params)
        else:
            return self.async_request("CopyModel", existing, new, params=params)

    def ConnectionRules(self, block=True):
        if block:
            return self.request("ConnectionRules")
        else:
            return self.async_request("ConnectionRules")

    def DisableStructuralPlasticity(self, block=True):
        if block:
            return self.request("DisableStructuralPlasticity")
        else:
            return self.async_request("DisableStructuralPlasticity")

    def EnableStructuralPlasticity(self, block=True):
        if block:
            return self.request("EnableStructuralPlasticity")
        else:
            return self.async_request("EnableStructuralPlasticity")

    def ResetKernel(self, block=True):
        if block:
            return self.request("ResetKernel")
        else:
            return self.async_request("ResetKernel")

    def GetKernelStatus(self, *args, block=True):
        if block:
            return self.request("GetKernelStatus", *args)
        else:
            return self.async_request("GetKernelStatus", *args)

    def SetKernelStatus(self, values_dict, block=True):
        if block:
            return self.request("SetKernelStatus", values_dict)
        else:
            return self.async_request("SetKernelStatus", values_dict)

    def set_verbosity(self, level, block=True):
        if block:
            return self.request("set_verbosity", level)
        else:
            return self.async_request("set_verbosity", level)

    def get_verbosity(self, block=True):
        if block:
            return self.request("get_verbosity")
        else:
            return self.async_request("get_verbosity")

    def sysinfo(self, block=True):
        if block:
            return self.request("sysinfo")
        else:
            return self.async_request("sysinfo")

    def help(self, obj=None, return_text=True, block=True):
        # TODO: a warning for when return_text = False
        if block:
            return self.request("help", obj=self._nodes(obj), return_text=True)
        else:
            return self.async_request("help", obj=self._nodes(obj), return_text=True)

    def Install(self, module_name, block=True):
        if block:
            return self.request("Install", module_name)
        else:
            return self.async_request("Install", module_name)

    def PrintNodes(self, block=True):
        if block:
            return self.request("PrintNodes")
        else:
            return self.async_request("PrintNodes")

    def authors(self, block=True):
        if block:
            return self.request("authors")
        else:
            return self.async_request("authors")

    def get_argv(self, block=True):
        if block:
            return self.request("get_argv")
        else:
            return self.async_request("get_argv")

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

    def _run(self, method, time, block=True):
        if not self.is_running:
            if method.lower() == "simulate":
                method = "Simulate"
            else:
                method = "Run"
            run_task_ref_obj = None
            while run_task_ref_obj is None:
                if block:
                    self.request(method, time)
                else:
                    run_task_ref_obj = self.async_request(method, time)
            self.run_task_ref_obj = run_task_ref_obj
        return self.run_task_ref_obj

    def Prepare(self, block=True):
        if block:
            return self.request("Prepare")
        else:
            return self.async_request("Prepare")

    def Run(self, time, block=True):
        return self._run("Run", time, block)

    def RunLock(self, time, ref_objs=[]):
        if len(ref_objs):
            ray.get(ref_objs)
        return self._run("Run", time, block=True)

    def Simulate(self, time, block=True):
        return self._run("Simulate", time, block)

    def Cleanup(self, block=True):
        if block:
            return self.request("Cleanup")
        else:
            return self.async_request("Cleanup")


class RayNESTClient(RayNESTClientBase):

    def __init__(self, nest_server):
        self.nest_server = nest_server
        super(RayNESTClient, self).__init__()

    def __getstate__(self):
        d = super(RayNESTClient, self).__getstate__()
        d["nest_server"] = self.nest_server

    def __setstate__(self, d):
        self.nest_server = d.get("nest_server", None)
        super(RayNESTClient, self).__setstate__(d)

    def async_request(self, call, *args, **kwargs):
        if call in self.nest_server.__dict__.keys():
            return getattr(self.nest_server, call).remote(*args, **kwargs)
        else:
            return getattr(self.nest_server, "nest").remote(call, *args, **kwargs)

    def request(self, call, *args, **kwargs):
        return ray.get(self.async_request(call, *args, **kwargs))
