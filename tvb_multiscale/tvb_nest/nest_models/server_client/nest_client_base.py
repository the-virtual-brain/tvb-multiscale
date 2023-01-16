# -*- coding: utf-8 -*-

import ray

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.tvb_nest.nest_models.server_client.node_collection import NodeCollection
from tvb_multiscale.tvb_nest.nest_models.server_client.synapse_collection import SynapseCollection


# def decode_numpy(inarr):
#     outlist = []
#     for elem in inarr:
#         if elem.size == 0:
#             outlist.append(elem)
#         elif elem.size == 1:
#             outlist.append(elem.item())
#         else:
#             outlist.append(decode_numpy(elem))
#     return outarr


# import numpy as np
# def decode_list(lin):
#     lout = list()
#     for ll in lin:
#         if isinstance(ll, np.ndarray):
#             print("\nnumpy array!: %s" % str(ll))
#             lout.append(ll.tolist())
#             print("\ntolist: %s" % str(lout[-1]))
#         elif isinstance(ll, dict):
#             lout.append(decode_dict(ll))
#         elif isinstance(ll, list):
#             lout.append(decode_list(ll))
#         else:
#             lout.append(ll)
#     return lout
#
#
# def decode_tuple(tin):
#     return tuple(decode_list(list(tin)))
#
#
# def decode_dict(din):
#     dout = dict()
#     for key, val in din.items():
#         if isinstance(val, np.ndarray):
#             # print("\nnumpy array!: %s" % str(val))
#             dout[key] = val.tolist()
#             # print("\nnumpy array!: %s" % str(dout[key]))
#         elif isinstance(val, dict):
#             dout[key] = decode_dict(val)
#         elif isinstance(val, list):
#             dout[key] = decode_list(val)
#         elif isinstance(val, tuple):
#             dout[key] = decode_tuple(val)
#         else:
#             dout[key] = val
#     return dout
#
#
# def decode_args_kwargs(argsin, kwargsin):
#     return decode_list(argsin), decode_dict(kwargsin)


class NESTClientBase(object):

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def request(self, call, *args, **kwargs):
        raise NotImplementedError

    def __getattr__(self, attr):
        return lambda *args, **kwargs: self.request(attr, *args, **kwargs)

    def _gids(self, gids):
        return tuple(ensure_list(gids))

    def _node_collection_to_gids(self, node_collection):
        return node_collection.gids

    def NodeCollection(self, gids):
        return NodeCollection(self, self._gids(gids))

    def SynapseCollection(self, conns_dict):
        return SynapseCollection(self, conns_dict)

    def _synapse_collection_to_source_target_dict(self, synapse_collection):
        return {"source": synapse_collection.source, "target": synapse_collection.target,
                "synapse_model": synapse_collection.synapse_model}

    def _nodes(self, nodes):
        if isinstance(nodes, NodeCollection):
            return self._node_collection_to_gids(nodes)
        elif isinstance(nodes, SynapseCollection):
            return self._synapse_collection_to_source_target_dict(nodes)
        else:
            return nodes

    def Create(self, model, n=1, params=None, positions=None):
        return self.NodeCollection(self.request("Create", model, n=n, params=params, positions=positions))

    def Connect(self, pre, post, conn_spec=None, syn_spec=None, **kwargs):
        return self.request("Connect",
                            self._node_collection_to_gids(pre), self._node_collection_to_gids(post),
                            conn_spec=conn_spec, syn_spec=syn_spec)

    def Disconnect(self, pre, post, conn_spec='one_to_one', syn_spec='static_synapse'):
        return self.request("Disconnect",
                            self._node_collection_to_gids(pre), self._node_collection_to_gids(post),
                            conn_spec=conn_spec, syn_spec=syn_spec)

    def GetLocalNodeCollection(self, node_collection):
        if len(node_collection):
            if block:
                return self.NodeCollection(
                    self.request("GetLocalNodeCollection", self._node_collection_to_gids(node_collection)))
        else:
            return self.NodeCollection(())

    def _get_kwargs_for_GetConnections(self, **kwargs):
        for source_or_target in ["source", "target"]:
            if source_or_target in kwargs:
                if kwargs[source_or_target] is None:
                    del kwargs[source_or_target]
                elif isinstance(kwargs[source_or_target], NodeCollection):
                    kwargs[source_or_target] = self._node_collection_to_gids(kwargs[source_or_target])
        return kwargs

    def GetConnections(self, source=None, target=None, synapse_model=None, synapse_label=None):
        conns = self.request("GetConnections",
                             **self._get_kwargs_for_GetConnections(source=source, target=target,
                                                                   synapse_model=synapse_model,
                                                                   synapse_label=synapse_label))
        if isinstance(conns, dict):
            return self.SynapseCollection(conns)
        return conns

    def GetNodes(self, properties={}, local_only=False):
        return self.NodeCollection(
                self.request("GetNodes", properties=properties, local_only=local_only))

    # def GetStatus(self, nodes, keys=None, output=''):
    #     return self.request("GetStatus", self._nodes(nodes), keys=keys, outpu=output)
    #
    # def SetStatus(self, nodes, params, val=None):
    #     return self.request("SetStatus", self._nodes(nodes), params, val=val)
    #
    # def Models(self):
    #     return self.request("Models")
    #
    # def GetDefaults(self, model, keys=None, output=''):
    #     return self.request("GetDefaults", model, keys=keys, output=output)
    #
    # def SetDefaults(self, model, params, val=None):
    #     return self.request("SetDefaults", model, params, val=val)
    #
    # def CopyModel(self, existing, new, params=None):
    #     return self.request("CopyModel", existing, new, params=params)
    #
    # def ConnectionRules(self):
    #     return self.request("ConnectionRules")
    #
    # def DisableStructuralPlasticity(self):
    #     return self.request("DisableStructuralPlasticity")
    #
    # def EnableStructuralPlasticity(self):
    #     return self.request("EnableStructuralPlasticity")
    #
    # def ResetKernel(self):
    #     return self.request("ResetKernel")
    #
    # def GetKernelStatus(self, *args):
    #     return self.request("GetKernelStatus", *args)
    #
    # def SetKernelStatus(self, values_dict):
    #     return self.request("SetKernelStatus", values_dict)
    #
    # def set_verbosity(self, level):
    #     return self.request("set_verbosity", level)
    #
    # def get_verbosity(self):
    #     return self.request("get_verbosity")
    #
    # def sysinfo(self):
    #     return self.request("sysinfo")
    #
    # def help(self, obj=None, return_text=True):
    #     # TODO: a warning for when return_text = False
    #     return self.request("help", obj=self._nodes(obj), return_text=True)
    #
    # def Install(self, module_name):
    #     return self.request("Install", module_name)
    #
    # def PrintNodes(self):
    #     return self.request("PrintNodes")
    #
    # def authors(self, block=True):
    #     return self.request("authors")
    #
    # def get_argv(self):
    #     return self.request("get_argv")
    #
    # def Prepare(self):
    #     return self.request("Prepare")
    #
    # def Run(self, time):
    #     return self.request("Run", time)
    #
    # def Simulate(self, time):
    #     return self.request("Simulate", time, block)
    #
    # def Cleanup(self):
    #     return self.request("Cleanup")


class NESTClientAsyncBase(NESTClientBase):

    run_task_ref_obj = None

    def __init__(self, *args, **kwargs):
        self.run_task_ref_obj = None

    def __getstate__(self):
        d = super(NESTClientAsyncBase, self).__getstate__()
        d.update({"run_task_ref_obj": self.run_task_ref_obj})
        return d

    def __setstate__(self, d):
        super(NESTClientAsyncBase, self).__setstate__(d)
        self.run_task_ref_obj = d.get("run_task_ref_obj", self.run_task_ref_obj)

    def async_request(self, call, *args, **kwargs):
        raise NotImplementedError

    def __getattr__(self, attr):
        return lambda *args, block=True, **kwargs: self.request(attr, *args, **kwargs) if block \
                                                        else self.async_request(attr, *args, **kwargs)

    def Create(self, model, n=1, params=None, positions=None, block=True):
        if block:
           return super(NESTClientAsyncBase, self).Create(model, n=n, params=params, positions=positions)
        else:
            return self.async_request("Create", model, n=n, params=params, positions=positions)

    def Connect(self, pre, post, conn_spec=None, syn_spec=None, block=True, **kwargs):
        if block:
            return super(NESTClientAsyncBase, self).Connect(pre, post, conn_spec=conn_spec, syn_spec=syn_spec)
        else:
            return self.async_request("Connect",
                                      self._node_collection_to_gids(pre), self._node_collection_to_gids(post),
                                      conn_spec=conn_spec, syn_spec=syn_spec)

    def Disconnect(self, pre, post, conn_spec='one_to_one', syn_spec='static_synapse', block=True):
        if block:
            return super(NESTClientAsyncBase, self).Disconnect(pre, post, conn_spec=conn_spec, syn_spec=syn_spec)
        else:
            return self.async_request("Disconnect",
                                      self._node_collection_to_gids(pre), self._node_collection_to_gids(post),
                                      conn_spec=conn_spec, syn_spec=syn_spec)

    def GetLocalNodeCollection(self, node_collection, block=True):
        if len(node_collection):
            if block:
                return super(NESTClientAsyncBase, self).GetLocalNodeCollection(node_collection)
            else:
                return self.async_request("GetLocalNodeCollection", self._node_collection_to_gids(node_collection))
        else:
            return self.NodeCollection(())

    def GetConnections(self, source=None, target=None, synapse_model=None, synapse_label=None, block=True):
        if block:
            return super(NESTClientAsyncBase, self).GetConnections(source=source, target=target,
                                                                   synapse_model=synapse_model,
                                                                   synapse_label=synapse_label)
        else:
            return self.async_request("GetConnections",
                                      **self._get_kwargs_for_GetConnections(source=source, target=target,
                                                                            synapse_model=synapse_model,
                                                                            synapse_label=synapse_label))

    def GetNodes(self, properties={}, local_only=False, block=True):
        if block:
            return super(NESTClientAsyncBase, self).GetNodes(properties=properties, local_only=local_only)
        else:
            return self.async_request("GetNodes", properties=properties, local_only=local_only)

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

    def _run(self, method, time, block=False):
        if not self.is_running:
            if method.lower() == "simulate":
                method = "Simulate"
            else:
                method = "Run"
            run_task_ref_obj = None
            while run_task_ref_obj is None:
                if block:
                    self.request(method, time)
                    break
                else:
                    run_task_ref_obj = self.async_request(method, time)
            self.run_task_ref_obj = run_task_ref_obj
        return self.run_task_ref_obj

    # def Prepare(self, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).Prepare()
    #     else:
    #         return self.async_request("Prepare")

    def Run(self, time, block=False):
        return self._run("Run", time, block)

    def RunLock(self, time, ref_objs=[], block=False):
        if len(ref_objs):
            ray.get(ref_objs)
        return self._run("Run", time, block=block)

    def Simulate(self, time, block=False):
        return self._run("Simulate", time, block)

    # def Cleanup(self, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).Cleanup()
    #     else:
    #         return self.async_request("Cleanup")
    #
    # def GetStatus(self, nodes, keys=None, output='', block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).GetStatus(nodes, keys=keys, output=output)
    #     else:
    #         return self.async_request("GetStatus", self._nodes(nodes), keys=keys, outpu=output)
    #
    # def SetStatus(self, nodes, params, val=None, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).SetStatus(nodes, params, val=val)
    #     else:
    #         return self.async_request("SetStatus", self._nodes(nodes), params, val=val)
    #
    # def Models(self, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).Models()
    #     else:
    #         return self.async_request("Models")
    #
    # def GetDefaults(self, model, keys=None, output='', block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).GetDefaults(model, keys=keys, output=output)
    #     else:
    #         return self.async_request("GetDefaults", model, keys=keys, output=output)
    #
    # def SetDefaults(self, model, params, val=None, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).SetDefaults(model, params, val=val)
    #     else:
    #         return self.async_request("SetDefaults", model, params, val=val)
    #
    # def CopyModel(self, existing, new, params=None, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).CopyModel(existing, new, params=params)
    #     else:
    #         return self.async_request("CopyModel", existing, new, params=params)
    #
    # def ConnectionRules(self, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).ConnectionRules()
    #     else:
    #         return self.async_request("ConnectionRules")
    #
    # def DisableStructuralPlasticity(self, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).DisableStructuralPlasticity()
    #     else:
    #         return self.async_request("DisableStructuralPlasticity")
    #
    # def EnableStructuralPlasticity(self, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).EnableStructuralPlasticity()
    #     else:
    #         return self.async_request("EnableStructuralPlasticity")
    #
    # def ResetKernel(self, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).ResetKernel()
    #     else:
    #         return self.async_request("ResetKernel")
    #
    # def GetKernelStatus(self, *args, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).GetKernelStatus()
    #     else:
    #         return self.async_request("GetKernelStatus", *args)
    #
    # def SetKernelStatus(self, values_dict, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).SetKernelStatus()
    #     else:
    #         return self.async_request("SetKernelStatus", values_dict)
    #
    # def set_verbosity(self, level, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).set_verbosity()
    #     else:
    #         return self.async_request("set_verbosity", level)
    #
    # def get_verbosity(self, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).get_verbosity()
    #     else:
    #         return self.async_request("get_verbosity")
    #
    # def sysinfo(self, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).sysinfo()
    #     else:
    #         return self.async_request("sysinfo")
    #
    # def help(self, obj=None, return_text=True, block=True):
    #     # TODO: a warning for when return_text = False
    #     if block:
    #         return super(NESTClientAsyncBase, self).help(obj=obj, return_text=True)
    #     else:
    #         return self.async_request("help", obj=self._nodes(obj), return_text=True)
    #
    # def Install(self, module_name, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).Install(module_name)
    #     else:
    #         return self.async_request("Install", module_name)
    #
    # def PrintNodes(self, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).PrintNodes()
    #     else:
    #         return self.async_request("PrintNodes")
    #
    # def authors(self, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).authors()
    #     else:
    #         return self.async_request("authors")
    #
    # def get_argv(self, block=True):
    #     if block:
    #         return super(NESTClientAsyncBase, self).get_argv()
    #     else:
    #         return self.async_request("get_argv")
