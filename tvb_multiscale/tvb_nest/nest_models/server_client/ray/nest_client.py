# -*- coding: utf-8 -*-

import ray

from tvb_multiscale.tvb_nest.nest_models.server_client.nest_client_base import NESTClientAsyncBase  #, decode_args_kwargs
from tvb_multiscale.tvb_nest.nest_models.server_client.node_collection import NodeCollection
from tvb_multiscale.tvb_nest.nest_models.server_client.synapse_collection import SynapseCollection


class RayNESTClient(NESTClientAsyncBase):

    nest_server = None

    def __init__(self, nest_server):
        super(RayNESTClient, self).__init__()
        self.nest_server = nest_server

    def __getstate__(self):
        d = super(RayNESTClient, self).__getstate__()
        d["nest_server"] = self.nest_server

    def __setstate__(self, d):
        super(RayNESTClient, self).__setstate__(d)
        self.nest_server = d.get("nest_server", None)

    def async_request(self, call, *args, **kwargs):
        if call in self.nest_server.__dict__.keys():
            return getattr(self.nest_server, call).remote(*args, **kwargs)
        else:
            return getattr(self.nest_server, "nest").remote(call, *args, **kwargs)

    def request(self, call, *args, **kwargs):
        # args2, kwargs2 = decode_args_kwargs(args, kwargs)
        # return ray.get(self.async_request(call, *args2, **kwargs2))
        return ray.get(self.async_request(call, *args, **kwargs))

    def get(self, nodes, *params, block=True, **kwargs):
        if block:
            return self.request("get", self._nodes(nodes), *params, **kwargs)
        else:
            return self.async_request("get", self._nodes(nodes), *params, **kwargs)

    def set(self, nodes, params=None, block=True, **kwargs):
        if block:
            return self.request("set", self._nodes(nodes), params=params, **kwargs)
        else:
            return self.async_request("set", self._nodes(nodes), params=params, **kwargs)

    # def Connect(self, pre, post, conn_spec=None, syn_spec=None, block=True, **kwargs):
    #     output = \
    #         super(RayNESTClient, self).Connect(pre, post, conn_spec=conn_spec, syn_spec=syn_spec, block=block, **kwargs)
    #     if isinstance(output, dict):
    #         return self.SynapseCollection(output)
    #     return output
