# -*- coding: utf-8 -*-

import ray

import numpy as np

from tvb_multiscale.tvb_nest.nest_models.server_client.nest_client_base import NESTClientAsyncBase
from tvb_multiscale.tvb_nest.nest_models.server_client.nest_server_client import NESTServerClient, nest_server_request


@ray.remote
def ray_nest_server_request(url, headers, call, *args, **kwargs):
    return nest_server_request(url, headers, call, *args, **kwargs)


class RayNESTServerClient(NESTServerClient, NESTClientAsyncBase):

    def __ini__(self, host='localhost', port=52425):
        NESTServerClient.__init__(self, host=host, port=port)
        NESTClientAsyncBase.__init__(self)

    def async_request(self, call, *args, **kwargs):
        return ray_nest_server_request.remote(self.url, self.headers, call, *args, **kwargs)

    def get(self, nodes, *params, block=True, **kwargs):
        if block:
            return NESTServerClient.get(self, nodes, *params, **kwargs)
        else:
            return self.async_request("GetStatus", self._nodes(nodes), *params, **kwargs)

    def set(self, nodes, params=None, block=True, **kwargs):
        if block:
            return NESTServerClient.set(self, nodes, params=params, **kwargs)
        else:
            return self.async_request("SetStatus", self._nodes(nodes), params=params, **kwargs)
