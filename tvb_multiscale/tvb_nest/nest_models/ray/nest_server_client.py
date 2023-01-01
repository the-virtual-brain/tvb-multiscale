# -*- coding: utf-8 -*-

import requests
from werkzeug.exceptions import BadRequest
import ray

import numpy as np

from NESTServerClient import NESTServerClient

from tvb_multiscale.tvb_nest.nest_models.ray.nest_client import RayNESTClientBase


def encode(response):
    if response.ok:
        return response.json()
    elif response.status_code == 400:
        raise BadRequest(response.text)


def nest_server_request(url, headers, call, *args, **kwargs):
    kwargs.update({'args': args})
    response = requests.post(url + 'api/' + call, json=kwargs, headers=headers)
    return encode(response)


@ray.remote
def ray_nest_server_request(url, headers, call, *args, **kwargs):
    return nest_server_request(url, headers, call, *args, **kwargs)


# class RayNESTRequest(object):
#     host = 'localhost'
#     port = 5000
#     url = 'http://{}:{}/'.format('localhost', 5000)
#     headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
#
#     def __init__(self, host='localhost', port=5000):
#         self.host = host
#         self.port = port
#         self.url = 'http://{}:{}/'.format('localhost', 5000)
#         self.headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
#
#     def __call__(self, call, *args, **kwargs):
#         kwargs.update({'args': args})
#         response = requests.post(self.url + 'api/' + call, json=kwargs, headers=self.headers)
#         return encode(response)


class RayNESTServerClient(RayNESTClientBase, NESTServerClient):

    host = 'localhost'
    port = 5000

    def __init__(self, host='localhost', port=5000):
        RayNESTClientBase.__init__(self)
        NESTServerClient.__init__(self, host=host, port=port)

    def __getstate__(self):
        d = RayNESTClientBase.__getstate__(self)
        d.update({"host": self.host, "port": self.port,
                 "url": self.url, "headers": self.headers})
        return d

    def __setstate__(self, d):
        RayNESTClientBase.__setstate__(self)
        self.host = d.get("host", self.host)
        self.port = d.get("host", self.port)
        self.url = d.get("url", 'http://{}:{}/'.format(self.host, self.port))
        self.headers = d.get("headers", {'Content-type': 'application/json', 'Accept': 'text/plain'})
        # self.ray = RayNESTRequest.remote(host=self.host, port=self.port)

    def _node_collection_to_gids(self, node_collection):
        return [int(gid) for gid in RayNESTClientBase._node_collection_to_gids(self, node_collection)]

    def request(self, call, *args, **kwargs):
        return nest_server_request(self.url, self.headers, call, *args, **kwargs)

    def async_request(self, call, *args, **kwargs):
        return ray_nest_server_request.remote(self.url, self.headers, call, *args, **kwargs)

    def get(self, nodes, *params, **kwargs):
        if self._block(kwargs):
            outputs = self.request("GetStatus", self._nodes(nodes), *params, **kwargs)
            if len(params) <= 1:
                # if len(params) == 0, tuple(dict(params, params_vals)) of all params
                # elif len(params) == 1, tuple(values) of a single param
                return outputs[0]
            else:
                # if len(params) > 0, tuple of param_values per node, needs transposing to be returned as a dict
                return dict(zip(params, np.array(outputs).T))
        else:
            return self.async_request("GetStatus", self._nodes(nodes), *params, **kwargs)

    def set(self, nodes, params=None, **kwargs):
        if self._block(kwargs):
            return self.request("SetStatus", self._nodes(nodes), params=params, **kwargs)
        else:
            return self.async_request("SetStatus", self._nodes(nodes), params=params, **kwargs)
