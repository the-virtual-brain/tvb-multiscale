# -*- coding: utf-8 -*-

import requests
from werkzeug.exceptions import BadRequest

import numpy as np

from nest_client import NESTClient as NESTServerClientBase

from tvb_multiscale.tvb_nest.nest_models.server_client.nest_client_base import NESTClientBase  # , decode_args_kwargs


def encode(response):
    if response.ok:
        return response.json()
    elif response.status_code == 400:
        raise BadRequest(response.text)


# import json
# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)


def nest_server_request(url, headers, call, *args, **kwargs):
    # args2, kwargs2 = decode_args_kwargs(args, kwargs)
    # kwargs2.update({'args': args2})
    # response = requests.post(url + 'api/' + call, data=json.dumps(kwargs2.copy(), cls=NpEncoder), headers=headers)
    kwargs.update({'args': args})
    response = requests.post(url + 'api/' + call, json=kwargs, headers=headers)
    if not response.ok:
        print("\nResponse NOT OK!:")
        print(kwargs)
    return encode(response)


class NESTServerClient(NESTServerClientBase, NESTClientBase):

    host = 'localhost'
    port = 52425

    def __init__(self, host='localhost', port=52425):
        NESTServerClientBase.__init__(self, host=host, port=port)
        NESTClientBase.__init__(self)

    def __getstate__(self):
        d = {"host": self.host, "port": self.port,
             "url": self.url, "headers": self.headers}
        d.update(NESTClientBase.__getstate__(self))
        return d

    def __setstate__(self, d):
        self.host = d.get("host", self.host)
        self.port = d.get("host", self.port)
        self.url = d.get("url", 'http://{}:{}/'.format(self.host, self.port))
        self.headers = d.get("headers", {'Content-type': 'application/json', 'Accept': 'text/plain'})
        NESTClientBase.__setstate__(self)

    def _node_collection_to_gids(self, node_collection):
        return [int(gid) for gid in NESTClientBase._node_collection_to_gids(self, node_collection)]

    def request(self, call, *args, **kwargs):
        return nest_server_request(self.url, self.headers, call, *args, **kwargs)

    def get(self, nodes, *params, **kwargs):
        outputs = self.request("GetStatus", self._nodes(nodes), *params, **kwargs)
        if len(params) <= 1:
            # if len(params) == 0, tuple(dict(params, params_vals)) of all params
            # elif len(params) == 1, tuple(values) of a single param
            return outputs[0]
        else:
            # if len(params) > 0, tuple of param_values per node, needs transposing to be returned as a dict
            return dict(zip(params, np.array(outputs).T))

    def set(self, nodes, params=None, **kwargs):
        return self.request("SetStatus", self._nodes(nodes), params=params, **kwargs)
