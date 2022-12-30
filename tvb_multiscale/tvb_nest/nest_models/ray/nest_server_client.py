# -*- coding: utf-8 -*-

import requests
from werkzeug.exceptions import BadRequest

from NESTServerClient import NESTServerClient

from tvb_multiscale.tvb_nest.nest_models.ray.nest_client import RayNESTClientBase


def encode(response):
    if response.ok:
        return response.json()
    elif response.status_code == 400:
        raise BadRequest(response.text)


@ray.remote
def ray_nest_request(url, headers, call, *args, **kwargs):
    kwargs.update({'args': args})
    response = requests.post(url + 'api/' + call, json=kwargs, headers=headers)
    return encode(response)


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

    def request(self, call, *args, **kwargs):
        kwargs.update({'args': args})
        response = requests.post(url + 'api/' + call, json=kwargs, headers=headers)
        return encode(response)

    def async_request(self, call, *args, **kwargs):
        return ray_nest_request.remote(self.url, self.headers, call, *args, **kwargs)
