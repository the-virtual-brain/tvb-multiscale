# -*- coding: utf-8 -*-

from types import MethodType

import ray

from tvb_multiscale.core.ray.server import create_ray_server


class RayClient(object):

    ray_server = None

    def __init__(self, ray_server=None):
        self.ray_server = ray_server
        super(RayClient, self).__init__()

    def __getattr__(self, attr):
        return ray.get(self.ray_server.__getattribute__.remote(attr))

    def __setattr__(self, attr, value):
        if attr == "ray_server":
            super(RayClient, self).__setattr__(attr, value)
        else:
            ray.get(self.ray_server.__setattr__.remote(attr, value))

    def __getstate__(self):
        return {"ray_server": self.ray_server}

    def __setstate__(self, d):
        self.ray_server = d.get("ray_server", None)


def create_ray_client_function(name, parallel=False):

    def ray_function(self, *args, **kwargs):
        return ray.get(getattr(self.ray_server, name).remote(*args, **kwargs))

    def ray_parallel_function(self, *args, **kwargs):
        return getattr(self.ray_server, name).remote(*args, **kwargs)

    if parallel:
        return ray_parallel_function
    else:
        return ray_function


def add_server_methods_to_client(ray_client, ray_server, input_class, non_blocking_methods=[]):

    for server_method in ray_server.__dict__['_ray_method_signatures']:
        if hasattr(input_class, server_method) and \
                server_method not in ["__init__", "__getattribute__", "__getattr__", "__setattr__"]:
            if server_method in non_blocking_methods:
                fun = lambda self, *args, **kwargs: \
                    getattr(self.ray_server, server_method).remote(*args, **kwargs)
            else:
                fun = lambda self, *args, **kwargs: \
                    ray.get(getattr(self.ray_server, server_method).remote(*args, **kwargs))
            if isinstance(getattr(input_class, server_method), property):
                setattr(ray_client, server_method, property(fun))
            else:
                setattr(ray_client, server_method, MethodType(fun, ray_client))

    return ray_client


def create_ray_client(input_class, client_type=RayClient, non_blocking_methods=[], *args, **kwargs):

    ray_server = create_ray_server(input_class, *args, **kwargs)

    # RayClient.__name___ = "Ray%s" % input_class.__name__

    return add_server_methods_to_client(client_type(ray_server), ray_server, input_class, non_blocking_methods)
