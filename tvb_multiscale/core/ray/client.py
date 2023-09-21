# -*- coding: utf-8 -*-

from types import FunctionType, MethodType

import ray

from tvb_multiscale.core.ray.server import create_ray_server


class RayClient(object):

    _own_attributes = ["ray", "ray_server"]

    def __init__(self, ray_server, ray_module=ray):
        self.ray = ray_module
        self.ray_server = ray_server
        super(RayClient, self).__init__()

    def __getattr__(self, attr):
        return ray.get(self.ray_server.__getattribute__.remote(attr))

    def __setattr__(self, attr, value):
        if attr in self._own_attributes:
            super(RayClient, self).__setattr__(attr, value)
        else:
            ray.get(self.ray_server.__setattr__.remote(attr, value))

    def __getstate__(self):
        return {"ray_server": self.ray_server}

    def __setstate__(self, d):
        self.ray_server = d.get("ray_server", None)


def create_ray_client_function(name, parallel=False):
    if parallel:
        funcode = compile("def %s(cls, *args, **kwargs): "
                          "return getattr(cls.ray_server, '%s').remote(*args, **kwargs)"
                          % (name, name), "<string>", "exec")
    else:
        funcode = compile("def %s(cls, *args, **kwargs): "
                          "return cls.ray.get(getattr(cls.ray_server, '%s').remote(*args, **kwargs))"
                          % (name, name), "<string>", "exec")
    return FunctionType(funcode.co_consts[0], globals(), name)


def create_ray_client_type_methods(ray_server, target_server_class, non_blocking_methods=[], attrs_dict={}):
    # ...and add methods and properties derived from the target_server_class instance and its ray_server:
    for server_method in ray_server.__dict__['_ray_method_signatures']:
        if hasattr(target_server_class, server_method) and \
                server_method not in ["__init__", "__getattribute__", "__getattr__", "__setattr__"]:
            if server_method in non_blocking_methods:
                parallel = True
            else:
                parallel = False
            attrs_dict[server_method] = create_ray_client_function(server_method, parallel)
    return attrs_dict


def create_ray_client(target_server_class, ray_client_type=RayClient, non_blocking_methods=[], *args, **kwargs):
    # Instantiate a Ray server to an instance of target_server_class:
    ray_server = create_ray_server(target_server_class, *args, **kwargs)
    # Generate a new ray client type based on ray_client_type...
    # ...with the proper name:
    new_ray_client_type_name = "%s%s" % (ray_client_type.__name__, target_server_class.__name__)
    # ...and with the class' attributes' dictionary with all necessary methods:
    d = create_ray_client_type_methods(ray_server, target_server_class, non_blocking_methods,
                                       {'ray_server': None, 'ray': ray})
    # ...this is the new type:
    new_ray_client_type = type(new_ray_client_type_name, (ray_client_type,), d)
    # Finally, generate an instance of this client and return it:
    return new_ray_client_type(ray_server, ray)

