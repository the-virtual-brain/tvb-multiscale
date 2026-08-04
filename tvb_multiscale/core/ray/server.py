# -*- coding: utf-8 -*-

import ray
from tvb_multiscale.core.neotraits import HasTraits


class RayServer(object):

    def __getattribute__(self, attr):
        return super(RayServer, self).__getattribute__(attr)

    def __setattr__(self, attr, val):
        return super(RayServer, self).__setattr__(attr, val)

    def __getstate__(self):
        d ={}
        for key, val in self.__dict__.items():
            if key[0] != "_" and not callable(getattr(self, key)):
                d[key] = val
        return d

    def __setstate__(self, d):
        for key, val in d.items():
            setattr(self, key, val)


def rayfy(input_class, ray_server_base=RayServer):
    if not isinstance(input_class, HasTraits) and not issubclass(input_class, HasTraits):
        d = dict(ray_server_base.__dict__)
        d.update(dict(input_class.__dict__))
        new_ray_server_type_name = "%s%s" % (ray_server_base.__name__, input_class.__name__)
        remote_class = type(new_ray_server_type_name, (input_class, ray_server_base,), d)
    else:
        remote_class = input_class
    for iP, parent in enumerate(remote_class.__mro__):
        if iP > 0:
            ray.remote(parent)
    return ray.remote(remote_class)


def create_ray_server(input_class, *args, **kwargs):
    return rayfy(input_class).remote(*args, **kwargs)
