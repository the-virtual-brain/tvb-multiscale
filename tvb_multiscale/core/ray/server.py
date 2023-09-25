# -*- coding: utf-8 -*-

import ray


class RayServer(object):

    def __getattribute__(self, attr):
        return super(RayServer, self).__getattribute__(attr)

    def __setattr__(self, attr, val):
        return super(RayServer, self).__setattr__(attr, val)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass


def rayfy(input_class, ray_server_base=RayServer):
    d = dict(ray_server_base.__dict__)
    d.update(dict(input_class.__dict__))
    new_ray_server_type_name = "%s%s" % (ray_server_base.__name__, input_class.__name__)
    new_ray_server_type = type(new_ray_server_type_name, (input_class, ray_server_base,), d)
    return ray.remote(new_ray_server_type)


def create_ray_server(input_class, *args, **kwargs):
    return rayfy(input_class).remote(*args, **kwargs)
