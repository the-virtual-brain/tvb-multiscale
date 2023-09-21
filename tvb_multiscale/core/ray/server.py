# -*- coding: utf-8 -*-


import ray


def rayfy(input_class):
    return ray.remote(input_class)


def create_ray_server(input_class, *args, **kwargs):
    return rayfy(input_class).remote(*args, **kwargs)

