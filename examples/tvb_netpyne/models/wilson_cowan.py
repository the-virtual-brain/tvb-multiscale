# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_netpyne.interfaces.models.wilson_cowan import WilsonCowanTVBNetpyneInterfaceBuilder
from tvb_multiscale.tvb_netpyne.netpyne_models.models.wilson_cowan import WilsonCowanBuilder

from examples.tvb_netpyne.example import main_example
from examples.models.wilson_cowan import wilson_cowan_example as wilson_cowan_example_base


def wilson_cowan_example(**kwargs):
    # model_params = {"model_params": {"lamda": 0.5}}
    # kwargs.update(model_params)

    netpyne_model_builder = WilsonCowanBuilder()
    tvb_netpyne_interface_builder = WilsonCowanTVBNetpyneInterfaceBuilder()

    return main_example(wilson_cowan_example_base, netpyne_model_builder, tvb_netpyne_interface_builder, **kwargs)


if __name__ == "__main__":
    wilson_cowan_example()
