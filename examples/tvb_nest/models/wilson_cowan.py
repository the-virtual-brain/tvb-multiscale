# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_nest.interfaces.models.wilson_cowan import WilsonCowanTVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.interfaces.models.wilson_cowan import WilsonCowanMultisynapseTVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.nest_models.models.wilson_cowan import \
    WilsonCowanBuilder, WilsonCowanMultisynapseBuilder

from examples.tvb_nest.example import main_example
from examples.models.wilson_cowan import wilson_cowan_example as wilson_cowan_example_base


def wilson_cowan_example(**kwargs):

    if kwargs.pop("multisynapse", True):
        nest_model_builder = WilsonCowanMultisynapseBuilder()
        tvb_nest_model_builder = WilsonCowanMultisynapseTVBNESTInterfaceBuilder()
    else:
        nest_model_builder = WilsonCowanBuilder()
        tvb_nest_model_builder = WilsonCowanTVBNESTInterfaceBuilder()

    return main_example(wilson_cowan_example_base, nest_model_builder, tvb_nest_model_builder, **kwargs)


if __name__ == "__main__":
    import sys
    if sys.argv[-1] == "1":
        wilson_cowan_example(model="RATE", multisynapse=True)
    else:
        wilson_cowan_example(model="RATE", multisynapse=False)
