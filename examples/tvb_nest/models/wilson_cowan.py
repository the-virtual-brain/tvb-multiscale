# -*- coding: utf-8 -*-

from examples.tvb_nest.example import main_example
from examples.models.wilson_cowan import wilson_cowan_example as wilson_cowan_example_base

from tvb_multiscale.tvb_nest.interfaces.models.builders.wilson_cowan \
    import WilsonCowanBuilder as WilsonCowanTVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.interfaces.models.builders.wilson_cowan \
    import WilsonCowanMultisynapseBuilder as WilsonCowanMultisynapseTVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.nest_models.models.wilson_cowan import \
    WilsonCowanBuilder, WilsonCowanMultisynapseBuilder


def wilson_cowan_example(**kwargs):

    if kwargs.pop("multisynapse", True):
        nest_model_builder = WilsonCowanMultisynapseBuilder()
        tvb_nest_model_builder = WilsonCowanMultisynapseTVBNESTInterfaceBuilder
    else:
        nest_model_builder = WilsonCowanBuilder()
        tvb_nest_model_builder = WilsonCowanTVBNESTInterfaceBuilder

    return main_example(wilson_cowan_example_base, nest_model_builder, tvb_nest_model_builder, **kwargs)


if __name__ == "__main__":
    import sys
    if sys.argv[-1] == "1":
        wilson_cowan_example(multisynapse=True)
    else:
        wilson_cowan_example(multisynapse=False)
