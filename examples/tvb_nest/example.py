# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_nest.config import Config, initialize_logger
from tvb_multiscale.tvb_nest.orchestrators import TVBNESTSerialOrchestrator
from tvb_multiscale.tvb_nest.interfaces.models.default import DefaultTVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.interfaces.models.default import DefaultMultisynapseTVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.nest_models.models.default import DefaultExcIOBuilder
from tvb_multiscale.tvb_nest.nest_models.models.default import DefaultExcIOMultisynapseBuilder

from examples.example import default_example as default_example_base


def main_example(model_fun_to_run, nest_model_builder, tvb_nest_model_builder, **kwargs):
    return model_fun_to_run(nest_model_builder, tvb_nest_model_builder, TVBNESTSerialOrchestrator,
                            config_type=Config, logger_initializer=initialize_logger, **kwargs)


def default_example(**kwargs):

    if kwargs.pop("multisynapse", False):
        nest_model_builder = DefaultExcIOMultisynapseBuilder()
        tvb_nest_model_builder = DefaultMultisynapseTVBNESTInterfaceBuilder()
    else:
        nest_model_builder = DefaultExcIOBuilder()
        tvb_nest_model_builder = DefaultTVBNESTInterfaceBuilder()

    return main_example(default_example_base, nest_model_builder, tvb_nest_model_builder, **kwargs)


if __name__ == "__main__":
    import sys

    if sys.argv[-1] == "1":
        default_example(model="RATE", multisynapse=True)
    else:
        default_example(model="RATE")
