# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_annarchy.config import Config, initialize_logger
from tvb_multiscale.tvb_annarchy.orchestrators import TVBANNarchySerialOrchestrator
from tvb_multiscale.tvb_annarchy.interfaces.models.default import DefaultTVBANNarchyInterfaceBuilder
from tvb_multiscale.tvb_annarchy.annarchy_models.models.default import DefaultExcIOBuilder

from examples.example import default_example as default_example_base


def main_example(model_fun_to_run, ann_model_builder, tvb_ann_model_builder, **kwargs):
    return model_fun_to_run(ann_model_builder, tvb_ann_model_builder, TVBANNarchySerialOrchestrator,
                            config_type=Config, logger_initializer=initialize_logger, **kwargs)


def default_example(**kwargs):
    return main_example(default_example_base, DefaultExcIOBuilder(), DefaultTVBANNarchyInterfaceBuilder(), **kwargs)


if __name__ == "__main__":
    import sys
    if sys.argv[-1] == "1":
        default_example(model="SPIKES")
    else:
        default_example(model="RATE")
