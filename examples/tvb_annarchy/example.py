# -*- coding: utf-8 -*-

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_annarchy.config import Config, initialize_logger
from tvb_multiscale.tvb_annarchy.interfaces.models.builders.default import DefaultInterfaceBuilder
from tvb_multiscale.tvb_annarchy.annarchy_models.models.default import DefaultExcIOBuilder

from examples.example import default_example as default_example_base


def main_example(model_fun_to_run, ann_model_builder, tvb_ann_model_builder, **kwargs):
    return model_fun_to_run(ann_model_builder, tvb_ann_model_builder,
                            config_type=Config, logger_initializer=initialize_logger, **kwargs)


def default_example(**kwargs):
    return main_example(default_example_base, DefaultExcIOBuilder(), DefaultInterfaceBuilder, **kwargs)


if __name__ == "__main__":
    default_example()
