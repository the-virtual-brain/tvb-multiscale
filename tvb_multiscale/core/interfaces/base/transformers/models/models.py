# -*- coding: utf-8 -*-

from enum import Enum

from tvb_multiscale.core.interfaces.base.transformers.models.base import Elementary, Scale, DotProduct, ScaleRate, \
    ScaleCurrent
from tvb_multiscale.core.interfaces.base.transformers.models.elephant import \
    ElephantRatesToSpikesTransformers, ElephantSpikesToRatesTransformers
from tvb_multiscale.core.utils.data_structures_utils import combine_enums


class BasicTransformers(Enum):
    ELEMENTARY = Elementary
    SCALE = Scale
    DOT_PRODUCT = DotProduct


RatesToSpikesTransformers = combine_enums("RatesToSpikesTransformers", ElephantRatesToSpikesTransformers)


SpikesToRatesTransformers = combine_enums("SpikesToRatesTransformers", ElephantSpikesToRatesTransformers)


class ScaleTransformers(Enum):
    RATE = ScaleRate
    CURRENT = ScaleCurrent


Transformers = combine_enums("Transformers",
                             BasicTransformers, RatesToSpikesTransformers, SpikesToRatesTransformers, ScaleTransformers)
