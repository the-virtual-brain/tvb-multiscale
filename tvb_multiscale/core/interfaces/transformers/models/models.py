# -*- coding: utf-8 -*-

from enum import Enum

from tvb_multiscale.core.interfaces.transformers.models.base import \
    Elementary, Linear, LinearRate, LinearCurrent, LinearPotential
from tvb_multiscale.core.interfaces.transformers.models.integration import IntegrationTransformers
from tvb_multiscale.core.interfaces.transformers.models.elephant import \
    ElephantRatesToSpikesTransformers, ElephantSpikesToRatesTransformers
from tvb_multiscale.core.utils.data_structures_utils import combine_enums


class BasicTransformers(Enum):
    ELEMENTARY = Elementary
    LINEAR = Linear


RatesToSpikesTransformers = combine_enums("RatesToSpikesTransformers", ElephantRatesToSpikesTransformers)


SpikesToRatesTransformers = combine_enums("SpikesToRatesTransformers", ElephantSpikesToRatesTransformers)


class LinearTransformers(Enum):
    LINEAR_RATE = LinearRate
    LINEAR_CURRENT = LinearCurrent
    LINEAR_POTENTIAL = LinearPotential


Transformers = combine_enums("Transformers",
                             BasicTransformers, RatesToSpikesTransformers, SpikesToRatesTransformers,
                             LinearTransformers, IntegrationTransformers)
