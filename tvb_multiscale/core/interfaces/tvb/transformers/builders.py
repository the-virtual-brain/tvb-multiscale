# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from enum import Enum

from six import string_types

from tvb.basic.neotraits._core import HasTraits
from tvb.basic.neotraits._attr import List

from tvb_multiscale.core.interfaces.tvb.transformers.models import TVBOutputTransformers, TVBInputTransformers, \
    TVBtoSpikeNetRateTransformer, TVBtoSpikeNetCurrentTransformer, \
    TVBRatesToSpikesElephantPoisson, TVBSpikesToRatesElephantRate, \
    TVBRatesToSpikesElephantPoissonMultipleInteraction, TVBRatesToSpikesElephantPoissonSingleInteraction


class DefaultTVBOutputTransformers(Enum):
    RATE = TVBtoSpikeNetRateTransformer
    RATE_TO_SPIKES = TVBRatesToSpikesElephantPoisson
    CURRENT = TVBtoSpikeNetCurrentTransformer


class DefaultTVBInputTransformers(Enum):
    SPIKES_TO_RATES = TVBSpikesToRatesElephantRate
    SPIKES_FILE_TO_RATES = TVBSpikesToRatesElephantRate


TVBOutputTransformersTypes = tuple([val.value for val in TVBOutputTransformers.__members__.values()])
TVBOutputTransformersModels = tuple([val.name for val in DefaultTVBOutputTransformers.__members__.values()])

TVBInputTransformersTypes = tuple([val.value for val in TVBInputTransformers.__members__.values()])
TVBInputTransformersModels = tuple([val.value for val in DefaultTVBInputTransformers.__members__.values()])


class TVBTransformerBuilder(HasTraits):
    __metaclass__ = ABCMeta

    """TVBTransformerBuilder abstract class"""

    output_interfaces = List(of=dict, default=(), label="Output interfaces configurations",
                             doc="List of dicts of configurations for the output interfaces to be built")

    input_interfaces = List(of=dict, default=(), label="Input interfaces configurations",
                            doc="List of dicts of configurations for the input interfaces to be built")

    @property
    @abstractmethod
    def tvb_dt(self):
        pass

    @abstractmethod
    def configure_and_build_transformer(self):
        pass


class TVBOutputTransformerBuilder(TVBTransformerBuilder):
    __metaclass__ = ABCMeta

    """TVBOutputTransformerBuilder abstract class"""

    _default_output_transformer_model = DefaultTVBOutputTransformers.RATE_TO_SPIKES.name
    _default_output_transformer_types = DefaultTVBOutputTransformers

    def configure_and_build_transformer(self):
        for interface in self.output_interfaces:
            model = interface.get("transformer",
                                  interface.get("transformer_model",
                                                interface.get("model",
                                                              self._default_output_transformer_model)))
            params = interface.get("transformer_params", {})
            params = params.pop("dt", self.tvb_dt)
            if model in TVBOutputTransformersTypes:
                interface["transformer"] = model(**params)
            elif isinstance(model, TVBOutputTransformersTypes):
                for p, pval in params.items():
                    setattr(interface["transformer"], p, pval)
            elif isinstance(model, string_types):
                model = model.upper()
                assert model in TVBOutputTransformersModels
                if model == DefaultTVBOutputTransformers.RATE_TO_SPIKES.name:
                    correlation_factor = params.get("correlation_factor", None)
                    scale_factor = params.get("scale_factor", 1.0)
                    if correlation_factor:
                        interaction = params.get("interaction", "multiple")
                        if interaction == "multiple":
                            interface["transformer"] = \
                                TVBRatesToSpikesElephantPoissonMultipleInteraction(
                                    scale_factor=scale_factor,
                                    correlation_factor=correlation_factor)
                        else:
                            interface["transformer"] = \
                                TVBRatesToSpikesElephantPoissonSingleInteraction(
                                    scale_factor=scale_factor,
                                    correlation_factor=correlation_factor)
                else:
                    interface["transformer"] = \
                        getattr(self._default_output_transformer_types, model).value(**params)
            else:
                raise ValueError("Transformer configuration\n%s\nof interface\n%s\n"
                                 "is not a model name (string), model type or transformer instance "
                                 "of the available TVBOutputTransformers types!:\n%s"
                                 % (str(model), str(interface), str(TVBOutputTransformersTypes)))


class TVBInputTransformerBuilder(TVBTransformerBuilder):
    __metaclass__ = ABCMeta

    """TVBInputTransformerBuilder abstract class"""

    _default_input_transformer_model = DefaultTVBOutputTransformers.RATE_TO_SPIKES.name
    _default_input_transformer_type = TVBSpikesToRatesElephantRate

    def configure_and_build_transformer(self):
        for interface in self.input_interfaces:
            model = interface.get("transformer",
                                  interface.get("transformer_model",
                                                interface.get("model",
                                                              self._default_input_transformer_model)))
            params = interface.get("transformer_params", {})
            params = params.pop("dt", self.tvb_dt)
            if model in TVBInputTransformersTypes:
                interface["transformer"] = model(**params)
            elif isinstance(model, TVBInputTransformersTypes):
                for p, pval in params.items():
                    setattr(interface["transformer"], p, pval)
            elif isinstance(model, string_types):
                model = model.upper()
                assert model in TVBInputTransformersModels
                interface["transformer"] = \
                    getattr(self._default_input_transformer_types, model).value(**params)
            else:
                raise ValueError("Transformer configuration\n%s\nof interface\n%s\n"
                                 "is not a model name (string), model type or transformer instance "
                                 "of the available TVBInputTransformers types!:\n%s"
                                 % (str(model), str(interface), str(TVBInputTransformersTypes)))
