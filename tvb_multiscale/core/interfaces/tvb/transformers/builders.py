# -*- coding: utf-8 -*-

from enum import Enum

from six import string_types
from tvb.basic.neotraits._core import HasTraits

from tvb_multiscale.core.interfaces.tvb.transformers.models import TVBtoSpikeNetRateTransformer, \
    TVBRatesToSpikesElephantPoisson, TVBtoSpikeNetCurrentTransformer, TVBSpikesToRatesElephantRate, \
    TVBRatesToSpikesElephantPoissonMultipleInteraction, TVBRatesToSpikesElephantPoissonSingleInteraction


class DefaultTVBOutputTransformers(Enum):
    RATE = TVBtoSpikeNetRateTransformer
    RATE_TO_SPIKES = TVBRatesToSpikesElephantPoisson
    CURRENT = TVBtoSpikeNetCurrentTransformer


class DefaultTVBInputTransformers(Enum):
    SPIKES_TO_RATES = TVBSpikesToRatesElephantRate
    SPIKES_FILE_TO_RATES = TVBSpikesToRatesElephantRate


class TVBOutputTransformerBuilder(HasTraits):

    _tvb_output_transformer_types = [val.value for val in DefaultTVBOutputTransformers.__members__.values()]
    _default_output_transformer_model = DefaultTVBOutputTransformers.RATE_TO_SPIKES.name
    _default_output_transformer_types = DefaultTVBOutputTransformers

    def configure(self, interfaces):
        super(TVBOutputTransformerBuilder, self).configure()
        for interface in interfaces:
            model = interface.get("transformer",
                                  interface.get("transformer_model",
                                                interface.get("model",
                                                              self._default_output_transformer_model)))
            if isinstance(model, string_types):
                model = model.upper()
                if model == DefaultTVBOutputTransformers.RATE_TO_SPIKES.name:
                    params = interface.get("transformer_params", {})
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
                        getattr(self._default_output_transformer_types, model).value(
                                                                            **interface.get("transformer_params", {}))
        return interfaces


class TVBInputTransformerBuilder(HasTraits):

    _tvb_input_transformer_types = TVBSpikesToRatesElephantRate
    _default_input_transformer_model = DefaultTVBOutputTransformers.RATE_TO_SPIKES.name
    _default_input_transformer_type = TVBSpikesToRatesElephantRate

    def configure(self, interfaces):
        super(TVBInputTransformerBuilder, self).configure()
        for interface in interfaces:
            model = interface.get("transformer",
                                  interface.get("transformer_model",
                                                interface.get("model",
                                                              self._default_input_transformer_model)))
            if isinstance(model, string_types):
                model = model.upper()
                interface["transformer"] = \
                    getattr(self._default_input_transformer_types, model).value(
                        **interface.get("transformer_params", {}))
        return interfaces