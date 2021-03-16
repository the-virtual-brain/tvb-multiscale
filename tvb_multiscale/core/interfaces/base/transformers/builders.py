# -*- coding: utf-8 -*-

from logging import Logger
from enum import Enum
from abc import ABCMeta, abstractmethod
from six import string_types

from tvb.basic.neotraits._core import HasTraits
from tvb.basic.neotraits._attr import Attr, Float

from tvb_multiscale.core.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.core.interfaces.base.transformers.models.base import Transformer, ScaleRate, ScaleCurrent
from tvb_multiscale.core.interfaces.base.transformers.models.elephant import \
    ElephantSpikesRate, ElephantSpikesHistogramRate, ElephantSpikesHistogram,  \
    RatesToSpikesElephantPoisson, RatesToSpikesElephantPoissonMultipleInteraction, \
    RatesToSpikesElephantPoissonSingleInteraction
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels, SpikeNetToTVBModels


class DefaultTVBtoSpikeNetModels(Enum):
    RATE = "RATE"
    SPIKES = "SPIKES"
    CURRENT = "CURRENT"


class DefaultSpikeNetToTVBModels(Enum):
    SPIKES = "SPIKES_TO_HIST_RATE"  # "SPIKES_TO_RATE", "SPIKES_TO_HIST", "SPIKES_TO_HIST_RATE"


class DefaultTVBtoSpikeNetTransformers(Enum):
    RATE = ScaleRate
    SPIKES = RatesToSpikesElephantPoisson
    SPIKES_SINGLE_INTERACTION = RatesToSpikesElephantPoissonSingleInteraction
    SPIKES_MULTIPLE_INTERACTION = RatesToSpikesElephantPoissonMultipleInteraction
    CURRENT = ScaleCurrent


class DefaultSpikeNetToTVBTransformers(Enum):
    SPIKES = ElephantSpikesHistogramRate
    SPIKES_TO_RATE = ElephantSpikesRate
    SPIKES_TO_HIST = ElephantSpikesHistogram
    SPIKES_TO_HIST_RATE = ElephantSpikesHistogramRate


class TransformerBuilder(HasTraits):
    __metaclass__ = ABCMeta

    """TransformerBuilder abstract class"""

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    logger = Attr(
        label="Logger",
        field_type=Logger,
        doc="""logging.Logger instance.""",
        required=True,
        default=initialize_logger(__name__, config=CONFIGURED)
    )

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True,
               default=0.1)

    @staticmethod
    def _configure_transformer_model(interface, interface_models, default_transformer_models, transformer_models):
        # Return a model or an Enum
        model = interface.get("transformer", interface.pop("transformer_model", None))
        if model is None:
            model = interface.get("model", interface_models[0])
            model = model.upper()
            assert model in interface_models
            model = getattr(default_transformer_models, model).value
        if isinstance(model, string_types):
            # String input:
            model = model.upper()
            model = getattr(transformer_models, model)
        elif isinstance(model, Enum):
            # Enum input:
            assert model in transformer_models
        elif not isinstance(model, Transformer):
            # assume model is a Transformer type
            model = model()
            assert isinstance(model, Transformer)
        interface["transformer"] = model

    def set_transformer_parameters(self, transformer, params):
        for p, pval in params.items():
            setattr(transformer, p, pval)

    @abstractmethod
    def configure_and_build_transformer(self, interfaces):
        pass


class TVBtoSpikeNetTransformerBuilder(TransformerBuilder):

    """TVBtoSpikeNetTransformerBuilder abstract class"""

    _tvb_to_spikeNet_models = TVBtoSpikeNetModels
    _default_tvb_to_spikeNet_models = DefaultTVBtoSpikeNetModels
    _tvb_to_spikeNet_transformer_models = DefaultTVBtoSpikeNetTransformers

    def configure_and_build_transformer(self, interfaces):
        for interface in interfaces:
            self._configure_transformer_model(interface, self._tvb_to_spikeNet_models,
                                              self._default_tvb_to_spikeNet_models,
                                              self._tvb_to_spikeNet_transformer_models)
            params = interface.pop("transformer_params", {})
            params["dt"] = params.pop("dt", self.tvb_dt)
            if isinstance(interface["transformer"], Enum):
                # It will be either an Enum...
                if interface["transformer"] == DefaultTVBtoSpikeNetTransformers.SPIKES:
                    # If the transformer is "SPIKES", but there are parameters that concern correlations...
                    correlation_factor = params.pop("correlation_factor", None)
                    scale_factor = params.pop("scale_factor", 1.0)
                    if correlation_factor:
                        interaction = params.pop("interaction", "multiple")
                        if interaction == "single":
                            interface["transformer"] = \
                                DefaultTVBtoSpikeNetTransformers.SPIKES_SINGLE_INTERACTION.value(
                                    scale_factor=scale_factor,
                                    correlation_factor=correlation_factor, **params)
                        else:
                            interface["transformer"] = \
                                DefaultTVBtoSpikeNetTransformers.SPIKES_MULTIPLE_INTERACTION.value(
                                    scale_factor=scale_factor,
                                    correlation_factor=correlation_factor, **params)
                    else:
                        interface["transformer"] = interface["transformer"].value(scale_factor=scale_factor,
                                                                                  **params)
                else:
                    interface["transformer"] = interface["transformer"].value(**params)
            else:
                # ...or a model
                self.set_transformer_parameters(interface["transformer"], params)


class SpikeNetToTVBTransformerBuilder(TransformerBuilder):

    """SpikeNetToTVBTransformerBuilder abstract class"""

    _spikeNet_to_tvb_models = SpikeNetToTVBModels
    _default_spikeNet_to_tvb_transformer_models = DefaultSpikeNetToTVBModels
    _spikeNet_to_tvb_transformer_models = DefaultSpikeNetToTVBTransformers

    def configure_and_build_transformer(self, interfaces):
        for interface in interfaces:
            self._configure_transformer_model(interface, self._spikeNet_to_tvb_models,
                                              self._default_spikeNet_to_tvb_transformer_models,
                                              self._spikeNet_to_tvb_transformer_models)
            params = interface.pop("transformer_params", {})
            params["dt"] = params.pop("dt", self.tvb_dt)
            if isinstance(interface["transformer"], Enum):
                # It will be either an Enum...
                interface["transformer"] = interface["transformer"].value(**params)
            else:
                # ...or a model
                self.set_transformer_parameters(interface["transformer"], params)
