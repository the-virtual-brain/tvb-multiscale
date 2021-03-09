# -*- coding: utf-8 -*-

from logging import Logger
from enum import Enum
from abc import ABCMeta, abstractmethod
from six import string_types

from tvb.basic.neotraits._core import HasTraits
from tvb.basic.neotraits._attr import Attr, Float, List

from tvb_multiscale.core.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.core.interfaces.tvb.transformers.models import TVBOutputTransformers, TVBInputTransformers, \
    TVBtoSpikeNetRateTransformer, TVBtoSpikeNetCurrentTransformer, \
    TVBRatesToSpikesElephantPoisson, TVBSpikesToRatesElephantRate, \
    TVBRatesToSpikesElephantPoissonMultipleInteraction, TVBRatesToSpikesElephantPoissonSingleInteraction
from tvb_multiscale.core.interfaces.spikeNet.interfaces import TVBtoSpikeNetModels, SpikeNetToTVBModels
from tvb_multiscale.core.utils.data_structures_utils import get_enum_values


class DefaultTVBtoSpikeNetModels(Enum):
    RATE = "RATE"
    SPIKES = "SPIKES"
    CURRENT = "CURRENT"


class DefaultSpikeNetToTVBModels(Enum):
    SPIKES = "SPIKES"


class DefaultTVBOutputTransformers(Enum):
    RATE = TVBtoSpikeNetRateTransformer
    SPIKES = TVBRatesToSpikesElephantPoisson
    SPIKES_SINGLE_INTERACTION = TVBRatesToSpikesElephantPoissonSingleInteraction
    SPIKES_MULTIPLE_INTERACTION = TVBRatesToSpikesElephantPoissonSingleInteraction
    CURRENT = TVBtoSpikeNetCurrentTransformer


class DefaultTVBInputTransformers(Enum):
    SPIKES = TVBSpikesToRatesElephantRate


class TVBTransformerBuilder(HasTraits):
    __metaclass__ = ABCMeta

    """TVBTransformerBuilder abstract class"""

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

    output_interfaces = List(of=dict, default=(), label="Output interfaces configurations",
                             doc="List of dicts of configurations for the output interfaces to be built")

    input_interfaces = List(of=dict, default=(), label="Input interfaces configurations",
                            doc="List of dicts of configurations for the input interfaces to be built")

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
        else:
            enum_values = tuple(get_enum_values(transformer_models))
            if model in enum_values:
                # type input:
                model = model()
            else:
                # model input
                assert isinstance(model, enum_values)
        interface["transformer"] = model

    def set_transformer_parameters(self, transformer, params):
        for p, pval in params.items():
            setattr(transformer, p, pval)

    @abstractmethod
    def configure_and_build_transformer(self):
        pass


class TVBOutputTransformerBuilder(TVBTransformerBuilder):

    """TVBOutputTransformerBuilder abstract class"""

    _tvb_to_spikeNet_models = TVBtoSpikeNetModels
    _default_tvb_to_spikeNet_models = DefaultTVBtoSpikeNetModels
    _output_transformer_models = DefaultTVBOutputTransformers

    def configure_and_build_transformer(self):
        for interface in self.output_interfaces:
            self._configure_transformer_model(interface, self._tvb_to_spikeNet_models,
                                              self._default_tvb_to_spikeNet_models, self._output_transformer_models)
            params = interface.pop("transformer_params", {})
            params["dt"] = params.pop("dt", self.tvb_dt)
            if isinstance(interface["transformer"], Enum):
                # It will be either an Enum...
                if interface["transformer"] == DefaultTVBOutputTransformers.SPIKES:
                    # If the transformer is "SPIKES", but there are parameters that concern correlations...
                    correlation_factor = params.pop("correlation_factor", None)
                    scale_factor = params.pop("scale_factor", 1.0)
                    if correlation_factor:
                        interaction = params.pop("interaction", "multiple")
                        if interaction == "multiple":
                            interface["transformer"] = \
                                TVBRatesToSpikesElephantPoissonMultipleInteraction(
                                    scale_factor=scale_factor,
                                    correlation_factor=correlation_factor, **params)
                        else:
                            interface["transformer"] = \
                                TVBRatesToSpikesElephantPoissonSingleInteraction(
                                    scale_factor=scale_factor,
                                    correlation_factor=correlation_factor, **params)
                else:
                    interface["transformer"] = interface["transformer"].value(**params)
            else:
                # ...or a model
                self.set_transformer_parameters(interface["transformer"], params)


class TVBInputTransformerBuilder(TVBTransformerBuilder):

    """TVBInputTransformerBuilder abstract class"""

    _spikeNet_to_tvb_models = SpikeNetToTVBModels
    _default_spikeNet_to_tvb_transformer_models = DefaultSpikeNetToTVBModels
    _input_transformer_models = DefaultTVBInputTransformers

    def configure_and_build_transformer(self):
        for interface in self.input_interfaces:
            self._configure_transformer_model(interface, self._spikeNet_to_tvb_models,
                                              self._default_spikeNet_to_tvb_transformer_models,
                                              self._output_transformer_models)
            params = interface.pop("transformer_params", {})
            params["dt"] = params.pop("dt", self.tvb_dt)
            if isinstance(interface["transformer"], Enum):
                # It will be either an Enum...
                interface["transformer"] = interface["transformer"].value(**params)
            else:
                # ...or a model
                self.set_transformer_parameters(interface["transformer"], params)
