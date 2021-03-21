# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.default import DefaultTVBSpikeNetInterfaceBuilder, \
    DefaultSpikeNetRemoteInterfaceBuilder, DefaultSpikeNetTransformerInterfaceBuilder, \
    DefaultSpikeNetOutputTransformerInterfaceBuilder, DefaultSpikeNetInputTransformerInterfaceBuilder, \
    DefaultSpikeNetInterfaceBuilder, DefaultSpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_nest.interfaces.builders import NESTProxyNodesBuilder, NESTInterfaceBuilder, \
    NESTRemoteInterfaceBuilder, TVBNESTInterfaceBuilder, \
    NESTTransformerInterfaceBuilder, NESTOutputTransformerInterfaceBuilder, NESTInputTransformerInterfaceBuilder

from tvb_multiscale.tvb_nest.nest_models.builders.nest_templates import receptor_by_source_region


class DefaultNESTProxyNodesBuilder(NESTProxyNodesBuilder, DefaultSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class DefaultNESTInterfaceBuilder(DefaultNESTProxyNodesBuilder, NESTInterfaceBuilder, DefaultSpikeNetInterfaceBuilder):
    pass


class DefaultNESTRemoteInterfaceBuilder(DefaultNESTInterfaceBuilder, NESTRemoteInterfaceBuilder,
                                        DefaultSpikeNetRemoteInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetRemoteInterfaceBuilder.default_input_config(self)


class DefaultNESNetOutputTransformerInterfaceBuilder(DefaultNESTInterfaceBuilder, NESTOutputTransformerInterfaceBuilder,
                                                     DefaultSpikeNetOutputTransformerInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetOutputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetOutputTransformerInterfaceBuilder.default_input_config(self)


class DefaultNESTInputTransformerInterfaceBuilder(DefaultNESTInterfaceBuilder, NESTInputTransformerInterfaceBuilder,
                                                  DefaultSpikeNetInputTransformerInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetInputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetInputTransformerInterfaceBuilder.default_input_config(self)


class DefaultNESTTransformerInterfaceBuilder(DefaultNESTInterfaceBuilder, NESTTransformerInterfaceBuilder,
                                             DefaultSpikeNetTransformerInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetTransformerInterfaceBuilder.default_input_config(self)


class DefaultTVBNESTInterfaceBuilder(DefaultNESTProxyNodesBuilder, TVBNESTInterfaceBuilder,
                                     DefaultTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_input_config(self)


class DefaultMultisynapseNESTProxyNodesBuilder(DefaultNESTProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    def _default_receptor_type(self, source_node, target_node):
        return receptor_by_source_region(source_node, target_node, start=3)


class DefaultMultisynapseNESTInterfaceBuilder(DefaultMultisynapseNESTProxyNodesBuilder, DefaultNESTInterfaceBuilder):

    pass


class DefaultMultisynapseNESTRemoteInterfaceBuilder(DefaultMultisynapseNESTInterfaceBuilder, NESTRemoteInterfaceBuilder,
                                                    DefaultSpikeNetRemoteInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetRemoteInterfaceBuilder.default_input_config(self)


class DefaultMultisynapseNESTOutputTransformerInterfaceBuilder(DefaultMultisynapseNESTInterfaceBuilder,
                                                               NESTOutputTransformerInterfaceBuilder,
                                                               DefaultSpikeNetOutputTransformerInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetOutputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetOutputTransformerInterfaceBuilder.default_input_config(self)


class DefaultMultisynapseNESTInputTransformerInterfaceBuilder(DefaultMultisynapseNESTInterfaceBuilder,
                                                              NESTInputTransformerInterfaceBuilder,
                                                              DefaultSpikeNetInputTransformerInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetInputTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetInputTransformerInterfaceBuilder.default_input_config(self)


class DefaultMultisynapseNESTTransformerInterfaceBuilder(DefaultMultisynapseNESTInterfaceBuilder,
                                                         NESTTransformerInterfaceBuilder,
                                                         DefaultSpikeNetTransformerInterfaceBuilder):

    def default_output_config(self):
        DefaultSpikeNetTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultSpikeNetTransformerInterfaceBuilder.default_input_config(self)


class DefaultMultisynapseTVBNESTInterfaceBuilder(DefaultMultisynapseNESTProxyNodesBuilder, TVBNESTInterfaceBuilder,
                                                 DefaultTVBSpikeNetInterfaceBuilder):

    def default_output_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        DefaultTVBSpikeNetInterfaceBuilder.default_input_config(self)
