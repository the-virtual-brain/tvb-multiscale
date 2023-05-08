# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC

from tvb_multiscale.core.interfaces.models.red_wong_wang import \
    RedWongWangExcIOTVBSpikeNetInterfaceBuilder, \
    RedWongWangExcIOSpikeNetRemoteInterfaceBuilder, RedWongWangExcIOSpikeNetTransformerInterfaceBuilder, \
    RedWongWangExcIOSpikeNetInterfaceBuilder, RedWongWangExcIOSpikeNetProxyNodesBuilder, \
    RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder, \
    RedWongWangExcIOInhISpikeNetRemoteInterfaceBuilder, RedWongWangExcIOInhISpikeNetTransformerInterfaceBuilder, \
    RedWongWangExcIOInhISpikeNetInterfaceBuilder, RedWongWangExcIOInhISpikeNetProxyNodesBuilder

from tvb_multiscale.tvb_nest.interfaces.builders import NESTProxyNodesBuilder, NESTInterfaceBuilder, \
    NESTRemoteInterfaceBuilder, NESTTransformerInterfaceBuilder, TVBNESTInterfaceBuilder

from tvb_multiscale.tvb_nest.nest_models.builders.nest_templates import receptor_by_source_region


class RedWongWangExcIONESTProxyNodesBuilder(NESTProxyNodesBuilder, RedWongWangExcIOSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class RedWongWangExcIONESTInterfaceBuilder(RedWongWangExcIONESTProxyNodesBuilder, NESTInterfaceBuilder,
                                           RedWongWangExcIOSpikeNetInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)


class RedWongWangExcIONESTRemoteInterfaceBuilder(RedWongWangExcIONESTInterfaceBuilder, NESTRemoteInterfaceBuilder,
                                                 RedWongWangExcIOSpikeNetRemoteInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        RedWongWangExcIOSpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOSpikeNetRemoteInterfaceBuilder.default_input_config(self)


class RedWongWangExcIONESTTransformerInterfaceBuilder(RedWongWangExcIONESTInterfaceBuilder,
                                                      NESTTransformerInterfaceBuilder,
                                                      RedWongWangExcIOSpikeNetTransformerInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        RedWongWangExcIOSpikeNetTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOSpikeNetTransformerInterfaceBuilder.default_input_config(self)


class RedWongWangExcIOTVBNESTInterfaceBuilder(RedWongWangExcIONESTProxyNodesBuilder, TVBNESTInterfaceBuilder,
                                              RedWongWangExcIOTVBSpikeNetInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        RedWongWangExcIOTVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOTVBSpikeNetInterfaceBuilder.default_input_config(self)


class RedWongWangExcIOInhINESTProxyNodesBuilder(NESTProxyNodesBuilder,
                                                RedWongWangExcIOInhISpikeNetProxyNodesBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def _default_receptor_type(self, source_node, target_node):
        return receptor_by_source_region(source_node, target_node, start=1)


class RedWongWangExcIOInhINESTInterfaceBuilder(RedWongWangExcIOInhINESTProxyNodesBuilder, NESTInterfaceBuilder,
                                               RedWongWangExcIOInhISpikeNetInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    pass


class RedWongWangExcIOInhINESTRemoteInterfaceBuilder(RedWongWangExcIOInhINESTInterfaceBuilder,
                                                     NESTRemoteInterfaceBuilder,
                                                     RedWongWangExcIOInhISpikeNetRemoteInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        RedWongWangExcIOInhISpikeNetRemoteInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOInhISpikeNetRemoteInterfaceBuilder.default_input_config(self)


class RedWongWangExcIOInhINESTTransformerInterfaceBuilder(RedWongWangExcIOInhINESTInterfaceBuilder,
                                                          NESTTransformerInterfaceBuilder,
                                                          RedWongWangExcIOInhISpikeNetTransformerInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        RedWongWangExcIOInhISpikeNetTransformerInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOInhISpikeNetTransformerInterfaceBuilder.default_input_config(self)


class RedWongWangExcIOInhITVBNESTInterfaceBuilder(RedWongWangExcIOInhINESTProxyNodesBuilder, TVBNESTInterfaceBuilder,
                                                  RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder):

    def __init__(self, spiking_network=None, **kwargs):
        if spiking_network:
            self.spiking_network = spiking_network
        super().__init__(**kwargs)

    def default_output_config(self):
        RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder.default_output_config(self)

    def default_input_config(self):
        RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder.default_input_config(self)
