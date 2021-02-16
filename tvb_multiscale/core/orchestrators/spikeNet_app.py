# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

from tvb.basic.neotraits._attr import Attr

from tvb_multiscale.core.orchestrators.base import NonTVBApp
from tvb_multiscale.core.spiking_models.network import SpikingNetwork
from tvb_multiscale.core.spiking_models.builders.base import SpikingModelBuilder
from tvb_multiscale.core.interfaces.spikeNet.builders import SpikeNetInterfaceBuilder
from tvb_multiscale.core.interfaces.spikeNet.interfaces import SpikeNetOutputInterfaces, SpikeNetInputInterfaces


class SpikeNetApp(NonTVBApp):
    __metaclass__ = ABCMeta

    """SpikeNetApp abstract base class"""

    spikeNet_builder = Attr(
        label="Spiking Network Builder",
        field_type=SpikingModelBuilder,
        doc="""Instance of Spiking Network Builder.""",
        required=False
    )

    spiking_network = Attr(
        label="Spiking Network",
        field_type=SpikingNetwork,
        doc="""Instance of Spiking Network class.""",
        required=False
    )

    spiking_cosimulator = None

    def setup_from_orchestrator(self, orchestrator):
        super(SpikeNetApp, self).setup_from_another_app(orchestrator)

    def configure(self):
        super(SpikeNetApp, self).configure()
        if not self.spikeNet_builder:
            self.spikeNet_builder = SpikingModelBuilder(self.tvb_cosimulator_serialized, self.spiking_proxy_inds,
                                                        config=self.config, logger=self.logger)
        self.spikeNet_builder.configure()

    def build_spiking_network(self):
        self.spiking_network = self.spikeNet_builder.build_spiking_brain()

    def build(self):
        if not self.spikeNet_builder:
            self.configure()
        self.spiking_network = self.build_spiking_network()

    @abstractmethod
    def configure_simulation(self):
        pass

    def run(self, *args, **kwargs):
        self.spiking_network.Run(self.simulation_length, *args, **kwargs)


class SpikeNetSerialApp(SpikeNetApp):
    __metaclass__ = ABCMeta

    """SpikeNetSerialApp abstract base class"""

    pass


class SpikeNetParallelApp(SpikeNetApp):
    __metaclass__ = ABCMeta

    """SpikeNetParallelApp abstract base class"""

    interfaces_builder = Attr(
        label="Spiking Network interfaces builder",
        field_type=SpikeNetInterfaceBuilder,
        doc="""Instance of Spiking Network interfaces' builder class.""",
        required=False
    )

    output_interfaces = Attr(
        label="Spiking Network output interfaces",
        field_type=SpikeNetOutputInterfaces,
        doc="""Instance of output Spiking Network interfaces.""",
        required=False
    )

    input_interfaces = Attr(
        label="Spiking Network input interfaces",
        field_type=SpikeNetInputInterfaces,
        doc="""Instance of input Spiking Network interfaces.""",
        required=False
    )

    def build_interfaces(self):
        self.output_interfaces, self.input_interfaces = self.interfaces_builder.build()

    def build(self):
        super(SpikeNetParallelApp, self).build()
        self.interfaces_builder.spiking_network = self.spiking_network
        self.build_interfaces()
