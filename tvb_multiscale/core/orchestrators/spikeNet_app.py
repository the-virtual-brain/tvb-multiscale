# -*- coding: utf-8 -*-

from abc import ABC, ABCMeta

from tvb.basic.neotraits._attr import Attr, Int

from tvb_multiscale.core.orchestrators.base import NonTVBApp
from tvb_multiscale.core.spiking_models.network import SpikingNetwork
from tvb_multiscale.core.spiking_models.builders.base import SpikingNetworkBuilder
from tvb_multiscale.core.interfaces.spikeNet.builders import SpikeNetInterfaceBuilder, SpikeNetRemoteInterfaceBuilder
from tvb_multiscale.core.interfaces.spikeNet.interfaces import \
    SpikeNetInputInterfaces, SpikeNetOutputInterfaces, \
    SpikeNetInputRemoteInterfaces, SpikeNetOutputRemoteInterfaces


class SpikeNetApp(NonTVBApp, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetApp abstract base class"""

    spikeNet_builder = Attr(
        label="Spiking Network Builder",
        field_type=SpikingNetworkBuilder,
        doc="""Instance of Spiking Network Builder.""",
        required=True
    )

    spiking_network = Attr(
        label="Spiking Network",
        field_type=SpikingNetwork,
        doc="""Instance of Spiking Network class.""",
        required=False
    )

    population_order = Int(
        label="Spiking Network populations' order",
        doc="""Size of spiking populations.""",
        required=False,
        default=None
    )

    spiking_cosimulator = None
    _attrs_to_info = ["spiking_cosimulator"]

    @property
    def _spiking_cosimulator(self):
        if self.spiking_cosimulator is None:
            self.start()
        return self.spiking_cosimulator

    @property
    def _spiking_network(self):
        if self.spiking_network is None:
            self.build_spiking_network()
        return self.spiking_network

    def configure(self):
        super(SpikeNetApp, self).configure()
        if self.spiking_network is None:
            self.spikeNet_builder.config = self.config
            self.spikeNet_builder.logger = self.logger
            if self.population_order is not None:
                self.spikeNet_builder.population_order = self.population_order
            else:
                self.population_order = self.spikeNet_builder.population_order
            self.spikeNet_builder.spiking_nodes_inds = self.spiking_proxy_inds

    def build_spiking_network(self):
        self.spikeNet_builder.tvb_serial_sim = self._serialized_tvb_cosimulator
        self.spikeNet_builder.configure()
        self.spiking_network = self.spikeNet_builder.build()

    def build(self):
        assert self._spiking_network

    def configure_simulation(self):
        self.spiking_network.configure()

    def simulate(self, simulation_length=None):
        if simulation_length is None:
            simulation_length = self.simulation_length
        return self.spiking_network.run(simulation_length)

    def reset(self):
        self.spiking_network = None


class SpikeNetSerialApp(SpikeNetApp, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetSerialApp abstract base class"""

    pass


class SpikeNetParallelApp(SpikeNetApp, ABC):
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

    _default_interface_builder = SpikeNetInterfaceBuilder

    def configure_interfaces_builder(self):
        # Get default options from the App and the TVB CoSimulator:
        self.interfaces_builder.spiking_network = self._spiking_network
        super(SpikeNetParallelApp, self).configure_interfaces_builder()

    def build(self):
        super(SpikeNetParallelApp, self).build()
        self.build_interfaces()

    def configure_simulation(self):
        self.spiking_network.configure()
        self.output_interfaces.configure()
        self.input_interfaces.configure()

    def reset(self):
        super(SpikeNetParallelApp, self).reset()
        self.input_interfaces = None
        self.output_interfaces = None
        self._interfaces_built = False


class SpikeNetRemoteParallelApp(SpikeNetParallelApp, ABC):
    __metaclass__ = ABCMeta

    """SpikeNetRemoteParallelApp abstract base class"""

    interfaces_builder = Attr(
        label="Spiking Network remote interfaces builder",
        field_type=SpikeNetRemoteInterfaceBuilder,
        doc="""Instance of Spiking Network remote interfaces' builder class.""",
        required=False
    )

    output_interfaces = Attr(
        label="Spiking Network remote output interfaces",
        field_type=SpikeNetOutputRemoteInterfaces,
        doc="""Instance of output Spiking Network remote interfaces.""",
        required=False
    )

    input_interfaces = Attr(
        label="Spiking Network remote input interfaces",
        field_type=SpikeNetInputRemoteInterfaces,
        doc="""Instance of input Spiking Network remote interfaces.""",
        required=False
    )

    _default_interface_builder = SpikeNetRemoteInterfaceBuilder

    pass
