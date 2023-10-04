# -*- coding: utf-8 -*-

from types import FunctionType

from tvb.basic.neotraits.api import Attr, Float, Int

from tvb_multiscale.core.orchestrators.base import NonTVBApp
from tvb_multiscale.core.spiking_models.network import SpikingNetwork
from tvb_multiscale.core.spiking_models.builders.base import SpikingNetworkBuilder
from tvb_multiscale.core.interfaces.spikeNet.builders import SpikeNetInterfaceBuilder, SpikeNetRemoteInterfaceBuilder
from tvb_multiscale.core.interfaces.spikeNet.interfaces import SpikeNetInputInterfaces, SpikeNetOutputInterfaces


class SpikeNetApp(NonTVBApp):

    """SpikeNetApp base class"""

    spikeNet_builder = Attr(
        label="Spiking Network Builder",
        field_type=SpikingNetworkBuilder,
        doc="""Instance of Spiking Network Builder.""",
        required=False
    )

    spikeNet_builder_function = Attr(
        label="Spiking Network Builder function",
        field_type=FunctionType,
        doc="""Instance of Spiking Network builder function.""",
        required=False
    )

    spiking_network = Attr(
        label="Spiking Network",
        field_type=SpikingNetwork,
        doc="""Instance of Spiking Network class.""",
        required=False
    )

    spiking_cosimulator = None

    _spikeNet_builder_type = SpikingNetworkBuilder
    _spikeNet_type = SpikingNetwork

    _attrs_to_info = ["spiking_cosimulator"]

    def __init__(self, **kwargs):
        self.spiking_cosimulator = None
        self.spiking_network = None
        self.spikeNet_builder = None
        self.spikeNet_builder_function = None
        super(SpikeNetApp, self).__init__(**kwargs)

    def start(self):
        super(SpikeNetApp, self).start()

    @property
    def _spiking_cosimulator(self):
        try:
            assert self.spiking_cosimulator is not None
        except Exception as e:
            self.logger.warn(e)
            self._logprint("Starting spiking simulator module!")
            self.start()
        return self.spiking_cosimulator

    @property
    def _spikeNet_builder(self):
        if isinstance(self.spikeNet_builder, self._spikeNet_builder_type):
            return self.spikeNet_builder
        elif hasattr(self.spikeNet_builder_function, "__call__"):
            return self.spikeNet_builder_function
        else:
            raise Exception("Neither a spiking network, "
                            "nor a spiking network builder has been provided as input to spikeNetApp!")

    @property
    def _spiking_network(self):
        try:
            assert isinstance(self.spiking_network, self._spikeNet_type)
        except:
            self.build_spiking_network()
        return self.spiking_network

    def configure(self):
        super(SpikeNetApp, self).configure()
        if self.spiking_network is None:
            if isinstance(self._spikeNet_builder, self._spikeNet_builder_type):
                self.spikeNet_builder.config = self.config
                self.spikeNet_builder.logger = self.logger
                self.spikeNet_builder.spiking_simulator = self._spiking_cosimulator

    def build_spiking_network(self):
        if isinstance(self._spikeNet_builder, self._spikeNet_builder_type):
            self._logprint("Building Spiking Network with builder %s of App %s..."
                           % (self.spikeNet_builder.__class__.__name__, self.__class__.__name__))
            self.spikeNet_builder.tvb_serial_sim = self._serialized_tvb_cosimulator
            self.spikeNet_builder.configure()
            self.spiking_network = self.spikeNet_builder.build()
        else:
            self._logprint("Building Spiking Network with builder script %s of App %s..."
                           % (self.spikeNet_builder_function.__name__, self.__class__.__name__))
            self.spiking_network = self.spikeNet_builder_function(self.config)
        if self.verbosity:
            self.spiking_network.print_summary_info_details(recursive=1+self.verbosity, connectivity=1-self.verbosity)

    def build(self):
        self._logprint("Building with %s %s..." % (self._app_or_orchestrator, self.__class__.__name__))
        assert self._spiking_network

    def configure_simulation(self):
        super(SpikeNetApp, self).configure_simulation()
        self.spiking_network.configure()

    def simulate(self, simulation_length):
        super(SpikeNetApp, self).simulate(simulation_length)
        return self.spiking_network.Run(simulation_length)

    def plot(self, connectivity=None, time=None, transient=None, monitor_period=None,
             plot_per_neuron=False, plotter=None, writer=None):
        super(SpikeNetApp, self).plot()
        from examples.plot_write_results import plot_write_spiking_network_results
        if connectivity is None:
            connectivity = self.tvb_connectivity
        if monitor_period is None:
            monitor_period = self.tvb_monitor_period
        if transient is None:
            transient = getattr(self.config, "TRANSIENT", 0.0)
        return plot_write_spiking_network_results(self._spiking_network, connectivity,
                                                  time, transient, monitor_period,
                                                  plot_per_neuron, plotter, writer, self.config)

    def _destroy(self):
        self.spikeNet_builder = None
        self.spiking_network = None
        del self.spiking_cosimulator
        self.spiking_cosimulator = None
        super(SpikeNetApp, self)._destroy()


class SpikeNetSerialApp(SpikeNetApp):

    """SpikeNetSerialApp base class"""

    pass


class SpikeNetParallelApp(SpikeNetApp):

    """SpikeNetParallelApp base class"""

    synchronization_time = Float(
        label="Synchronization time (ms)",
        default=0.0,
        required=True,
        doc="""Synchronization time (default in milliseconds).""")

    interfaces_builder = Attr(
        label="Spiking Network interfaces builder",
        field_type=SpikeNetInterfaceBuilder,
        doc="""Instance of Spiking Network interfaces' builder class.""",
        required=False
    )

    _default_interface_builder_type = SpikeNetInterfaceBuilder

    def __init__(self, **kwargs):
        self.synchronization_time = 0.0
        super(SpikeNetParallelApp, self).__init__(**kwargs)

    def configure_interfaces_builder(self):
        # Get default options from the App and the TVB CoSimulator:
        self._interfaces_builder.spiking_network = self._spiking_network
        super(SpikeNetParallelApp, self).configure_interfaces_builder()

    def build_interfaces(self):
        if self._interfaces_built is False:
            super(SpikeNetParallelApp, self).build_interfaces()
            self.spiking_network = self.interfaces_builder.build()
            self._interfaces_build = True
            if self.verbosity:
                self._logprint(
                    self.spiking_network.input_interfaces.summary_info_to_string(recursive=self.verbosity+1))
                self._logprint(
                    self.spiking_network.output_interfaces.summary_info_to_string(recursive=self.verbosity+1))

    def build(self):
        super(SpikeNetParallelApp, self).build()
        self.build_interfaces()

    def configure_simulation(self):
        super(SpikeNetParallelApp, self).configure_simulation()
        self.spiking_network.input_interfaces.configure()
        self.spiking_network.output_interfaces.configure()
        try:
            assert self.synchronization_time == self.spiking_network.input_interfaces.synchronization_time
        except:
            self.synchronization_time = self.spiking_network.input_interfaces.synchronization_time
            self._logprint("Setting %s synchronization_time from interfaces = %g" %
                           (self.__class__.__name__, self.synchronization_time))

    def run_for_synchronization_time(self, cosim_updates, cosimulation=True):
        if cosimulation: self.spiking_network.input_interfaces(cosim_updates)
        self.spiking_network.Run(self.synchronization_time)
        if cosimulation:
            return self.spiking_network.output_interfaces()

    def reset(self):
        super(SpikeNetParallelApp, self).reset()
        self._interfaces_built = False


class SpikeNetRemoteParallelApp(SpikeNetParallelApp):

    """SpikeNetRemoteParallelApp base class"""

    interfaces_builder = Attr(
        label="Spiking Network interfaces builder",
        field_type=SpikeNetRemoteInterfaceBuilder,
        doc="""Instance of Spiking Network interfaces' builder class.""",
        required=False
    )

    _default_interface_builder_type = SpikeNetRemoteInterfaceBuilder

    def run_for_synchronization_time(self, cosimulation=True):
        if cosimulation: self.spiking_network.input_interfaces()
        self.spiking_network.Run(self.synchronization_time)
        if cosimulation:
            return self.spiking_network.output_interfaces()
