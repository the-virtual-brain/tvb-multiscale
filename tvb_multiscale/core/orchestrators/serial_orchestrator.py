# -*- coding: utf-8 -*-

from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.orchestrators.base import Orchestrator
from tvb_multiscale.core.orchestrators.spikeNet_app import SpikeNetSerialApp
from tvb_multiscale.core.orchestrators.tvb_app import TVBSerialApp


class SerialOrchestrator(Orchestrator):

    """SerialOrchestrator base class"""

    tvb_app = Attr(
        label="TVBSerial app",
        field_type=TVBSerialApp,
        doc="""Application for running TVB serially.""",
        required=False,
        default=TVBSerialApp()
    )

    spikeNet_app = Attr(
        label="Spiking Network app",
        field_type=SpikeNetSerialApp,
        doc="""Application for running a Spiking Network (co)simulator serially.""",
        required=False,
        default=None
    )

    def __init__(self, **kwargs):
        super(SerialOrchestrator, self).__init__(**kwargs)
        assert isinstance(self.tvb_app, TVBSerialApp)
        assert isinstance(self.spikeNet_app, SpikeNetSerialApp)
        self.tvb_app.setup_from_another_app(self)
        self.spikeNet_app.setup_from_another_app(self)

    @property
    def tvb_cosimulator(self):
        return self.tvb_app._cosimulator

    @property
    def dt(self):
        return self.tvb_app._cosimulator.integrator.dt

    @property
    def spiking_network(self):
        return self.spikeNet_app._spiking_network

    def configure(self):
        super(SerialOrchestrator, self).configure()
        self.tvb_app.setup_from_another_app(self)
        self.tvb_app.configure()
        self.spikeNet_app.setup_from_another_app(self)
        self.spikeNet_app.configure()

    def start(self):
        super(SerialOrchestrator, self).start()
        self.tvb_app.start()
        self.spikeNet_app.start()

    def link_spikeNet_to_TVB_cosimulator(self):
        self.tvb_app.cosimulator.simulate_spiking_simulator = self.spiking_network.Run

    def build_cosimulators(self):
        if self.verbosity:
            self._logprint("Building cosimulators with %s..." % self.__class__.__name__)
        self.tvb_app.build_tvb_simulator()
        self.spikeNet_app.tvb_cosimulator_serialized = self.tvb_app.serialize_tvb_cosimulator()
        self.spikeNet_app.build_spiking_network()
        self.link_spikeNet_to_TVB_cosimulator()

    @property
    def populations_sizes(self):
        return self.spikeNet_app.populations_sizes

    def build_interfaces(self):
        super(SerialOrchestrator, self).build_interfaces()
        self.tvb_app.spiking_network = self.spiking_network
        self.tvb_app.build_interfaces()

    def configure_simulation(self):
        super(SerialOrchestrator, self).configure_simulation()
        self.tvb_app.configure_simulation()
        self.spikeNet_app.synchronization_time = self.tvb_cosimulator.synchronization_time
        self.spikeNet_app.configure_simulation()

    def simulate(self, simulation_length=None):
        if simulation_length is not None:
            self.tvb_app.cosimulator.simulation_length = simulation_length
        simulation_length = self.tvb_app.cosimulator.simulation_length
        super(SerialOrchestrator, self).simulate(simulation_length)
        self.configure_simulation()
        self.tvb_app.simulate()

    def plot(self, tvb_results=None, spiking_nodes_ids=None, transient=None,
             # populations=["E", "I"], populations_sizes=[],
             tvb_state_variable_type_label="State Variable", tvb_state_variables_labels=[],
             plot_per_neuron=False, plotter=None):
        super(SerialOrchestrator, self).plot()
        from examples.plot_write_results import plot_write_results
        if transient is None:
            transient = getattr(self.config, "TRANSIENT", 0.0)
        if tvb_results is None:
            tvb_results = self.tvb_app.results
            if tvb_results is None:
                return self.spikeNet_app.plot(connectivity=self.tvb_cosimulator.connectivity,
                                              transient=transient,
                                              monitor_period=self.tvb_cosimulator.monitors[0].period,
                                              plot_per_neuron=plot_per_neuron, plotter=plotter)
        if spiking_nodes_ids is None:
            spiking_nodes_ids = getattr(self.config, "SPIKING_NODES_INDS", [])
        return plot_write_results(tvb_results, self.tvb_cosimulator,
                                  self.spiking_network, spiking_nodes_ids,
                                  transient,  # populations=["E", "I"], populations_sizes=[],
                                  tvb_state_variable_type_label, tvb_state_variables_labels,
                                  plot_per_neuron, plotter, config=self.config)

    def _destroy(self):
        self.spikeNet_app = None
        self.tvb_app = None
        super(SerialOrchestrator, self)._destroy()

    def stop(self):
        self.tvb_app.stop()
        self.spikeNet_app.stop()
        super(SerialOrchestrator, self).stop()

    def clean_up(self):
        super(SerialOrchestrator, self).clean_up()
        self.tvb_app.clean_up()
        self.spikeNet_app.clean_up()

    def reset(self):
        self.tvb_app.reset()
        self.spikeNet_app.reset()
        super(SerialOrchestrator, self).reset()
