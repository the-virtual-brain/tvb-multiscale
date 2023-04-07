# -*- coding: utf-8 -*-

from tvb_multiscale.core.orchestrators.nrp_apps import NRPSpikeNetApp
from tvb_multiscale.tvb_nest.orchestrators import NESTApp


class NESTNRPApp(NESTApp, NRPSpikeNetApp):

    """NESTNRPApp class"""

    def configure(self):
        NRPSpikeNetApp.configure(self)
        NESTApp.configure(self)

    def configure_simulation(self):
        NRPSpikeNetApp.configure_simulation(self)
        NESTApp.configure_simulation(self)

    def reset(self):
        NRPSpikeNetApp.reset(self)
        NESTApp.reset(self)
        self.spiking_cosimulator = None

    def stop(self):
        NESTApp.stop(self)
        NRPSpikeNetApp.stop(self)
