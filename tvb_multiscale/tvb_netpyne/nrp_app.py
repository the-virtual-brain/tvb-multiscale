# -*- coding: utf-8 -*-

from tvb_multiscale.core.orchestrators.nrp_apps import NRPSpikeNetApp
from tvb_multiscale.tvb_netpyne.orchestrators import NetpyneApp


class NetpyneNRPApp(NetpyneApp, NRPSpikeNetApp):

    """NetpyneNRPApp class"""

    def configure(self):
        NRPSpikeNetApp.configure(self)
        NetpyneApp.configure(self)

    def configure_simulation(self):
        NRPSpikeNetApp.configure_simulation(self)
        NetpyneApp.configure_simulation(self)

    def reset(self):
        NRPSpikeNetApp.reset(self)
        # NetpyneApp.reset(self)
        self.spiking_cosimulator = None

    def stop(self):
        NetpyneApp.stop(self)
        NRPSpikeNetApp.stop(self)
