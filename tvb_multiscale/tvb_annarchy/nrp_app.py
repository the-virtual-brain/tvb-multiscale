# -*- coding: utf-8 -*-

from tvb_multiscale.core.orchestrators.nrp_apps import NRPSpikeNetApp
from tvb_multiscale.tvb_annarchy.orchestrators import ANNarchyApp


class ANNarchyNRPApp(ANNarchyApp, NRPSpikeNetApp):

    """ANNarchyNRPApp class"""

    def configure(self):
        NRPSpikeNetApp.configure(self)
        ANNarchyApp.configure(self)

    def configure_simulation(self):
        NRPSpikeNetApp.configure_simulation(self)
        ANNarchyApp.configure_simulation(self)

    def reset(self):
        NRPSpikeNetApp.reset(self)
        ANNarchyApp.reset(self)
        self.spiking_cosimulator = None

    def stop(self):
        ANNarchyApp.stop(self)
        NRPSpikeNetApp.stop(self)
