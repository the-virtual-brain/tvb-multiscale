# -*- coding: utf-8 -*-

from tvb_multiscale.core.interfaces.spikeNet_to_tvb_interface import SpikeNetToTVBinterface


class NESTtoTVBinterface(SpikeNetToTVBinterface):

    @property
    def nest_instance(self):
        return self.spiking_network.annarchy_instance
