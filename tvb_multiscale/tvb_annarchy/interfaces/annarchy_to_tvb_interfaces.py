# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.interfaces.spikeNet_to_tvb_interface import SpikeNetToTVBinterface


class ANNarchytoTVBinterface(SpikeNetToTVBinterface):

    @property
    def anarchy_instance(self):
        return self.spiking_network.annarchy_instance

    @property
    def population_mean_spikes_number(self):
        values = []
        for i_node, (node, n_events) in enumerate(zip(self.devices(), self.number_of_events)):
            values.append(self[node].number_of_events / self[node].number_of_neurons)
        self.do_for_all_devices("reset")
        return np.array(values).flatten()

    @property
    def current_population_mean_values(self):
        values = self.do_for_all_devices("get_mean_data", return_type="values")
        self.do_for_all_devices("reset")
        return np.array(values).flatten()

    @property
    def population_mean_spikes_activity(self):
        return self.current_population_mean_values
