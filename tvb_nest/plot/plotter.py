# -*- coding: utf-8 -*-

from tvb_scripts.plot.plotter import Plotter as BasePlotter
from tvb_nest.plot.spikes_plotter import SpikesPlotter


class Plotter(BasePlotter):

    def plot_spikes(self, *args, **kwargs):
        return SpikesPlotter(self.config).plot_spikes(*args, **kwargs)
