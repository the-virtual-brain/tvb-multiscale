# -*- coding: utf-8 -*-

from tvb_scripts.plot.plotter import Plotter as BasePlotter
from tvb_nest.plot.multimeter_plotter import MultimeterPlotter
from tvb_nest.plot.spikes_plotter import SpikesPlotter


class Plotter(BasePlotter):

    def plot_spikes(self, *args, **kwargs):
        return SpikesPlotter(self.config).plot_spikes(*args, **kwargs)

    def plot_multimeter_timeseries(self, *args, **kwargs):
        return MultimeterPlotter(self.config).plot_multimeter_timeseries(*args, **kwargs)

    def plot_multimeter_raster(self, *args, **kwargs):
        return MultimeterPlotter(self.config).plot_multimeter_raster(*args, **kwargs)
