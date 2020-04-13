# -*- coding: utf-8 -*-
from tvb_multiscale.config import Config, CONFIGURED
from tvb_multiscale.plot.multimeter_plotter import MultimeterPlotter
from tvb_multiscale.plot.spikes_plotter import SpikesPlotter

from tvb.simulator.plot.plotter import Plotter as BasePlotter


class Plotter(BasePlotter):

    def __init__(self, config=CONFIGURED.figures):
        if isinstance(config, Config) or issubclass(config.__class__, Config):
            super(Plotter, self).__init__(config.figures)
        else:
            super(Plotter, self).__init__(config)

    def plot_spikes(self, *args, **kwargs):
        return SpikesPlotter(self.config).plot_spikes(*args, **kwargs)

    def plot_spike_events(self, *args, **kwargs):
        return SpikesPlotter(self.config).plot_spike_events(*args, **kwargs)

    def plot_spike_detectors(self, *args, **kwargs):
        return SpikesPlotter(self.config).plot_spike_detectors(*args, **kwargs)

    def plot_multimeter_timeseries(self, *args, **kwargs):
        return MultimeterPlotter(self.config).plot_multimeter_timeseries(*args, **kwargs)

    def plot_multimeter_raster(self, *args, **kwargs):
        return MultimeterPlotter(self.config).plot_multimeter_raster(*args, **kwargs)
