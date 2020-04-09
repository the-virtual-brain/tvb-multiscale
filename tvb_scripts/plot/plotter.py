# -*- coding: utf-8 -*-

from tvb.simulator.plot.plotter import Plotter

from tvb_scripts.plot.time_series_plotter import TimeSeriesPlotter as ScriptsTimeSeriesPlotter


class ScriptsPlotter(Plotter):
    TS_PLOTTER_CLASS = ScriptsTimeSeriesPlotter
