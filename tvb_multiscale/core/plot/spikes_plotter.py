# -*- coding: utf-8 -*-
import os
from six import string_types

from matplotlib import pyplot
import numpy as np
from xarray import DataArray

from tvb_multiscale.core.config import Config, CONFIGURED

from tvb.simulator.plot.base_plotter import BasePlotter
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


class SpikesPlotter(BasePlotter):

    _y_axes_labels = {}

    def __init__(self, config=CONFIGURED.figures):
        if isinstance(config, Config) or issubclass(config.__class__, Config):
            super(SpikesPlotter, self).__init__(config.figures)
        else:
            super(SpikesPlotter, self).__init__(config)

    def _compute_time_axis(self, **kwargs):
        # Time axis
        self.time_lims = [0.0, 1.0]
        if self.time is not None:
            self.time_lims = [self.time[0], self.time[-1]]
        self.time_lims = kwargs.get("time_lims", self.time_lims)
        self.time_lims[0] = kwargs.get("time_axis_min", self.time_lims[0])
        self.time_lims[1] = kwargs.get("time_axis_max", self.time_lims[1])
        time_step = (self.time_lims[-1] - self.time_lims[0]) / (kwargs.get('n_time_ticks', 11) - 1)
        self.xticks = np.arange(self.time_lims[0], self.time_lims[-1]+time_step, time_step)
        self.xticklabels = ["%0.0f" % xtick for xtick in self.xticks]

    def _get_y_ticks_labels(self, **kwargs):
        if self.max_n_neurons == 0:
            self.ylims = [0, 1]
            self.yticks = np.arange(0, 1.1, 0.1)
        else:
            self.ylims = [self.min_n_neurons, self.max_n_neurons]
            # We want n_y_ticks points of n_y_ticks-1 equal steps:
            n_y_ticks = int(kwargs.get('n_y_ticks', 11))
            neurons_step = np.int(np.ceil(np.maximum(1.0 * (self.max_n_neurons - self.min_n_neurons) /
                                                     (n_y_ticks - 1), 1.0)))
            if neurons_step == 1.0:  # This is the case of no or very few neurons...
                n_y_ticks = 11
            self.yticks = np.arange(self.ylims[0], self.ylims[0] + n_y_ticks*neurons_step, neurons_step)

    def _neurons_axis_from_indices(self, neurons, **kwargs):
        if len(neurons) > 0:
            self.max_n_neurons = np.max(neurons)
            self.min_n_neurons = np.min(neurons)
            if self.max_n_neurons == self.min_n_neurons:
                self.max_n_neurons = self.min_n_neurons + 1
                self.min_n_neurons = np.maximum(0, self.min_n_neurons - 1)
        else:
            self.max_n_neurons = 1.0
            self.min_n_neurons = 0.0
        self._get_y_ticks_labels(**kwargs)

    def _get_time_series_ytick_labels(self):
        n_ticks = len(self.yticks)
        n_steps = n_ticks - 1
        if self.max_time_series <= 10.0:
            str_format = "%g"
            time_series_step = self.max_time_series / n_steps
        else:
            str_format = "%d"
            time_series_step = np.ceil(self.max_time_series / n_steps)
        yticklabels = np.arange(0.0, n_ticks * time_series_step, time_series_step)
        self._time_series_ytick_labels = [str_format % yticklabel for yticklabel in yticklabels]

    def _get_from_time_series(self, time_series, **kwargs):
        # If we plot time_series, we need to....
        self.time_series = time_series
        self.max_time_series = 0.0
        self.time_lims = None
        self.xticks = None
        self.xticklabels = None
        self.get_time_series_fun = None
        self.plot_time_series = False
        self._time_series_ytick_labels = None
        self.time = None
        if self.time_series is not None:
            self.plot_time_series = True
            # ...compute the maximum of time series to adjust the y axis accordingly
            if isinstance(time_series, DataArray):
                if self.time_series.size > 0:  # Assuming DataArray
                    self.max_time_series = np.nanmax(self.time_series).item()
                    self.time = time_series.get_index(self.time_series.dims[0])
                    self.get_time_series_fun = \
                        lambda pop_lbl, reg_lbl: self.time_series.loc[:, pop_lbl, reg_lbl].values.squeeze()
                else:
                    self.plot_time_series = False
            else:  # Assuming TimeSeries
                if self.time_series.size > 0:
                    self.max_time_series = np.nanmax(self.time_series)
                    self.time = self.time_series.time
                    self.get_time_series_fun = \
                        lambda pop_lbl, reg_lbl: self.time_series[:, pop_lbl, reg_lbl].data.squeeze()
                else:
                    self.plot_time_series = False
            if self.max_time_series == 0.0:
                self.max_time_series = 1.0  # if no spikes at all...
            if self.plot_time_series:
                # ..and set the time axis accordingly
                # Time axis
                self._compute_time_axis(**kwargs)

    def _default_title(self, title, time_series, figure_name=None, **kwargs):
        if title is None:
            if figure_name is not None:
                title = figure_name
            else:
                title = "Population spikes"
                if time_series is not None:
                    time_series_label = kwargs.pop("time_series_label", time_series.name)
                    if len(time_series_label) == 0:
                        time_series_label = "time series"
                    title = title + " and %s" % time_series_label
        return str(title)

    def _get_figname_figsize(self, title="Population spikes and spike time series",
                             figure_name=None, figsize=None):
        if figure_name is None:
            figure_name = title
        if figsize is None:
            figsize = self.config.LARGE_SIZE
        return figure_name, figsize

    def _format_time_axis(self, axes, i_region, **kwargs):
        show_time_axis = kwargs.get("show_time_axis", None)
        if self.time_lims is not None:
            axes.set_xlim(self.time_lims)
        if self.xticks is not None:
            axes.set_xticks(self.xticks)
        if (i_region == self.n_regions - 1 or show_time_axis is True) and show_time_axis is not False:
            if self.xticklabels is not None:
                axes.set_xticklabels(self.xticklabels)
            show_time_xlabel = kwargs.get("show_time_xlabel", None)
            if (i_region == self.n_regions - 1 or show_time_xlabel is True) and show_time_xlabel is not False:
                axes.set_xlabel("Time (ms)", fontsize=kwargs.get("xlabel_fontsize", self.config.FONTSIZE))
        elif self.time_lims is not None:
            axes.set_xticklabels([])
        return axes

    def _format_axes(self, axes, i_pop, i_region, pop_label, reg_label, **kwargs):
        title = str(pop_label)
        if self.mean_results is not None:
            if self.mean_results.name.lower().find("rate") > -1:
                mean_results_units = "spikes/s"
            else:
                mean_results_units = ""
            mean_results_units = kwargs.get("mean_results_units", mean_results_units)
            title += ": %.1f %s" % (self.mean_results.loc[pop_label, reg_label], mean_results_units)
        axes.set_title(title, fontsize=kwargs.get("title_fontsize", self.config.FONTSIZE))
        axes = self._format_time_axis(axes, i_region, **kwargs)
        axes.set_ylim(self.ylims)
        axes.set_yticks(self.yticks)
        axes.set_yticklabels([])
        if self._time_series_ytick_labels is not None:
            if i_pop == 0:
                axes.set_yticklabels(self._time_series_ytick_labels)
        if i_region not in self._y_axes_labels.keys():
            self._y_axes_labels[i_region] = []
        if reg_label not in self._y_axes_labels[i_region]:
            self._y_axes_labels[i_region].append(reg_label)
            axes.set_ylabel("%s neurons" % reg_label, fontsize=kwargs.get("ylabel_fontsize", self.config.FONTSIZE))
        elif i_pop == 0:
            axes.set_ylabel("%s neurons" % reg_label, fontsize=kwargs.get("ylabel_fontsize", self.config.FONTSIZE))
        return axes

    def _scale_time_series_to_axis(self, time_series):
        return time_series / self.max_time_series * self.max_n_neurons + self.min_n_neurons

    def _prepare_time_axes_adjustment(self):
        self._any_spikes = False
        if self.time is None:
            self._adjust_time_axes_flag = True
            finfo = np.finfo(dtype="f")
            self.time = [finfo.max, finfo.min]
        else:
            self._adjust_time_axes_flag = False

    def _adjust_time_axes(self, axes, **kwargs):
        if not self._any_spikes:
            self.time = [0.0, 1.0]
        self._compute_time_axis(**kwargs)
        for i_pop, pop_axes in enumerate(axes):
            for i_region, reg_ax in enumerate(pop_axes):
                axes[i_pop][i_region] = self._format_time_axis(reg_ax, i_region, **kwargs)
        return axes

    def _plot(self, axes, i_pop, i_region, pop_label, reg_label,
              spikes_times, spikes_senders, **kwargs):
        axes[i_pop].append(pyplot.subplot(self.n_regions, self.n_pops, i_region * self.n_pops + i_pop + 1))

        if kwargs.get("plot_spikes", True):
            axes[i_pop][i_region].plot(spikes_times, spikes_senders,
                                       linestyle="None",
                                       marker=kwargs.get("spikes_marker", "o"),
                                       markerfacecolor=kwargs.get("spikes_color", "k"),
                                       markeredgecolor=kwargs.get("spikes_color", "k"),
                                       markersize=kwargs.get("spikes_markersize", 2.0),
                                       alpha=kwargs.get("spikes_alpha", 1.0))

        if self.time_series is not None:
            # Adjust time_series values to neurons axis range
            time_series = self._scale_time_series_to_axis(self.get_time_series_fun(pop_label, reg_label))
            self._get_time_series_ytick_labels()
            time_series_linestyle = kwargs.get("time_series_linestyle", None)
            time_series_marker = kwargs.get("time_series_marker", None)
            if time_series_linestyle is None and time_series_marker is None:
                time_series_linestyle = "-"
            if time_series_linestyle is not None:
                linewidth = kwargs.get("time_series_linewidth", 5.0)
            else:
                linewidth = 0.0
            if time_series_marker is not None:
                markerfacecolor = kwargs.get("time_series_color", "k")
                markeredgecolor = kwargs.get("time_series_color", "k")
                markersize = kwargs.get("time_series_markersize", 5.0)
            else:
                markerfacecolor = None
                markeredgecolor = None
                markersize = 0.0
            axes[i_pop][i_region].plot(self.time, time_series,
                                       linestyle=time_series_linestyle, linewidth=linewidth,
                                       marker=time_series_marker, markerfacecolor=markerfacecolor,
                                       markeredgecolor=markeredgecolor, markersize=markersize,
                                       color=kwargs.get("time_series_color", "k"),
                                       alpha=kwargs.get("time_series_alpha", 0.5))
        elif self._adjust_time_axes_flag:
            if len(spikes_times):
                self._any_spikes = True
                self.time[0] = np.minimum(self.time[0], np.min(spikes_times))
                self.time[-1] = np.maximum(self.time[-1], np.max(spikes_times))

        if self.stimulus is not None:
            # Adjust time_series values to neurons axis range
            for stim in self.stimulus:
                axes[i_pop][i_region].plot([stim]*2, [0, self.max_n_neurons + self.min_n_neurons],
                                           linestyle=kwargs.get("stimulus_linestyle", "-"),
                                           color=kwargs.get("stimulus_color", "r"),
                                           linewidth=kwargs.get("stimulus_linewidth", 1.0),
                                           alpha=kwargs.get("stimulus_alpha", 1.0))

        axes[i_pop][i_region] = self._format_axes(axes[i_pop][i_region], i_pop, i_region, pop_label, reg_label,
                                                  **kwargs)

        return axes

    def _check_show(self):
        pyplot.tight_layout()
        if self.config.SHOW_FLAG:
            # mp.use('TkAgg')
            pyplot.ion()
            pyplot.show()
        else:
            # mp.use('Agg')
            pyplot.ioff()
            pyplot.close()

    def _save_figure(self, fig=pyplot.gcf(), figure_name=None):
        fig.tight_layout()
        if self.config.SAVE_FLAG:
            figure_name = self._figure_filename(fig, figure_name)
            figure_name = figure_name[:np.min([100, len(figure_name)])] + '.' + self.config.FIG_FORMAT
            figure_dir = self.config.FOLDER_FIGURES
            if not (os.path.isdir(figure_dir)):
                os.mkdir(figure_dir)
            pyplot.savefig(os.path.join(figure_dir, figure_name))

    def _return_figure(self, axes, figure_name):
        fig = pyplot.gcf()
        # fig.tight_layout()
        self._save_figure(fig, figure_name)
        self._check_show()
        return fig, axes

    def plot_spikes(self, pop_spikes, time_series=None, mean_results=None,
                    title=None, figure_name=None, figsize=None, **kwargs):
        if isinstance(pop_spikes, DataArray):
            self.get_spikes_fun = lambda spikes, i_region: spikes[:, :, i_region].values.squeeze()
            self.get_time = lambda spikes, time: spikes.get_index(spikes.dims[0])
        else:
            self.get_spikes_fun = lambda spikes, i_region: spikes[:, :, i_region].squeezed
            self.get_time = lambda spikes, time: time
        self._get_from_time_series(time_series, **kwargs)
        self.mean_results = mean_results
        pop_spikes = ensure_list(pop_spikes)
        self.n_pops = len(pop_spikes)
        self.n_regions = np.max([spikes.number_of_labels for spikes in pop_spikes])
        self._prepare_time_axes_adjustment()
        self.stimulus = kwargs.pop("stimulus", None)

        # Create figure
        title = self._default_title(title, time_series, figure_name=None, **kwargs)
        axes = []
        figure_name, figsize = self._get_figname_figsize(title, figure_name, figsize)
        pyplot.figure(figure_name, figsize=figsize)
        for i_pop, spikes in enumerate(pop_spikes):
            pop_label = spikes.labels_dimensions[spikes.labels_ordering[1]][0]
            spike_time = self.get_time(spikes, self.time)
            axes.append([])
            for i_region in range(spikes.number_of_labels):
                reg_label = spikes.labels_dimensions[spikes.labels_ordering[2]][i_region]
                # Get spikes
                indices = np.argwhere(self.get_spikes_fun(spikes, i_region) > 0.0)
                this_time = spike_time[indices[:, 0]]
                spike_senders_indices = spike_time[indices[:, 1]]
                self._neurons_axis_from_indices(spike_senders_indices, **kwargs)
                axes = self._plot(axes, i_pop, i_region, pop_label, reg_label,
                                  this_time, spike_senders_indices, **kwargs)
        if self._adjust_time_axes_flag:
            axes = self._adjust_time_axes(axes, **kwargs)

        return self._return_figure(axes, figure_name)

    def plot_spike_detectors(self, spike_detectors, time_series=None, mean_results=None,
                             title=None, figure_name=None, figsize=None, **kwargs):
        # This method will plot a spike raster and, optionally,
        # it will superimpose the time series as a faded line.
        self._get_from_time_series(time_series, **kwargs)
        self.mean_results = mean_results
        self.n_pops = len(spike_detectors)
        self.n_regions = np.max([len(spike_detector) for spike_detector in spike_detectors])
        self._prepare_time_axes_adjustment()
        self.stimulus = kwargs.pop("stimulus", None)

        # Create figure
        title = self._default_title(title, time_series, figure_name=None, **kwargs)
        axes = []
        figure_name, figsize = self._get_figname_figsize(title, figure_name, figsize)
        pyplot.figure(figure_name, figsize=figsize)
        # Plot by arranging populations in columns and regions in rows
        for i_pop, (pop_label, pop_spike_detector) in enumerate(spike_detectors.iteritems()):
            axes.append([])
            for i_region, (reg_label, region_spike_detector) in enumerate(pop_spike_detector.iteritems()):
                # Get spikes
                senders = region_spike_detector.senders
                neurons = region_spike_detector.neurons
                self._neurons_axis_from_indices(neurons, **kwargs)
                axes = self.c(axes, i_pop, i_region, pop_label, reg_label,
                                  region_spike_detector.spikes_times, senders, **kwargs)
        if self._adjust_time_axes_flag:
            axes = self._adjust_time_axes(axes, **kwargs)

        return self._return_figure(axes, figure_name)

    def plot_spike_events(self, spikes_events, time_series=None, mean_results=None,
                          title=None, figure_name=None, figsize=None, **kwargs):
        # This method will plot a spike raster and, optionally,
        # it will superimpose the time series as a faded line.
        self._get_from_time_series(time_series, **kwargs)
        self.mean_results = mean_results
        self.n_pops = len(spikes_events)
        self.n_regions = np.max([len(pop_spikes_events) for pop_spikes_events in spikes_events])
        self._prepare_time_axes_adjustment()
        self.stimulus = kwargs.pop("stimulus", None)

        # Create figure
        title = self._default_title(title, time_series, figure_name=None, **kwargs)
        axes = []
        figure_name, figsize = self._get_figname_figsize(title, figure_name, figsize)
        pyplot.figure(figure_name, figsize=figsize)
        # Plot by arranging populations in columns and regions in rows
        for i_pop, (pop_label, pop_spikes_events) in enumerate(spikes_events.iteritems()):
            axes.append([])
            for i_region, (reg_label, region_spikes_events) in enumerate(pop_spikes_events.iteritems()):
                # Define spike senders and time_series' axis
                neurons = region_spikes_events["senders"]
                self._neurons_axis_from_indices(neurons, **kwargs)
                axes = self._plot(axes, i_pop, i_region, pop_label, reg_label,
                                  region_spikes_events["times"], neurons, **kwargs)
                axes[i_pop][i_region] = self._format_axes(axes[i_pop][i_region], i_pop, i_region, pop_label, reg_label,
                                                          **kwargs)
        if self._adjust_time_axes_flag:
            axes = self._adjust_time_axes(axes, **kwargs)

        return self._return_figure(axes, figure_name)
