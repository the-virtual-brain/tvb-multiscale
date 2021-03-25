# -*- coding: utf-8 -*-
import os

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
        time_step = (self.time_lims[-1] - self.time_lims[0]) / kwargs.get('n_time_ticks', 10)
        self.xticks = np.arange(self.time_lims[0], self.time_lims[-1]+time_step, time_step)
        self.xticklabels = ["%0.0f" % xtick for xtick in self.xticks]

    def _get_y_ticks_labels(self):
        if self.max_n_neurons == 0:
            self.ylims = [0, 1]
            self.yticks = np.arange(0, 1.1, 0.1)
        else:
            self.ylims = [self.min_n_neurons, self.max_n_neurons]
            # We want 11 points of 10 equal steps:
            neurons_step = np.int(np.ceil(np.maximum(1.0 * (self.max_n_neurons - self.min_n_neurons) / 10, 1.0)))
            self.yticks = np.arange(self.ylims[0], self.ylims[0] + 11*neurons_step, neurons_step)

    def _neurons_axis_from_indices(self, neurons):
        if len(neurons) > 0:
            self.max_n_neurons = np.max(neurons)
            self.min_n_neurons = np.min(neurons)
            if self.max_n_neurons == self.min_n_neurons:
                self.max_n_neurons = self.min_n_neurons + 1
                self.min_n_neurons = np.maximum(0, self.min_n_neurons - 1)
        else:
            self.max_n_neurons = 1.0
            self.min_n_neurons = 0.0
        self._get_y_ticks_labels()

    def _get_rate_ytick_labels(self):
        n_ticks = len(self.yticks)
        n_steps = n_ticks - 1
        if self.max_rate <= 10.0:
            str_format = "%g"
            rate_step = self.max_rate / n_steps
        else:
            str_format = "%d"
            rate_step = np.ceil(self.max_rate / n_steps)
        yticklabels = np.arange(0.0, n_ticks * rate_step, rate_step)
        self._rate_ytick_labels = [str_format % yticklabel for yticklabel in yticklabels]

    def _get_from_rates(self, rates, **kwargs):
        # If we plot rates, we need to....
        self.rates = rates
        self.max_rate = 0.0
        self.time_lims = None
        self.xticks = None
        self.xticklabels = None
        self.get_rate_fun = None
        self.plot_rates = False
        self._rate_ytick_labels = None
        self.time = None
        if self.rates is not None:
            self.plot_rates = True
            # ...compute the maximum rate to adjust the y axis accordingly
            if isinstance(rates, DataArray):  # In case we call this within the NEST interface context using pandas
                if self.rates.size > 0:
                    self.max_rate = np.nanmax(self.rates).item()
                    self.time = rates.get_index(self.rates.dims[0])
                    self.get_rate_fun = lambda pop_lbl, reg_lbl: self.rates.loc[:, pop_lbl, reg_lbl].values.squeeze()
                else:
                    self.plot_rates = False
            else:  # Assuming TimeSeries
                if self.rates.size > 0:
                    self.max_rate = np.nanmax(self.rates)
                    self.time = self.rates.time
                    self.get_rate_fun = lambda pop_lbl, reg_lbl: self.rates[:, pop_lbl, reg_lbl].data.squeeze()
                else:
                    self.plot_rates = False
            if self.max_rate == 0.0:
                self.max_rate = 1.0  # if no spikes at all...
            if self.plot_rates:
                # ..and set the time axis accordingly
                # Time axis
                self._compute_time_axis(**kwargs)

    def _default_title(self, title, rates_flag, figure_name=None):
        if title is None:
            if figure_name is not None:
                title = figure_name
            else:
                title = "Population spikes"
                if rates_flag:
                    title = title + " and spike rates"
        return str(title)

    def _get_figname_figsize(self, title="Population spikes and spike rate",
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
                axes.set_xlabel("Time (ms)")
        elif self.time_lims is not None:
            axes.set_xticklabels([])
        return axes

    def _format_axes(self, axes, i_pop, i_region, pop_label, reg_label, mean_rates=None, **kwargs):
        title = str(pop_label)
        if mean_rates is not None:
            title += ": %.1fHz" % mean_rates.loc[pop_label, reg_label]
        axes.set_title(title)
        axes = self._format_time_axis(axes, i_region, **kwargs)
        axes.set_ylim(self.ylims)
        axes.set_yticks(self.yticks)
        axes.set_yticklabels([])
        if self._rate_ytick_labels is not None:
            if i_pop == 0:
                axes.set_yticklabels(self._rate_ytick_labels)
        if i_region not in self._y_axes_labels.keys():
            self._y_axes_labels[i_region] = []
        if reg_label not in self._y_axes_labels[i_region]:
            self._y_axes_labels[i_region].append(reg_label)
            axes.set_ylabel("%s neurons" % reg_label)
        elif i_pop == 0:
            axes.set_ylabel("%s neurons" % reg_label)
        return axes

    def _scale_rate_to_axis(self, rate):
        return rate / self.max_rate * self.max_n_neurons + self.min_n_neurons

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
              spikes_times, spikes_senders, mean_rates=None, **kwargs):
        axes[i_pop].append(pyplot.subplot(self.n_regions, self.n_pops, i_region * self.n_pops + i_pop + 1))

        axes[i_pop][i_region].plot(spikes_times, spikes_senders,
                                   linestyle="None",
                                   marker=kwargs.get("spikes_marker", "o"),
                                   markerfacecolor=kwargs.get("spikes_color", "k"),
                                   markeredgecolor=kwargs.get("spikes_color", "k"),
                                   markersize=kwargs.get("spikes_markersize", 2.0),
                                   alpha=kwargs.get("spikes_alpha", 1.0))

        if self.rates is not None:
            # Adjust rates values to neurons axis range
            rate_vals = self._scale_rate_to_axis(self.get_rate_fun(pop_label, reg_label))
            print(np.unique(rate_vals))
            self._get_rate_ytick_labels()
            axes[i_pop][i_region].plot(self.time, rate_vals,
                                       linestyle=kwargs.get("rate_linestyle", "-"),
                                       color=kwargs.get("rate_color", "k"),
                                       linewidth=kwargs.get("rate_linewidth", 5.0),
                                       alpha=kwargs.get("rate_alpha", 0.5))
        elif self._adjust_time_axes_flag:
            if len(spikes_times):
                self._any_spikes = True
                self.time[0] = np.minimum(self.time[0], np.min(spikes_times))
                self.time[-1] = np.maximum(self.time[-1], np.max(spikes_times))

        axes[i_pop][i_region] = self._format_axes(axes[i_pop][i_region], i_pop, i_region, pop_label, reg_label,
                                                  mean_rates=mean_rates, **kwargs)

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

    def plot_spikes(self, pop_spikes, rates=None, mean_rates=None,
                    title=None, figure_name=None, figsize=None, **kwargs):
        if isinstance(pop_spikes, DataArray):
            self.get_spikes_fun = lambda spikes, i_region: spikes[:, :, i_region].values.squeeze()
            self.get_time = lambda spikes, time: spikes.get_index(spikes.dims[0])
        else:
            self.get_spikes_fun = lambda spikes, i_region: spikes[:, :, i_region].squeezed
            self.get_time = lambda spikes, time: time
        self._get_from_rates(rates)
        pop_spikes = ensure_list(pop_spikes)
        self.n_pops = len(pop_spikes)
        self.n_regions = np.max([spikes.number_of_labels for spikes in pop_spikes])
        self._prepare_time_axes_adjustment()

        # Create figure
        title = self._default_title(title, rates is not None, figure_name=None)
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
                self._neurons_axis_from_indices(spike_senders_indices)
                axes = self._plot(axes, i_pop, i_region, pop_label, reg_label,
                                  this_time, spike_senders_indices, mean_rates=mean_rates, **kwargs)
        if self._adjust_time_axes_flag:
            axes = self._adjust_time_axes(axes, **kwargs)

        return self._return_figure(axes, figure_name)

    def plot_spike_detectors(self, spike_detectors, rates=None, mean_rates=None,
                             title=None, figure_name=None, figsize=None, **kwargs):
        # This method will plot a spike raster and, optionally,
        # it will superimpose the mean rate as a faded line.
        self._get_from_rates(rates)
        self.n_pops = len(spike_detectors)
        self.n_regions = np.max([len(spike_detector) for spike_detector in spike_detectors])
        self._prepare_time_axes_adjustment()

        # Create figure
        title = self._default_title(title, rates is not None, figure_name=None)
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
                self._neurons_axis_from_indices(neurons)
                axes = self._plot(axes, i_pop, i_region, pop_label, reg_label,
                                  region_spike_detector.spikes_times, senders, mean_rates=mean_rates, **kwargs)
        if self._adjust_time_axes_flag:
            axes = self._adjust_time_axes(axes, **kwargs)

        return self._return_figure(axes, figure_name)

    def plot_spike_events(self, spikes_events, rates=None, mean_rates=None,
                          title=None, figure_name=None, figsize=None, **kwargs):
        # This method will plot a spike raster and, optionally,
        # it will superimpose the mean rate as a faded line.
        self._get_from_rates(rates)
        self.n_pops = len(spikes_events)
        self.n_regions = np.max([len(pop_spikes_events) for pop_spikes_events in spikes_events])
        self._prepare_time_axes_adjustment()

        # Create figure
        title = self._default_title(title, rates is not None, figure_name=None)
        axes = []
        figure_name, figsize = self._get_figname_figsize(title, figure_name, figsize)
        pyplot.figure(figure_name, figsize=figsize)
        # Plot by arranging populations in columns and regions in rows
        for i_pop, (pop_label, pop_spikes_events) in enumerate(spikes_events.iteritems()):
            axes.append([])
            for i_region, (reg_label, region_spikes_events) in enumerate(pop_spikes_events.iteritems()):
                # Define spike senders and rates' axis
                neurons = region_spikes_events["senders"]
                self._neurons_axis_from_indices(neurons)
                axes = self._plot(axes, i_pop, i_region, pop_label, reg_label,
                                  region_spikes_events["times"], neurons, mean_rates=mean_rates, **kwargs)
                axes[i_pop][i_region] = self._format_axes(axes[i_pop][i_region], i_pop, i_region, pop_label, reg_label,
                                                          mean_rates=mean_rates, **kwargs)
        if self._adjust_time_axes_flag:
            axes = self._adjust_time_axes(axes, **kwargs)

        return self._return_figure(axes, figure_name)
