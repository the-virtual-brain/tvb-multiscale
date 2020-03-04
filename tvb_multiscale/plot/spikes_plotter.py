# -*- coding: utf-8 -*-
from matplotlib import pyplot
import numpy as np
from xarray import DataArray
from pandas import Series
from tvb_scripts.plot.base_plotter import BasePlotter
from tvb_scripts.utils.data_structures_utils import ensure_list


class SpikesPlotter(BasePlotter):

    def plot_spikes(self, spikes_data, **kwargs):
        if isinstance(spikes_data, Series):
            # Either a Series (for populations) of Series (for regions) of SpikeDetectors'outputs...
            return self.plot_spike_detectors(spikes_data, **kwargs)
        else:
            # ...or a list (for populations) of TVB time series or TVB xarray time series (for regions)
            return self._plot_spikes(spikes_data, **kwargs)

    def _get_rates(self, rates, yticks):
        # If we plot rates, we need to....
        max_rate = 0.0
        time_lims = None
        xticks = None
        xticklabels = None
        get_rate_fun = None
        if rates is not None:
            plot_rates = True
            # ...compute the maximum rate to adjust the y axis accordingly
            if isinstance(rates, DataArray):  # In case we call this within the NEST interface context using pandas
                if rates.size > 0:
                    max_rate = np.max(rates).item()
                    time = rates.get_index(rates.dims[0])
                    get_rate_fun = lambda rates, i_region, i_pop: rates[:, i_pop, i_region].values
                else:
                    plot_rates = False
            else:  # Assuming TVB TimeSeries
                if rates.size > 0:
                    max_rate = rates.data.max()
                    time = rates.time
                    get_rate_fun = lambda rates, i_region, i_pop: rates[:, i_pop, i_region]
                else:
                    plot_rates = False
            if max_rate == 0:
                max_rate = 1.0  # if no spikes at all...
            rate_step = max_rate / len(yticks)
            yticklabels = np.arange(0.0, max_rate + rate_step, rate_step)
            yticklabels = ["%0.2f" % yticklabel for yticklabel in yticklabels]
            if plot_rates:
                # ..and set the time axis accordingly
                # Time axis
                time_lims = [time[0], time[-1]]
                time_step = np.int(np.ceil(np.maximum(1.0 * len(time) / 10, 1.0)))
                xticks = np.round(time[0:-1:time_step])
                xticklabels = ["%0.0f" % xtick for xtick in xticks]
        else:
            yticklabels = ["%d" % ytick for ytick in yticks]
            plot_rates = False
        return time, max_rate, get_rate_fun, time_lims, xticks, xticklabels, yticklabels, plot_rates

    def _get_y_ticks_labels(self, max_n_neurons):
        if max_n_neurons == 0:
            ylims = [0, 1]
            yticks = np.arange(0, 1.1, 0.1)
        else:
            ylims = [0, max_n_neurons]
            neurons_step = np.int(np.ceil(np.maximum(1.0 * max_n_neurons / 10, 1.0)))
            yticks = np.arange(0, max_n_neurons + neurons_step, neurons_step)
        return ylims, yticks, neurons_step

    def _get_figname_figsize(self, title="Population spikes and spike rate",
                             figure_name=None, figsize=None):
        if figure_name is None:
            figure_name = title
        if figsize is None:
            figsize = self.config.figures.LARGE_SIZE
        return figure_name, figsize

    def _plot_spikes(self, pop_spikes, rates=None,
                     title="Population spikes and spike rate",
                     figure_name=None, figsize=None, **kwargs):
        if isinstance(rates, DataArray):
            get_spikes_fun = lambda spikes, i_region: spikes[:, :, i_region].values.squeeze()
            get_time = lambda spikes, time: spikes.get_index(spikes.dims[0])
        else:
            get_spikes_fun = lambda spikes, i_region: spikes[:, :, i_region].squeeze()
            get_time = lambda spikes, time: time

        pop_spikes = ensure_list(pop_spikes)
        max_n_neurons = np.max([spikes.number_of_samples for spikes in pop_spikes])
        ylims, yticks, neurons_step = self._get_y_ticks_labels(max_n_neurons)

        time, max_rate, get_rate_fun, time_lims, xticks, xticklabels, yticklabels, plot_rates = \
            self._get_rates(rates, yticks)

        # Create figure
        n_pops = len(pop_spikes)
        n_regions = np.max([spikes.number_of_labels for spikes in pop_spikes])
        axes = []
        figure_name, figsize = self._get_figname_figsize(title, figure_name, figsize)
        pyplot.figure(figure_name, figsize=figsize)

        for i_pop, spikes in enumerate(pop_spikes):
            pop_label = spikes.labels_dimensions[spikes.labels_ordering[1]][0]
            spike_time = get_time(spikes, time)
            axes.append([])
            for i_region in range(spikes.number_of_labels):
                reg_label = spikes.labels_dimensions[spikes.labels_ordering[2]][i_region]

                axes[i_pop].append(pyplot.subplot(n_regions, n_pops, i_region * n_pops + i_pop + 1))

                if plot_rates:
                    # Adjust rates values to neurons axis range
                    rate_vals = get_rate_fun(rates, i_region, i_pop) / max_rate * max_n_neurons
                    axes[i_pop][i_region].plot(time, rate_vals,
                                               linestyle=kwargs.get("rate_linestyle", "-"),
                                               color=kwargs.get("rate_color", "k"),
                                               linewidth=kwargs.get("rate_linewidth", 5.0),
                                               alpha=kwargs.get("rate_alpha", 0.5))

                # Plot spikes
                indices = np.argwhere(get_spikes_fun(spikes, i_region) > 0.0)
                this_time = spike_time[indices[:, 0]]
                spike_senders_indices = spike_time[indices[:, 1]]
                axes[i_pop][i_region].plot(this_time, spike_senders_indices,
                                           linestyle="None",
                                           marker=kwargs.get("spikes_marker", "o"),
                                           markerfacecolor=kwargs.get("spikes_color", "k"),
                                           markeredgecolor=kwargs.get("spikes_color", "k"),
                                           markersize=kwargs.get("spikes_markersize", 2.0),
                                           alpha=kwargs.get("spikes_alpha", 1.0))

                axes[i_pop][i_region].set_ylim(ylims)
                axes[i_pop][i_region].set_yticks(yticks)
                if time_lims is not None:
                    axes[i_pop][i_region].set_xlim(time_lims)
                if xticks is not None:
                    axes[i_pop][i_region].set_xticks(xticks)

                if i_pop == 0:
                    axes[i_pop][i_region].set_yticklabels(yticklabels)
                    axes[i_pop][i_region].set_ylabel("%s (spikes/s)" % reg_label)
                else:
                    axes[i_pop][i_region].set_yticklabels([])

                if i_region == 0:
                    axes[i_pop][i_region].set_title(pop_label)

                if i_region == n_regions - 1:
                    axes[i_pop][i_region].set_xticklabels(xticklabels)
                    axes[i_pop][i_region].set_xlabel("Time (ms)")
                else:
                    axes[i_pop][i_region].set_xticklabels([])

        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()

        return pyplot.gcf(), axes

    def plot_spike_detectors(self, spike_detectors, rates=None,
                             title="Population spikes and spike rate",
                             figure_name=None, figsize=None, **kwargs):
        # This method will plot a spike raster and, optionally,
        # it will superimpose the mean rate as a faded line.

        # Y axis limits and ticks according to maximum number of neurons
        max_n_neurons = np.max([np.max(spike_detector.number_of_neurons) for spike_detector in spike_detectors])
        ylims, yticks, neurons_step = self._get_y_ticks_labels(max_n_neurons)

        time, max_rate, get_rate_fun, time_lims, xticks, xticklabels, yticklabels, plot_rates = \
            self._get_rates(rates, yticks)

        # Create figure
        n_pops = len(spike_detectors)
        n_regions = np.max([len(spike_detector) for spike_detector in spike_detectors])
        axes = []
        figure_name, figsize = self._get_figname_figsize(title, figure_name, figsize)
        pyplot.figure(figure_name, figsize=figsize)

        # Plot by arranging populations in columns and regions in rows
        for i_pop, (pop_label, pop_spike_detector) in enumerate(spike_detectors.iteritems()):
            axes.append([])
            for i_region, (reg_label, region_spike_detector) in enumerate(pop_spike_detector.iteritems()):
                # Define spike senders and rates' axis
                neurons = np.unique(region_spike_detector.neurons).tolist()
                spike_senders_indices = [neurons.index(sender) for sender in region_spike_detector.spikes_senders]

                axes[i_pop].append(pyplot.subplot(n_regions, n_pops, i_region * n_pops + i_pop + 1))

                if plot_rates:
                    # Adjust rates values to neurons axis range
                    rate_vals = get_rate_fun(rates, i_region, i_pop) / max_rate * max_n_neurons
                    axes[i_pop][i_region].plot(time, rate_vals,
                                               linestyle=kwargs.get("rate_linestyle", "-"),
                                               color=kwargs.get("rate_color", "k"),
                                               linewidth=kwargs.get("rate_linewidth", 5.0),
                                               alpha=kwargs.get("rate_alpha", 0.5))

                # Plot spikes
                axes[i_pop][i_region].plot(region_spike_detector.spikes_times, spike_senders_indices,
                                           linestyle="None",
                                           marker=kwargs.get("spikes_marker", "o"),
                                           markerfacecolor=kwargs.get("spikes_color", "k"),
                                           markeredgecolor=kwargs.get("spikes_color", "k"),
                                           markersize=kwargs.get("spikes_markersize", 2.0),
                                           alpha=kwargs.get("spikes_alpha", 1.0))

                axes[i_pop][i_region].set_ylim(ylims)
                axes[i_pop][i_region].set_yticks(yticks)
                if time_lims is not None:
                    axes[i_pop][i_region].set_xlim(time_lims)
                if xticks is not None:
                    axes[i_pop][i_region].set_xticks(xticks)

                if i_pop == 0:
                    axes[i_pop][i_region].set_yticklabels(yticklabels)
                    axes[i_pop][i_region].set_ylabel("%s (spikes/s)" % reg_label)
                else:
                    axes[i_pop][i_region].set_yticklabels([])

                if i_region == 0:
                    axes[i_pop][i_region].set_title(pop_label)

                if i_region == n_regions - 1:
                    axes[i_pop][i_region].set_xticklabels(xticklabels)
                    axes[i_pop][i_region].set_xlabel("Time (ms)")
                else:
                    axes[i_pop][i_region].set_xticklabels([])

        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()

        return pyplot.gcf(), axes
