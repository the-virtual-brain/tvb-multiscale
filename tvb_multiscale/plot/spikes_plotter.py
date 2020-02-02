# -*- coding: utf-8 -*-
import numpy as np
from xarray import DataArray
from tvb_scripts.plot.base_plotter import BasePlotter
from matplotlib import pyplot


class SpikesPlotter(BasePlotter):

    def plot_spikes(self, spike_detectors, rates=None,
                    title="Population spikes and spike rate",
                    figure_name=None, figsize=None, **kwargs):
        # This method will plot a spike raster and, optionally,
        # it will superimpose the mean rate as a faded line.

        # Y axis limits and ticks according to maximum number of neurons
        max_n_neurons = np.max([np.max(spike_detector.number_of_neurons) for spike_detector in spike_detectors])
        if max_n_neurons == 0:
            ylims = [0, 1]
            yticks = np.arange(0, 1.1, 0.1)
        else:
            ylims = [0, max_n_neurons]
            neurons_step = np.int(np.ceil(np.maximum(1.0 * max_n_neurons / 10, 1.0)))
            yticks = np.arange(0, max_n_neurons + neurons_step, neurons_step)

        # If we plot rates, we need to....
        max_rate = 0.0
        time_lims = None
        xticks = None
        xticklabels = None
        if rates is not None:
            plot_rates = True
            # ...compute the maximum rate to adjust the y axis accordingly
            if isinstance(rates, DataArray):  # In case we call this within the NEST interface context using pandas
                if rates.size > 0:
                    max_rate = np.max(rates).item()
                    time = rates.get_index(rates.dims[0])
                    get_rate_fun = lambda reg_lbl, pop_lbl, i_region, i_pop: rates[:, i_pop, i_region].values
                else:
                    plot_rates = False
            else:  # Assuming TVB TimeSeries
                if rates.size > 0:
                    max_rate = rates.data.max()
                    time = rates.time
                    get_rate_fun = lambda reg_lbl, pop_lbl, i_region, i_pop: rates[:, i_pop, i_region]
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
        # Create figure
        n_pops = len(spike_detectors)
        n_regions = np.max([len(spike_detector) for spike_detector in spike_detectors])
        axes = []
        if figure_name is None:
            figure_name = title
        if figsize is None:
            figsize = self.config.figures.LARGE_SIZE
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
                    rate_vals = get_rate_fun(reg_label, pop_label, i_region, i_pop) / max_rate * max_n_neurons
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
