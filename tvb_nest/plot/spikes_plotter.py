# -*- coding: utf-8 -*-
import numpy as np

from tvb.simulator.plot.base_plotter import BasePlotter
from matplotlib import pyplot


class SpikesPlotter(BasePlotter):

    def plot_spikes(self, spike_detectors_dict, time, rates=None, max_rate=None,
                    title="Population spikes and spike rate",
                    figure_name=None, figsize=None, **kwargs):

        # Y axis
        max_n_neurons = np.max([spike_detector.number_of_neurons
                                for spike_detector in spike_detectors_dict.values()])
        ylims = [0, max_n_neurons]
        neurons_step = np.int(np.ceil(np.maximum(1.0 * max_n_neurons / 10, 1.0)))
        yticks = np.arange(0, max_n_neurons + neurons_step, neurons_step)

        if rates is not None:
            if max_rate is None:
                max_rate = np.max([np.max([np.max(region_vals) for region_vals in pop_rate.values()])
                                   for pop_rate in rates.values()])
            if max_rate == 0:
                max_rate = 1.0
            rate_step = max_rate / len(yticks)
            yticklabels = np.arange(0.0, max_rate + rate_step, rate_step)
            yticklabels = ["%0.2f" % yticklabel for yticklabel in yticklabels]
            plot_rates = True
        else:
            yticklabels = ["%d" % ytick for ytick in yticks]
            plot_rates = False

        # Time axis
        time_lims = [time[0], time[-1]]
        time_step = np.int(np.ceil(np.maximum(1.0 * len(time) / 10, 1.0)))
        xticks = time[0:-1:time_step]
        xticklabels = ["%0.1f" % xtick for xtick in xticks]

        # Create figure
        n_pops = len(spike_detectors_dict.keys())
        n_regions = np.max([len(spike_detector.keys())
                            for spike_detector in spike_detectors_dict.values()])
        axes = []
        if figure_name is None:
            figure_name = title
        if figsize is None:
            figsize = self.config.figures.LARGE_SIZE
        pyplot.figure(figure_name, figsize=figsize)

        # Plot
        for i_pop, (pop_label, pop_spike_detector) in enumerate(spike_detectors_dict.items()):
            axes.append([])
            for i_region, (reg_label, region_spike_detector) in enumerate(pop_spike_detector.items()):
                # Define spike senders and rates' axis
                neurons = np.unique(region_spike_detector.neurons).tolist()
                spike_senders_indices = [neurons.index(sender) for sender in region_spike_detector.spike_senders]

                axes[i_pop].append(pyplot.subplot(n_regions, n_pops, i_region * n_pops + i_pop + 1))

                if plot_rates:
                    # Adjust rates values to neurons axis range
                    rate_vals = rates[pop_label][reg_label] / max_rate * max_n_neurons
                    axes[i_pop][i_region].plot(time, rate_vals,
                                               linestyle=kwargs.get("rate_linestyle", "-"),
                                               color=kwargs.get("rate_color", "k"),
                                               linewidth=kwargs.get("rate_linewidth", 5.0),
                                               alpha=kwargs.get("rate_alpha", 0.5))

                # Plot spikes
                axes[i_pop][i_region].plot(region_spike_detector.spike_times, spike_senders_indices,
                                           linestyle="None",
                                           marker=kwargs.get("spikes_marker", "o"),
                                           markerfacecolor=kwargs.get("spikes_color", "k"),
                                           markeredgecolor=kwargs.get("spikes_color", "k"),
                                           markersize=kwargs.get("spikes_markersize", 2.0),
                                           alpha=kwargs.get("spikes_alpha", 1.0))

                axes[i_pop][i_region].set_ylim(ylims)
                axes[i_pop][i_region].set_yticks(yticks)
                axes[i_pop][i_region].set_xlim(time_lims)
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
