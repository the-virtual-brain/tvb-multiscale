# -*- coding: utf-8 -*-

import numpy
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import ensure_list, rotate_n_list_elements
from tvb.simulator.plot.timeseries_interactive import TimeSeriesInteractive, pylab, time_series_datatypes
from matplotlib.pyplot import rcParams

logger = initialize_logger(__name__)


class TimeseriesInteractivePlotter(TimeSeriesInteractive):

    def plot_time_series(self, **kwargs):
        """ Plot a view on the timeseries. """
        # Set title and axis labels
        # time_series_type = self.time_series.__class__.__name__
        # self.ts_ax.set(title = time_series_type)
        # self.ts_ax.set(xlabel = "Time (%s)" % self.units)

        # This assumes shape => (time, space)
        step = self.scaling * self.peak_to_peak
        if step == 0:
            offset = 0.0
        else:  # NOTE: specifying step in arange is faster, but it fence-posts.
            offset = numpy.arange(0, self.nsrs) * step
        if hasattr(self.ts_ax, 'autoscale'):
            self.ts_ax.autoscale(enable=True, axis='both', tight=True)

        self.ts_ax.set_yticks(offset)
        self.ts_ax.set_yticklabels(self.labels, fontsize=10)
        # import pdb; pdb.set_trace()

        # Light gray guidelines
        self.ts_ax.plot([self.nsrs * [self.time[self.time_view[0]]],
                         self.nsrs * [self.time[self.time_view[-1]]]],
                        numpy.vstack(2 * (offset,)), "0.85")

        # Determine colors and linestyles for each variable of the Timeseries
        linestyle = ensure_list(kwargs.pop("linestyle", "-"))
        colors = kwargs.pop("linestyle", None)
        if colors is not None:
            colors = ensure_list(colors)
        if self.data.shape[1] > 1:
            linestyle = rotate_n_list_elements(linestyle, self.data.shape[1])
            if not isinstance(colors, list):
                colors = (rcParams['axes.prop_cycle']).by_key()['color']
            colors = rotate_n_list_elements(colors, self.data.shape[1])
        else:
            # If no color,
            # or a color sequence is given in the input
            # but there is only one variable to plot,
            # choose the black color
            if colors is None or len(colors) > 1:
                colors = ["k"]
            linestyle = linestyle[:1]

        # Determine the alpha value depending on the number of modes/samples of the Timeseries
        alpha = 1.0
        if len(self.data.shape) > 3 and self.data.shape[3] > 1:
            alpha /= self.data.shape[3]

        # Plot the timeseries per variable and sample
        self.ts_view = []
        for i_var in range(self.data.shape[1]):
            for ii in range(self.data.shape[3]):
                # Plot the timeseries
                self.ts_view.append(self.ts_ax.plot(self.time[self.time_view],
                                                    offset + self.data[self.time_view, i_var, :, ii],
                                                    alpha=alpha, color=colors[i_var], linestyle=linestyle[i_var],
                                                    **kwargs))

        self.hereiam[0].remove()
        self.hereiam = self.whereami_ax.plot(self.time_view,
                                             numpy.zeros((len(self.time_view),)),
                                             'b-', linewidth=4)

        pylab.draw()

    def show(self, block=True, **kwargs):
        """ Generate the interactive time-series figure. """
        time_series_type = self.time_series.__class__.__name__
        msg = "Generating an interactive time-series plot for %s"
        if isinstance(self.time_series, time_series_datatypes.TimeSeriesSurface):
            logger.warning("Intended for region and sensors, not surfaces.")
        logger.info(msg % time_series_type)

        # Make the figure:
        self.create_figure()

        # Selectors
        # self.add_mode_selector()

        # Sliders
        self.add_window_length_slider()
        self.add_scaling_slider()
        # self.add_time_slider()

        # time-view buttons
        self.add_step_back_button()
        self.add_step_forward_button()
        self.add_big_step_back_button()
        self.add_big_step_forward_button()
        self.add_start_button()
        self.add_end_button()

        # Plot timeseries
        self.plot_time_series()

        pylab.show(block=block, **kwargs)
