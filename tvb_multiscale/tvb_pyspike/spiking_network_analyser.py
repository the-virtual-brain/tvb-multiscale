# -*- coding: utf-8 -*-

import numpy as np
from pandas import Series
from xarray import DataArray

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.data_analysis.spiking_network_analyser_base import SpikingNetworkAnalyserBase

from tvb.basic.neotraits.api import Attr
from tvb.contrib.scripts.utils.data_structures_utils import sort_events_by_x_and_y


LOG = initialize_logger(__name__)


class SpikingNetworkAnalyser(SpikingNetworkAnalyserBase):

    """SpikingNetworkAnalyserBase
       - gets data from the output devices of a SpikingNetwork,
       (TODO: - or reads them from files,)
       - performs computations of spike trains' synchronization,
       - and returns the results in the desired output type
         (numpy.array, xarray.DataArray, TVB TimeSeries, pandas.Series of xarray.DataArray)
       """

    resample = Attr(field_type=bool, default=True, required=True, label="Resample flag",
                    doc="""If resample is True, we return time series results 
                           on a fixed time step time vector instead of along spikes' times""")

    _spikes_trains_name = None

    def _assert_start_end_times_from_spikes_trains(self, spikes):
        """Method to set the starting and ending time of a computation,
           given the start_time and end_time attributes and the input spikes' times, if any.
           Arguments:
            - spikes: a Spike Train or a numpy array of spikes' times
           Returns:
            - start_time: float (ms)
            - end_time: float (ms)
        """
        # Initialize from the class instance attributes
        flatten_spike_train = []
        for spike in spikes:
            flatten_spike_train += list(spike)
        return super(SpikingNetworkAnalyser, self)._assert_start_end_times_from_spikes_times(flatten_spike_train)

    def _get_spikes_times_per_sender_from_spikes_events(self, spikes, number_of_neurons=1):
        """Method to safely get a numpy.array of spikes
           from a spikes input that might be a dict or pandas.Series.
           Arguments:
            - spikes: a dict or pandas.Series with key "times" for spikes' times and "senders" for spikes' senders.
        """
        events = sort_events_by_x_and_y(spikes, x="senders", y="times", hashfun=tuple)
        ii = 0
        while len(events) < number_of_neurons:
            events["dummy%s" % ii] = []
        return events

    def compute_spikes_trains(self, spikes, number_of_neurons=1, edges=[]):
        """Method to compute a collection of pyspike.SpikesTrain instances from an array of spikes" times.
           Arguments:
            - spikes: a collection of collections of spikes' times.
           Returns:
            - the neo.core.SpikeTrain.
        """
        from pyspike import SpikeTrain
        if len(edges) == 0:
            edges = list(self._assert_start_end_times_from_spikes_trains(spikes))
        spiketrains = []
        for spike in spikes:
            spiketrains.append(SpikeTrain(spike, edges))
        while len(spiketrains) < number_of_neurons:
            spiketrains.append(SpikeTrain([], edges))
        return spiketrains

    @property
    def spikes_trains_name(self):
        """A method to set and return the name of the Spikes Train data type."""
        if not self._spikes_trains_name:
            self._spikes_trains_name = self._get_comput_res_type(self.compute_spikes_trains)
        return self._spikes_trains_name

    def _assert_spikes_trains(self, spikes, number_of_neurons=1, **kwargs):
        """A method to assert that an argument is a collection of pyspike.SpikesTrain instances.
           If not, a list of pyspike.SpikesTrain instances is computed
           from the dictionary of spikes' times per spikes' senders.
           Arguments:
            - spikes: a pyspike.SpikesTrain or a dictionary or pandas.Series of spikes' times per spikes' senders,
                      or a dictionary or pandas.Series of spikes' events with keys "times" and "senders"
           Returns:
            - the collection of pyspike.SpikesTrain instances.
        """
        from pyspike import SpikeTrain
        if isinstance(spikes, SpikeTrain):
            return [spikes]
        elif np.all([isinstance(spike_train, SpikeTrain) for spike_train in spikes]):
            return spikes
        else:
            if isinstance(spikes, Series):
                spikes = spikes.to_dict()
            if isinstance(spikes, dict):
                if "senders" in spikes.keys():
                    spikes = self._get_spikes_times_per_sender_from_spikes_events(spikes.copy(),
                                                                                  number_of_neurons, **kwargs)
                return self.compute_spikes_trains(list(spikes.values()), number_of_neurons)
            else:
                return self.compute_spikes_trains([spikes], number_of_neurons)

    def _resample(self, old_time, values, new_time=None, t_start=None, t_stop=None, dt=None, **interp_kwargs):
        from scipy.interpolate import interp1d
        F = interp1d(old_time, values, fill_value=interp_kwargs.pop("fill_value", 'extrapolate'), **interp_kwargs)
        if new_time is None:
            if t_start is None:
                t_start = old_time.min()
            if t_stop is None:
                t_stop = old_time.max()
            if dt is None:
                dt = self.period
            new_time = np.linspace(t_start, t_stop, dt)
        return F(new_time)

    def _compute_pyspike_float_result(self, spikes, fun, res_type, number_of_neurons=1, **kwargs):
        spikes_trains = self._assert_spikes_trains(spikes, number_of_neurons, edges=kwargs.pop("edges", []))
        if len(spikes_trains):
            result = fun(spikes_trains, **kwargs)
        else:
            result = 0.0
        return {res_type: DataArray([float(result)]).squeeze(), self.spikes_trains_name: spikes_trains}

    def _compute_pyspike_matrix(self, spikes, fun, res_type, number_of_neurons=1, **kwargs):
        spikes_trains = self._assert_spikes_trains(spikes, number_of_neurons, edges=kwargs.pop("edges", []))
        if len(spikes_trains):
            result = fun(spikes_trains, **kwargs)
        else:
            result = np.array([[]])
        return {res_type: DataArray(result, dims=["Neuron_i", "Neuron_j"],
                                    coords={"Neuron_i": np.arange(result.shape[0]),
                                            "Neuron_j": np.arange(result.shape[1])}),
                self.spikes_trains_name: spikes_trains}

    def _compute_time_vector(self, spikes_train=None, start_end_times=[], dt=None):
        if dt is None:
            dt = self.period
        if len(start_end_times) == 0:
            start_end_times = list(self._assert_start_end_times_from_spikes_trains(spikes_train))
        time = np.arange(start_end_times[0], start_end_times[-1] + self._fmin_resolution, dt)
        return time

    def _compute_pyspike_time_series(self, spikes, fun, res_type, number_of_neurons=1, **kwargs):
        spikes_trains = self._assert_spikes_trains(spikes, number_of_neurons, edges=kwargs.pop("edges", []))
        if len(spikes_trains) > 0:
            times, values = fun(spikes_trains).get_plottable_data()
            if self.resample:
                time = self._compute_time_vector(spikes_trains)
                values = self._resample(times, values, time, **kwargs)
            else:
                time = times
        else:
            time = self._compute_time_vector(spikes_trains)
            values = 0.0 * time
        # TODO: A better solution for handling the last time point!
        try:
            return {res_type: DataArray(values, dims=["Time"], coords={"Time": time}),
                    self.spikes_trains_name: spikes_trains}
        except Exception as e:
            LOG.warning("Failed with exception %s!\n"
                        "Length of time is %d and of result values is %d.\n"
                        "Removing last time point and retrying!" % (str(e), len(time), len(values)))
            return {res_type: DataArray(values, dims=["Time"], coords={"Time": time[:-1]}),
                    self.spikes_trains_name: spikes_trains}

    def compute_spikes_sync(self, spikes, number_of_neurons=1, **kwargs):
        """A method to compute spike trains' synchronization using the pyspike.spike_sync method.
           Arguments:
            - spikes: a collection of pyspike.SpikeTrain instances
                      or an array of spikes' times or a collection of such arrays
                      or a dict of spikes' times (values) per spikes' senders (keys)
           Returns:
            - a dictionary of the following key-value pair(s):
             "spikes_sync": xarray.DataArray(spikes_sync).squeeze(), where spikes_sync is a float
             "spikes_trains": the collection of pyspike.SpikeTrain instances used for the computation
        """
        from pyspike import spike_sync
        return self._compute_pyspike_float_result(spikes, spike_sync, self._get_comput_res_type(),
                                                  number_of_neurons, **kwargs)

    def compute_spikes_sync_matrix(self, spikes, number_of_neurons=1, **kwargs):
        """A method to compute bivariate spike trains' synchronization matrix
           using the pyspike.spike_sync_matrix method.
           Arguments:
            - spikes: a collection of pyspike.SpikeTrain instances
                      or an array of spikes' times or a collection of such arrays
                      or a dict of spikes' times (values) per spikes' senders (keys)
           Returns:
            - a dictionary of the following key-value pair(s):
             "spikes_sync_matrix": xarray.DataArray(spikes_sync_matrix), where spikes_sync_matrix is a 2D matrix
             "spikes_trains": the collection of pyspike.SpikeTrain instances used for the computation
        """
        from pyspike import spike_sync_matrix
        return self._compute_pyspike_matrix(spikes, spike_sync_matrix, self._get_comput_res_type(),
                                            number_of_neurons, **kwargs)

    def compute_spikes_sync_time_series(self, spikes, number_of_neurons=1, **kwargs):
        """A method to compute spike trains' synchronization time series using the pyspike.spike_sync method.
           Arguments:
            - spikes: a collection of pyspike.SpikeTrain instances
                      or an array of spikes' times or a collection of such arrays
                      or a dict of spikes' times (values) per spikes' senders (keys)
           Returns:
            - a dictionary of the following key-value pair(s):
             "spikes_sync_time_series": xarray.DataArray(spikes_sync_time_series).squeeze()
                where spikes_sync_time_series is the spike synchronization time series (numpy.array)
             "spikes_trains": the collection of pyspike.SpikeTrain instances used for the computation
        """
        from pyspike import spike_sync_profile
        return self._compute_pyspike_time_series(spikes, spike_sync_profile,
                                                 self._get_comput_res_type(), number_of_neurons, **kwargs)

    def compute_spikes_distance(self, spikes, number_of_neurons=1, **kwargs):
        """A method to compute spike trains' distance using the pyspike.spike_distance method.
           Arguments:
            - spikes: a collection of pyspike.SpikeTrain instances
                      or an array of spikes' times or a collection of such arrays
                      or a dict of spikes' times (values) per spikes' senders (keys)
           Returns:
            - a dictionary of the following key-value pair(s):
             "spikes_distance": xarray.DataArray(spikes_distance).squeeze(), where spikes_distance is a float
             "spikes_trains": the collection of pyspike.SpikeTrain instances used for the computation
        """
        from pyspike import spike_distance
        return self._compute_pyspike_float_result(spikes, spike_distance, self._get_comput_res_type(),
                                                  number_of_neurons, **kwargs)

    def compute_spikes_distance_matrix(self, spikes, number_of_neurons=1, **kwargs):
        """A method to compute bivariate spike trains' distance matrix using the pyspike.spike_distance_matrix method.
           Arguments:
            - spikes: a collection of pyspike.SpikeTrain instances
                      or an array of spikes' times or a collection of such arrays
                      or a dict of spikes' times (values) per spikes' senders (keys)
           Returns:
            - a dictionary of the following key-value pair(s):
             "spikes_distance_matrix": xarray.DataArray(spikes_distance_matrix),
                                       where spikes_distance_matrix is a 2D matrix
             "spikes_trains": the collection of pyspike.SpikeTrain instances used for the computation
        """
        from pyspike import spike_distance_matrix
        return self._compute_pyspike_matrix(spikes, spike_distance_matrix, self._get_comput_res_type(),
                                            number_of_neurons, **kwargs)

    def compute_spikes_distance_time_series(self, spikes, number_of_neurons=1, **kwargs):
        """A method to compute spike trains' distance time series using the pyspike.spike_profile method.
           Arguments:
            - spikes: a collection of pyspike.SpikeTrain instances
                      or an array of spikes' times or a collection of such arrays
                      or a dict of spikes' times (values) per spikes' senders (keys)
           Returns:
            - a dictionary of the following key-value pair(s):
             "spikes_distance_time_series": xarray.DataArray(spikes_distance_time_series).squeeze()
                where spikes_distance_time_series is the spike distance time series (numpy.array)
             "spikes_trains": the collection of pyspike.SpikeTrain instances used for the computation
        """
        from pyspike import spike_profile
        return self._compute_pyspike_time_series(spikes, spike_profile, self._get_comput_res_type(),
                                                 number_of_neurons, **kwargs)

    def compute_spikes_isi_distance(self, spikes, number_of_neurons=1, **kwargs):
        """A method to compute spike trains' distance using the pyspike.isi_distance method.
           Arguments:
            - spikes: a collection of pyspike.SpikeTrain instances
                      or an array of spikes' times or a collection of such arrays
                      or a dict of spikes' times (values) per spikes' senders (keys)
           Returns:
            - a dictionary of the following key-value pair(s):
             "spikes_isi_distance": xarray.DataArray(spikes_isi_distance).squeeze(),
                                    where spikes_isi_distance is a float
             "spikes_trains": the collection of pyspike.SpikeTrain instances used for the computation
        """
        from pyspike import isi_distance
        return self._compute_pyspike_float_result(spikes, isi_distance, self._get_comput_res_type(),
                                                  number_of_neurons, **kwargs)

    def compute_spikes_isi_distance_matrix(self, spikes, number_of_neurons=1, **kwargs):
        """A method to compute bivariate spike trains' distance matrix using the pyspike.isi_distance_matrix method.
           Arguments:
            - spikes: a collection of pyspike.SpikeTrain instances
                      or an array of spikes' times or a collection of such arrays
                      or a dict of spikes' times (values) per spikes' senders (keys)
           Returns:
            - a dictionary of the following key-value pair(s):
             "spikes_isi_distance_matrix": xarray.DataArray(spikes_isi_distance_matrix),
                                           where spikes_isi_distance_matrix is a 2D matrix
             "spikes_trains": the collection of pyspike.SpikeTrain instances used for the computation
        """
        from pyspike import isi_distance_matrix
        return self._compute_pyspike_matrix(spikes, isi_distance_matrix, self._get_comput_res_type(),
                                            number_of_neurons, **kwargs)

    def compute_spikes_isi_distance_time_series(self, spikes, number_of_neurons=1, **kwargs):
        """A method to compute spike trains' distance time series using the pyspike.isi_profile method.
           Arguments:
            - spikes: a collection of pyspike.SpikeTrain instances
                      or an array of spikes' times or a collection of such arrays
                      or a dict of spikes' times (values) per spikes' senders (keys)
           Returns:
            - a dictionary of the following key-value pair(s):
             "spikes_isi_distance_time_series": xarray.DataArray(spikes_isi_distance_time_series).squeeze()
                where spikes_isi_distance_time_series is the spike isi_distance time series (numpy.array)
             "spikes_trains": the collection of pyspike.SpikeTrain instances used for the computation
        """
        from pyspike import isi_profile
        return self._compute_pyspike_time_series(spikes, isi_profile, self._get_comput_res_type(),
                                                 number_of_neurons, **kwargs)
