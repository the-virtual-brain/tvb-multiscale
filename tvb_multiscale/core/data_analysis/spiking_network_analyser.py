# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import OrderedDict

import numpy as np
from pandas import Series
from xarray import DataArray

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.spiking_models.network import SpikingNetwork
from tvb_multiscale.core.utils.data_structures_utils import cross_dimensions_and_coordinates_MultiIndex, \
    get_ordered_dimensions, get_caller_fun_name

from tvb.basic.neotraits.api import HasTraits, Attr, Float
from tvb.datatypes import connectivity

from tvb.contrib.scripts.utils.data_structures_utils import \
    ensure_list, concatenate_heterogeneous_DataArrays, sort_events_by_x_and_y
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeries, TimeSeriesRegion


LOG = initialize_logger(__name__)


class SpikingNetworkAnalyser(HasTraits):
    """SpikingNetworkAnalyser
       - gets data from the output devices of a SpikingNetwork,
       - or reads them from files,
       - performs computations of mean field quantities, including spikes_train' rates,
       - and returns the results in the desired output type
         (numpy.array, xarray.DataArray, TVB TimeSeries, pandas.Series of xarray.DataArray)
       """

    spikeNet = Attr(field_type=SpikingNetwork,
                    label="Spiking Network",
                    default=None,
                    required=False,
                    doc="""An instance of a SpikingNetwork class""")

    start_time = Float(
        label="Start time",
        default=None,
        required=False,
        doc="""Starting time of data (ms)""")

    end_time = Float(
        label="End time",
        default=None,
        required=False,
        doc="""Ending time of data (ms)""")

    period = Float(
        label="Sampling period",
        default=1.0,
        required=True,
        doc="""Sampling period (ms)""")

    transient = Float(
        label="Transient time",
        default=0.0,
        required=True,
        doc="""Transient time to be excluded from computations, 
               but to not be cut out from data returned (ms). Default = 0.0.
               In case that start_time is set by the first spike time, 
               the start_time of computations is set as the maximum(start_time, transient).
               In all other cases, the start_time of computations is given as start_time + transient.""")

    output_type = Attr(field_type=str, default="array", required=True,
                       label="Output type option for the results.",
                       doc="""The output type of the results, which can be either 'array' (Default), 
                             in which case a DataArray or a Series of DataArrays 
                             for unmerged heterogeneous results is returned, 
                             or "TVB", in which case a TVB TimeSeries instance is returned.""")

    force_homogeneous_results = Attr(field_type=bool, default=True, required=True,
                                     label="Flag to merge heteregoneous results.",
                                     doc="""If force_homogeneous_results is True, 
                                            DataArrays from different regions and or populations " \
                                           "are merged to a single DataArray. It defaults to True and "
                                           "it is also forced to True, if the output_type = 'TVB' TimeSeries.""")

    connectivity = Attr(
        field_type=connectivity.Connectivity,
        label="Long-range connectivity",
        default=None,
        required=False,
        doc="""A tvb.datatypes.Connectivity object which contains the
         structural long-range connectivity data (i.e., white-matter tracts). In
         combination with the ``Long-range coupling function`` it defines the inter-regional
         connections. These couplings undergo a time delay via signal propagation
         with a propagation speed of ``Conduction Speed``.
         \nIt is necessary if the output_type="TVB" for TimeSeriesRegion class instance.""")

    flatten_neurons_inds = Attr(field_type=bool, default=True, required=False,
                                label="Flag to flatten neurons indices",
                                doc="""If flatten_neurons_inds is True, "
                                      "we substitute the global neurons' indices by their local rank""")

    return_data = Attr(field_type=bool, default=False, required=True,
                       label="Flag to return data for plotting.",
                       doc="""If return_data is True, 
                              the data used for the computation of the measures are also returned.""")

    _spikes_train_name = None

    _binned_spikes_trains_name = None

    def _get_method_name(self, method=None, caller_id=2):
        if method:
            return method.__name__
        else:
            return get_caller_fun_name(caller_id)

    def _get_data_name(self, method=None):
        return self._get_method_name(method, caller_id=3).split("get_")[-1].split("_from")[0]

    def get_spikes_from_device(self, device, name="", **kwargs):
        data_name = self._get_data_name()
        return {data_name: Series(device.get_spikes_events(**kwargs), name=name),
                "number_of_neurons": device.number_of_neurons,
                "data_name": data_name}

    def get_spikes_times_from_device(self, device, name="", **kwargs):
        data_name = self._get_data_name()
        return {data_name: Series(np.array(np.sort(device.get_spikes_times(**kwargs)), name=name)),
                "number_of_neurons": device.number_of_neurons,
                "data_name": data_name}

    def get_spikes_times_by_neuron_from_device(self, device, name="", **kwargs):
        kwargs["full_senders"] = True
        spikes_times_by_neuron = device.get_spikes_times_by_neurons(**kwargs)
        number_of_neurons = len(spikes_times_by_neuron)
        if self.flatten_neurons_inds:
            spikes_times_by_neuron = OrderedDict(zip(range(number_of_neurons), spikes_times_by_neuron.values()))
        data_name = self._get_data_name()
        return {data_name: Series(spikes_times_by_neuron, name=name),
                "number_of_neurons": number_of_neurons,
                "data_name": data_name}

    def get_data_by_neuron_from_device(self, device, name="", **kwargs):
        this_kwargs = deepcopy(kwargs)
        this_kwargs["flatten_neurons_inds"] = this_kwargs.get("flatten_neurons_inds", self.flatten_neurons_inds)
        data = device.get_data(**kwargs)
        data_name = self._get_data_name()
        if name:
            data.name = name
        return {data_name: data,
                "number_of_neurons": data.shape[-2],
                "data_name": data_name}

    def get_mean_data_from_device(self, device, name="", **kwargs):
        data_name = self._get_data_name()
        data = device.get_mean_data(**kwargs)
        if name:
            data.name = name
        return {data_name: data,
                "number_of_neurons": device.number_of_neurons,
                "data_name": data_name}

    def get_total_data_from_device(self, device, name="", **kwargs):
        data_name = self._get_data_name()
        data = device.get_total_data(**kwargs)
        if name:
            data.name = name
        return {data_name: data,
                "number_of_neurons": device.number_of_neurons,
                "data_name": data_name}

    def assert_start_end_times_from_spikes_times(self, spikes):
        start_time = self.start_time
        end_time = self.end_time
        if start_time is None:
            if len(spikes):
                start_time = np.min(spikes)
            else:
                start_time = 0.0
            start_time = np.maximum(start_time, self.transient)
        else:
            start_time = self.start_time + self.transient
        if end_time is None:
            if len(spikes):
                end_time = np.max(spikes)
            else:
                end_time = start_time + self.period
        if end_time <= start_time:
            end_time = start_time + self.period
        return start_time, end_time

    def assert_start_end_times_from_data_time(self, time):
        start_time = self.start_time
        end_time = self.end_time
        if start_time is None:
            if len(time):
                start_time = np.min(time)
            else:
                start_time = 0.0
        start_time = start_time + self.transient
        if end_time is None:
            if len(time):
                end_time = np.max(time)
            else:
                end_time = start_time + self.period
        if end_time <= start_time:
            end_time = start_time + self.period
        return start_time, end_time

    def _get_spikes_times_from_spikes_events(self, spikes):
        if isinstance(spikes, (dict, Series)):
            return np.array(spikes["times"])
        else:
            return np.array(spikes)

    def _get_comput_res_type(self, method=None):
        return self._get_method_name(method, caller_id=3).split("compute_")[-1]

    def compute_spikes_train(self, spikes):
        pass

    @property
    def spikes_train_name(self):
        if not self._spikes_train_name:
            self._spikes_train_name = self._get_comput_res_type(self.compute_spikes_train)
        return self._spikes_train_name

    def assert_spike_train(self, spikes):
        pass

    def compute_binned_spikes_trains(self, spikes_trains, binsize=None, num_bins=None):
        pass

    @property
    def binned_spikes_trains_name(self):
        if not self._binned_spikes_trains_name:
            self._binned_spikes_trains_name = self._get_comput_res_type(self.compute_binned_spikes_trains)
        return self._binned_spikes_trains_name

    def assert_binned_spikes_trains(self, spikes_trains, binsize=None, num_bins=None):
        pass

    def compute_rate(self, spikes, number_of_neurons=1, name="", duration=None, **kwargs):
        res_type = self._get_comput_res_type()
        spikes_times = self._get_spikes_times_from_spikes_events(spikes)
        if not duration:
            start_time, end_time = self.assert_start_end_times_from_spikes_times(spikes_times)
            duration = end_time - start_time
            if duration < 0.0:
                raise ValueError("start_time (=%g) cannot be smaller than end_time (=%g)!" % (start_time, end_time))
            elif duration == 0.0:
                duration = 1.0
        return {res_type: DataArray(len(spikes_times) / duration * 1000, name=name).squeeze()}

    def compute_mean_rate(self, spikes, number_of_neurons=1, name="", duration=None, **kwargs):
        res_type = self._get_comput_res_type()
        res2_type = self._get_comput_res_type(self.compute_rate)
        results = self.compute_rate(spikes, name=name, duration=duration, **kwargs)
        results[res_type] = results[res2_type] / number_of_neurons
        del results[res2_type]
        return results

    def compute_rate_time_series(self, spikes_times, name="", **elephant_kwargs):
        pass

    def compute_mean_rate_time_series(self, spikes_times, number_of_neurons=1, name="", **elephant_kwargs):
        res_type = self._get_comput_res_type()
        res2_type = self._get_comput_res_type(self.compute_rate_time_series)
        results = self.compute_rate_time_series(spikes_times, name=name, **elephant_kwargs)
        results[res_type] = results[res2_type] / number_of_neurons
        del results[res2_type]
        return results

    def compute_spikes_rates_by_neuron(self, spikes, number_of_neurons=1, name="", rate_method=None, **kwargs):
        if rate_method is None:
            rate_method = self.compute_rate_time_series
        res_type = self._get_comput_res_type(rate_method) + "_by_neuron"
        if "times" in spikes.keys():
            spikes_times_by_neuron = Series(sort_events_by_x_and_y(spikes, x="senders", y="times"))
        else:
            spikes_times_by_neuron = spikes
        neurons = list(spikes_times_by_neuron.keys())
        spikes_times_by_neuron = list(spikes_times_by_neuron.values)
        rates = OrderedDict()
        if len(neurons) < number_of_neurons:
            neurons = np.arange(number_of_neurons)
        for i_neuron, neuron in enumerate(neurons):
            if len(spikes_times_by_neuron):
                spikes_times = spikes_times_by_neuron.pop(0)
            else:
                spikes_times = []
            rates[neuron] = \
                    list(rate_method(spikes_times, number_of_neurons=number_of_neurons, **kwargs).values())[0]
        rates = concatenate_heterogeneous_DataArrays(rates, "Neuron", name=name, fill_value=0.0)
        return {res_type: rates}

    def compute_spikes_correlation_coefficient(self, spikes, binsize=None, num_bins=None, **kwargs):
        pass

    def compute_spikes_covariance(self, spikes, binsize=None, num_bins=None, **kwargs):
        pass

    def _get_data_time(self, data):
        if data.size:
            return np.array(data.coords.get(data.dims[0], []))
        return np.array([])

    def _apply_transient_to_data(self, data):
        if self.transient and data.size:
            time = self._get_data_time(data)
            if time.size:
                t_start = self.assert_start_end_times_from_data_time(time)
                if t_start > time[0]:
                    return data.loc[t_start:]
        return data

    def compute_activity(self, data, name="", **kwargs):
        data = self._apply_transient_to_data(DataArray(data)).mean(dim="Time")
        if name:
            data.name = name
        return {self._get_comput_res_type(): data}

    def compute_mean_field(self, data, name="", **kwargs):
        data = self._apply_transient_to_data(DataArray(data)).mean(dim="Time")
        if data.ndim > 1:
            data = data.mean(dim="Neuron")
        if name:
            data.name = name
        return {self._get_comput_res_type(): data}

    def compute_total_activity(self, data, name="", **kwargs):
        data = self._apply_transient_to_data(DataArray(data)).mean(dim="Time")
        if data.ndim > 1:
            data = data.sum(dim="Neuron")
        if name:
            data.name = name
        return {self._get_comput_res_type(): data}

    def compute_activity_time_series(self, data, name="", **kwargs):
        data = self._apply_transient_to_data(DataArray(data))
        if name:
            data.name = name
        return {self._get_comput_res_type(): data}

    def compute_mean_field_time_series(self, data, name="", **kwargs):
        data = self._apply_transient_to_data(DataArray(data)).mean(dim="Neuron")
        if name:
            data.name = name
        return {self._get_comput_res_type(): data}

    def compute_total_activity_time_series(self, data, name="", **kwargs):
        data = self._apply_transient_to_data(DataArray(data)).sum(dim="Neuron")
        if name:
            data.name = name
        return {self._get_comput_res_type(): data}

    def prepare_results(self, results_names, computations_methods):
        results = OrderedDict()
        for i_res, computation_method in enumerate(computations_methods):
            res_name = self._get_comput_res_type(computation_method)
            try:
                results[res_name] = Series(name=results_names[i_res])
            except:
                results[res_name] = Series(name=res_name)
        return results

    def _get_safe_computation_kwargs(self, i_computation, computations_kwargs):
        try:
            computation_kwargs = computations_kwargs[i_computation]
        except:
            try:
                computation_kwargs = computations_kwargs[0]
            except:
                computation_kwargs = {}
        return computation_kwargs

    def regions_generator(self, reg_device_or_data, population_size=1,
                          computation_methods=[lambda x, **kwargs: x], computations_kwargs=[{}],
                          data_method=lambda x, **kwargs: x, data_kwargs={}):
        computation_methods = ensure_list(computation_methods)
        computations_kwargs = ensure_list(computations_kwargs)
        for reg_label, reg_data in reg_device_or_data.iteritems():
            data = data_method(reg_data, name=reg_label, **data_kwargs)
            population_size = data.get("number_of_neurons", population_size)
            outputs = OrderedDict()
            if self.return_data:
                outputs["data_name"] = data["data_name"]
                outputs[outputs["data_name"]] = data[data["data_name"]]
            for i_comput, computation_method in enumerate(computation_methods):
                this_data = data[data["data_name"]]
                if computation_method.__name__.find("rate") > -1:
                    this_data = outputs.get(self.spikes_train_name, this_data)
                outputs.update(computation_method(this_data, number_of_neurons=population_size, name=reg_label,
                                                  **self._get_safe_computation_kwargs(i_comput, computations_kwargs)))
            yield outputs

    def populations_generator(self, pop_device_or_data, populations_sizes=[],
                              computation_methods=[lambda x, **kwargs: x], computation_kwargs=[{}],
                              data_method=lambda x, **kwargs: x, data_kwargs={}):
        data_name = None
        for i_pop, (pop_label, pop_data) in enumerate(pop_device_or_data.iteritems()):
            pop_results = OrderedDict()
            try:
                population_size = populations_sizes[i_pop]
            except:
                population_size = 1
            for reg_results in \
                    self.regions_generator(pop_data, population_size,
                                           computation_methods, computation_kwargs,
                                           data_method, data_kwargs):
                data_name = reg_results.pop("data_name", data_name)
                if len(pop_results) == 0:
                    for res_type in reg_results.keys():
                        pop_results[res_type] = Series(name=pop_label)
                for res_type, res in reg_results.items():
                    pop_results[res_type][res.name] = res
            if self.force_homogeneous_results:
                for res_type, res in pop_results.items():
                    if isinstance(res[0], DataArray):
                        pop_results[res_type] = \
                            concatenate_heterogeneous_DataArrays(res, "Region",
                                                                 name=pop_label, fill_value=np.nan)
            if data_name is not None:
                pop_results["data_name"] = data_name
            yield pop_results

    def compute_results_from_population_generator(self, results, population_generator):
        data_name = None
        for pop_results in population_generator:
            data_name = pop_results.pop("data_name", data_name)
            for res_name in pop_results.keys():
                if res_name not in results.keys():
                    results[res_name] = Series(name=res_name)
                results[res_name][pop_results[res_name].name] = pop_results[res_name]
        if data_name is not None:
            results["data_name"] = data_name
        return results

    def convert_to_TVB_TimeSeries(self, results, concat_dim_name="Population", name=None,
                                  data_keys=None, fill_value=np.nan, transpose_dims=None, **kwargs):
        """This method converts the results' output pandas.Series or xarray.DataArray into a TVB TimeSeries instance.
           It will call the concatenate_heterogeneous_DataArrays function if the input is a pandas.Series.
           Arguments:
            - results: the pandas.Series of heterogeneous results to be combined
            - concat_dim_name: the label (string) of the dimension across which the concatenation should take place
            - name: the label (string) of the resulting xarray.DataArray output.
                    It defaults to the input results' pandas.Series name.
            - data_keys: a list of labels (strings) of the results' xarray.DataArray to be combined.
                         If None, it defaults to the keys of the results' pandas.Series index
            - fill_value: the value to fill missing data upon combination of
                          heterogeneous results' xarray.DataArray instances. Default = np.nan
            - transpose_dims: a list of labels (strings) corresponding to the desired ordering of dimensions
                              of the resulting results' xarray.DataArray output.
                              If None, the output will not be transposed.
            - **kwargs: other possible keyword arguments to the xarray.concat method.

        """
        if not isinstance(results, DataArray):
            results = \
                concatenate_heterogeneous_DataArrays(results, concat_dim_name, name=name, data_keys=data_keys,
                                                     fill_value=fill_value, transpose_dims=transpose_dims, **kwargs)

        while results.ndim > 4:
            results = results.mean(axis=-1)

        if self.connectivity is None:
            time_series = TimeSeries()
        else:
            time_series = TimeSeriesRegion(connectivity=self.connectivity)
        time_series.from_xarray_DataArray(results)
        return time_series

    def convert_results_to_output_type(self, results, dims_order):
        dims_order_no_population = list(dims_order)
        dims_order_no_population.remove("Population")
        for res_name, rates in results.items():
            if len(results[res_name]) and isinstance(results[res_name][0], DataArray):
                if self.force_homogeneous_results or self.output_type.upper().find("TVB"):
                    transpose_dims = \
                        get_ordered_dimensions(list(results[res_name][0].dims) + ["Population"], dims_order)
                    if self.output_type.upper() == "TVB" \
                            and results[res_name][0].ndim <= 3 \
                                and "Time" in results[res_name][0].dims:
                        results[res_name] = \
                            self.convert_to_TVB_TimeSeries(results[res_name], concat_dim_name="Population",
                                                           fill_value=np.nan, transpose_dims=transpose_dims)
                    else:
                        results[res_name] = \
                            concatenate_heterogeneous_DataArrays(results[res_name], concat_dim_name="Population",
                                                                 fill_value=np.nan, transpose_dims=transpose_dims)
                else:
                    for pop_label, pop_res in results[res_name].iteritems():
                        if isinstance(results[res_name][pop_label], DataArray):
                            transpose_dims = \
                                get_ordered_dimensions(list(results[res_name][pop_label][0].dims),
                                                       dims_order_no_population)
                            results[res_name][pop_label] = results[res_name][pop_label].transpose(*transpose_dims)
        return results

    def compute_spikes_measures(self, spikes_data, populations_sizes=[],
                                computations_methods=[], computations_kwargs=[], results_names=[],
                                data_method=lambda x, **kwargs: x, data_kwargs={},
                                return_spikes_trains=False):
        computations_methods = ensure_list(computations_methods)
        computations_kwargs = ensure_list(computations_kwargs)
        results_names = ensure_list(results_names)
        if len(computations_methods) == 0:
            computations_methods = [self.compute_mean_rate, self.compute_mean_rate_time_series]
            results_names = ["Mean Populations' Spikes' Rates",
                             "Mean Populations' Spikes' Rates' Time Series"]
        results = self.prepare_results(results_names, computations_methods)
        if return_spikes_trains:
            results[self.spikes_train_name] = Series(name="Spikes Trains")
        if len(spikes_data):
            results = \
                self.compute_results_from_population_generator(results,
                                                               self.populations_generator(spikes_data,
                                                                                          populations_sizes,
                                                                                          computations_methods,
                                                                                          computations_kwargs,
                                                                                          data_method,
                                                                                          data_kwargs)
                                                               )
            # TODO: find a more elegant solution to remove this entry from results' dict
            false_res_name = self._get_comput_res_type(self.compute_spikes_rates_by_neuron)
            if false_res_name in results.keys():
                del results[false_res_name]
            results = self.convert_results_to_output_type(results, ["Time", "Population", "Region", "Neuron"])
        return results

    def compute_spikes_correlations_from_spikes_trains(self, spikes_trains,
                                                       pop_labels, pop_reg_labels, all_regions_labels,
                                                       computations_methods=[], computations_kwargs=[],
                                                       bin_kwargs={}, results_names=[]):
        n_spikes_trains = len(spikes_trains)
        spikes_trains = self.compute_binned_spikes_trains(spikes_trains, **bin_kwargs)
        computations_methods = ensure_list(computations_methods)
        computations_kwargs = ensure_list(computations_kwargs)
        results_names = ensure_list(results_names)
        if len(computations_methods) == 0:
            computations_methods = [self.compute_spikes_correlation_coefficient]
            results_names = ["Populations' Correlation Coefficient"]
        if self.force_homogeneous_results:
            dims, coords = cross_dimensions_and_coordinates_MultiIndex(["Population", "Region"],
                                                                       pop_labels, all_regions_labels)
            n_cross_dims = len(pop_labels) * len(all_regions_labels)
            dummy = np.nan * np.ones((n_cross_dims, n_cross_dims))
        else:
            dummy = np.nan * np.ones((n_spikes_trains, n_spikes_trains))
            dims = ["Population_i-Region_i", "Population_j-Region_j"]
            coords = {dims[0]: pop_reg_labels, dims[1]: pop_reg_labels}
        results = OrderedDict()
        for i_comput, computation_method in enumerate(computations_methods):
            res_name = self._get_comput_res_type(computation_method)
            try:
                output_res_name = results_names[i_comput]
            except:
                output_res_name = res_name
            results[res_name] = DataArray(dummy, name=output_res_name, dims=dims, coords=coords)
            corr = computation_method(spikes_trains,
                                      self._get_safe_computation_kwargs(i_comput, computations_kwargs))
            spikes_trains = corr.get(self.binned_spikes_trains_name, spikes_trains)
            corr = corr[res_name]
            if self.force_homogeneous_results:
                results[res_name].loc[pop_reg_labels, pop_reg_labels] = corr
                results[res_name] = results[res_name].unstack(dims)
                temp_dims = list(results[res_name].dims)
                # Put population's variables in front of regions:
                results[res_name] = results[res_name].transpose(*tuple(temp_dims[0::2] + temp_dims[1::2]))
            else:
                results[res_name] = DataArray(corr, name=results[res_name].name, dims=dims, coords=coords)
        return results

    def compute_spikes_correlations(self, spikes_data, populations=[], regions=[],
                                    computations_methods=[], computations_kwargs=[], results_names=[],
                                    bin_kwargs={}, data_method=lambda x, **kwargs: x, data_kwargs={}):
        computations_methods = ensure_list(computations_methods)
        computations_kwargs = ensure_list(computations_kwargs)
        results_names = ensure_list(results_names)
        results = OrderedDict()
        pop_labels = []
        all_regions_labels = []
        pop_reg_labels = []
        spikes_trains = []
        if not populations:
            populations = spikes_data.keys()
        for pop_label in populations:
            pop_spikes = spikes_data[pop_label]
            pop_labels.append(pop_label)
            if self.return_data:
                data_to_return = Series(name=pop_label)
            for reg_label, reg_spikes in pop_spikes.iteritems():
                if not regions or reg_label in regions:
                    pop_reg_labels.append((pop_label, reg_label))
                    if reg_label not in all_regions_labels:
                        all_regions_labels.append(reg_label)
                    spikes = data_method(reg_spikes, name=reg_label, **data_kwargs)
                    data_name = spikes["data_name"]
                    spikes_trains.append(self._compute_spikes_train(spikes[data_name]))
                    if self.return_data:
                        data_to_return[reg_label] = spikes[data_name]
            if self.return_data:
                results["data_name"] = data_name
                if data_name not in results.keys():
                    results[data_name] = Series(name=data_name)
                results[data_name][pop_label] = data_to_return
        results.update(
            self.compute_spikes_correlations_from_spikes_trains(spikes_trains,
                                                                pop_labels, pop_reg_labels, all_regions_labels,
                                                                computations_methods, computations_kwargs,
                                                                bin_kwargs, results_names))
        return results

    def compute_spikes_rates_and_correlations(self, spikes_data, populations_sizes=[],
                                              rates_methods=[], rates_kwargs=[{}], rate_results_names=[],
                                              corrs_methods=[], corrs_kwargs=[{}], corrs_results_names=[],
                                              bin_kwargs={}, data_method=lambda x, **kwargs: x, data_kwargs={}):
        results = self.compute_spikes_measures(spikes_data, populations_sizes,
                                               rates_methods, rates_kwargs, rate_results_names,
                                               data_method, data_kwargs, return_spikes_trains=True)
        pop_labels = []
        all_regions_labels = []
        pop_reg_labels = []
        spikes_trains = []
        for pop_label, pop_spikes_trains in results[self.spikes_train_name].iteritems():
            pop_labels.append(pop_label)
            for reg_label, reg_spikes_trains in pop_spikes_trains.iteritems():
                pop_reg_labels.append((pop_label, reg_label))
                if reg_label not in all_regions_labels:
                    all_regions_labels.append(reg_label)
                spikes_trains.append(reg_spikes_trains)
        results.update(
            self.compute_spikes_correlations_from_spikes_trains(spikes_trains,
                                                                pop_labels, pop_reg_labels, all_regions_labels,
                                                                corrs_methods, corrs_kwargs,
                                                                bin_kwargs, corrs_results_names))
        return results

    def compute_continuous_data_measures(self, data, populations_sizes=[], variables=None,
                                         computations_methods=[], computations_kwargs=[], results_names=[],
                                         data_method=lambda x, **kwargs: x, data_kwargs={}):
        device_dims = data_kwargs.get("dims_names", ["Time", "Variable"])
        if variables:
            data_kwargs["variables"] = variables
        if len(computations_methods) == 0:
            computations_methods = [self.compute_mean_field_time_series]
            results_names = ["Populations' Mean Field Time Series"]
        results = self.prepare_results(results_names, computations_methods)
        if len(data):
            results = \
                self.compute_results_from_population_generator(results,
                                                               self.populations_generator(data,
                                                                                          populations_sizes,
                                                                                          computations_methods,
                                                                                          computations_kwargs,
                                                                                          data_method,
                                                                                          data_kwargs)
                                                               )
            results = self.convert_results_to_output_type(results, device_dims + ["Region", "Population", "Neuron"])
        return results

    def compute_spikeNet_rates(self, populations_devices=None, regions=None, populations_sizes=[],
                               computations_kwargs={}, data_method=None, data_kwargs={},
                               return_spikes_trains=False, return_devices=False):
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            if data_method is None:
                data_method = self.get_spikes_times_by_neuron_from_device
            computations_kwargs["rate_method"] = self.compute_rate
            results = self.compute_spikes_measures(spikes_devices, populations_sizes,
                                                   [self.compute_spikes_rates_by_neuron], [computations_kwargs],
                                                   ["Neurons' Spikes' Rates"],
                                                   data_method, data_kwargs, return_spikes_trains)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return (results, spikes_devices)
        else:
            return results

    def compute_spikeNet_rates_time_series(self, populations_devices=None, regions=None, populations_sizes=[],
                                           computations_kwargs={}, data_kwargs={}, data_method=None,
                                           return_spikes_trains=False, return_devices=False):
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            if data_method is None:
                data_method = self.get_spikes_times_by_neuron_from_device
            computations_kwargs["rate_method"] = self.compute_rate_time_series
            results = self.compute_spikes_measures(spikes_devices, populations_sizes,
                                                   [self.compute_spikes_rates_by_neuron], [computations_kwargs],
                                                   ["Neurons' Spikes' Rates' Time Series"],
                                                   data_method, data_kwargs, return_spikes_trains)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return (results, spikes_devices)
        else:
            return results

    def compute_spikeNet_mean_rates(self, populations_devices=None, regions=None, populations_sizes=[],
                                    computations_kwargs={}, data_method=None, data_kwargs={},
                                    return_spikes_trains=False, return_devices=False):
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            if data_method is None:
                data_method = self.get_spikes_times_from_device
            results = self.compute_spikes_measures(spikes_devices, populations_sizes,
                                                   [self.compute_mean_rate], [computations_kwargs],
                                                   ["Mean Populations' Spikes' Rates"],
                                                   data_method, data_kwargs, return_spikes_trains)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return (results, spikes_devices)
        else:
            return results

    def compute_spikeNet_mean_rates_time_series(self, populations_devices=None, regions=None, populations_sizes=[],
                                                computations_kwargs={}, data_kwargs={}, data_method=None,
                                                return_spikes_trains=False, return_devices=False):
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            if data_method is None:
                data_method = self.get_spikes_times_from_device
            results = self.compute_spikes_measures(spikes_devices, populations_sizes,
                                                   [self.compute_mean_instant_rate], [computations_kwargs],
                                                   ["Mean Populations' Spikes' Rates' Time Series"],
                                                   data_method, data_kwargs, return_spikes_trains)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return (results, spikes_devices)
        else:
            return results

    def compute_spikeNet_total_rates(self, populations_devices=None, regions=None, populations_sizes=[],
                                     computations_kwargs={}, data_kwargs={}, data_method=None,
                                     return_spikes_trains=False, return_devices=False):
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            if data_method is None:
                data_method = self.get_spikes_times_from_device
            results = self.compute_spikes_measures(spikes_devices, populations_sizes,
                                                   [self.compute_total_rate], [computations_kwargs],
                                                   ["Total Populations' Spikes' Rates"],
                                                   data_method, data_kwargs, return_spikes_trains)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return (results, spikes_devices)
        else:
            return results

    def compute_spikeNet_total_rates_time_series(self, populations_devices=None, regions=None, populations_sizes=[],
                                                 computations_kwargs={}, data_kwargs={}, data_method=None,
                                                 return_spikes_trains=False, return_devices=False):
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            if data_method is None:
                data_method = self.get_spikes_times_from_device
            results = self.compute_spikes_measures(spikes_devices, populations_sizes,
                                                   [self.compute_total_instant_rate], [computations_kwargs],
                                                   ["Total Populations' Spikes' Rates' Time Series"],
                                                   data_method, data_kwargs,
                                                   return_spikes_trains)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return (results, spikes_devices)
        else:
            return results

    def compute_spikeNet_spikes_correlations(self, populations_devices=None, regions=None,
                                             computations_methods=[], computations_kwargs=[{}], results_names=[],
                                             bin_kwargs={}, data_method=None, data_kwargs={}, return_devices=False):
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            if data_method is None:
                data_method = self.get_spikes_times_from_device
            results = self.compute_spikes_correlations(spikes_devices, computations_methods=computations_methods,
                                                       computations_kwargs=computations_kwargs,
                                                       results_names=results_names, bin_kwargs=bin_kwargs,
                                                       data_method=data_method, data_kwargs=data_kwargs)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return (results, spikes_devices)
        else:
            return results

    def compute_spikeNet_spikes_rates_and_correlations(self, populations_devices=None, regions=None,
                                                       populations_sizes=[],
                                                       rates_methods=[], rates_kwargs=[{}], rate_results_names=[],
                                                       corrs_methods=[], corrs_kwargs=[{}], corrs_results_names=[],
                                                       bin_kwargs={}, data_method=None, data_kwargs={},
                                                       return_devices=False):
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            if data_method is None:
                data_method = self.get_spikes_times_from_device
            results = self.compute_spikes_rates_and_correlations(spikes_devices, populations_sizes,
                                                                 rates_methods, rates_kwargs, rate_results_names,
                                                                 corrs_methods, corrs_kwargs, corrs_results_names,
                                                                 bin_kwargs, data_method, data_kwargs)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return (results, spikes_devices)
        else:
            return results

    def compute_spikeNet_mean_field(self, populations_devices=None, regions=None,
                                    variables=None, populations_sizes=[],
                                    computations_kwargs={}, data_kwargs={}, return_devices=False):
        devices = self.spikeNet.get_continuous_time_devices(regions=regions, populations_devices=populations_devices)
        if len(devices):
            results = self.compute_continuous_data_measures(devices, populations_sizes, variables,
                                                            [self.compute_mean_field], [computations_kwargs],
                                                            ["Populations' Mean Field"],
                                                            self.get_data_by_neuron_from_device, data_kwargs)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return (results, devices)
        else:
            return results

    def compute_spikeNet_total_activity(self, populations_devices=None, regions=None,
                                        variables=None, populations_sizes=[],
                                        computations_kwargs={}, data_kwargs={}, return_devices=False):
        devices = self.spikeNet.get_continuous_time_devices(regions=regions, populations_devices=populations_devices)
        if len(devices):
            results = self.compute_continuous_data_measures(devices, populations_sizes, variables,
                                                            [self.compute_total_activity], [computations_kwargs],
                                                            ["Populations' Total Activity"],
                                                            self.get_data_by_neuron_from_device, data_kwargs)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return (results, devices)
        else:
            return results

    def compute_spikeNet_mean_field_time_series(self, populations_devices=None, regions=None,
                                                variables=None, populations_sizes=[],
                                                computations_kwargs={}, data_kwargs={}, return_devices=False):
        devices = self.spikeNet.get_continuous_time_devices(regions=regions, populations_devices=populations_devices)
        if len(devices):
            results = self.compute_continuous_data_measures(devices, populations_sizes, variables,
                                                           [self.compute_mean_field_time_series], [computations_kwargs],
                                                           ["Populations' Mean Field Time Series"],
                                                           self.get_data_by_neuron_from_device, data_kwargs)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return (results, devices)
        else:
            return results

    def compute_spikeNet_total_activity_time_series(self, populations_devices=None, regions=None,
                                                    variables=None, populations_sizes=[],
                                                    computations_kwargs={}, data_kwargs={}, return_devices=False):
        devices = self.spikeNet.get_continuous_time_devices(regions=regions, populations_devices=populations_devices)
        if len(devices):
            results = self.compute_continuous_data_measures(devices, populations_sizes, variables,
                                                            [self.compute_total_activity_time_serie],
                                                            [computations_kwargs],
                                                            ["Populations' Total Activity Time Series"],
                                                            self.get_data_by_neuron_from_device, data_kwargs)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return (results, devices)
        else:
            return results
