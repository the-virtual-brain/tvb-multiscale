# -*- coding: utf-8 -*-

from six import string_types
from collections import OrderedDict

import numpy as np
from pandas import Series
from xarray import DataArray

from tvb_multiscale.core.data_analysis.spiking_network_analyser_base import \
    SpikingNetworkAnalyserBase, _get_safely_list_item
from tvb_multiscale.core.utils.data_structures_utils import \
    cross_dimensions_and_coordinates_MultiIndex, get_ordered_dimensions

from tvb.contrib.scripts.utils.data_structures_utils import \
    ensure_list, concatenate_heterogeneous_DataArrays
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeries, TimeSeriesRegion


class SpikingNetworkAnalyser(SpikingNetworkAnalyserBase):

    elephant_analyser = None
    pyspike_analyser = None

    def __init__(self, elephant=True, pyspike=True, **kwargs):
        super(SpikingNetworkAnalyser, self).__init__(**kwargs)
        self.configure()
        if elephant:
            from tvb_multiscale.tvb_elephant.spiking_network_analyser \
                import SpikingNetworkAnalyser as SpikingNetworkAnalyserElephant
            self.elephant_analyser = SpikingNetworkAnalyserElephant().from_instance(self, **kwargs)
        if pyspike:
            from tvb_multiscale.tvb_pyspike.spiking_network_analyser \
                import SpikingNetworkAnalyser as SpikingNetworkAnalyserPySpike
            self.pyspike_analyser = SpikingNetworkAnalyserPySpike().from_instance(self, **kwargs)

    def __getattr__(self, attr):
        if hasattr(SpikingNetworkAnalyser, attr):
            return SpikingNetworkAnalyserBase.__getattr__(self, attr)
        else:
            try:
                from tvb_multiscale.tvb_elephant.spiking_network_analyser \
                    import SpikingNetworkAnalyser as SpikingNetworkAnalyserElephant
                assert hasattr(SpikingNetworkAnalyserElephant, attr)
                if not isinstance(self.elephant_analyser, SpikingNetworkAnalyserElephant):
                    self.elephant_analyser = SpikingNetworkAnalyserElephant().from_instance(self)
                return getattr(self.elephant_analyser, attr)
            except:
                from tvb_multiscale.tvb_pyspike.spiking_network_analyser \
                    import SpikingNetworkAnalyser as SpikingNetworkAnalyserPySpike
                assert hasattr(SpikingNetworkAnalyserPySpike, attr)
                if not isinstance(self.pyspike_analyser, SpikingNetworkAnalyserPySpike):
                    self.pyspike_analyser = SpikingNetworkAnalyserPySpike().from_instance(self)
                return getattr(self.pyspike_analyser, attr)

    def _setattr_to_elephant_analyser(self, attr, val):
        try:
            from tvb_multiscale.tvb_elephant.spiking_network_analyser \
                import SpikingNetworkAnalyser as SpikingNetworkAnalyserElephant
        except:
            return
        if isinstance(self.elephant_analyser, SpikingNetworkAnalyserElephant):
            setattr(self.elephant_analyser, attr, val)

    def _setattr_to_pyspike_analyser(self, attr, val):
        try:
            from tvb_multiscale.tvb_pyspike.spiking_network_analyser \
                import SpikingNetworkAnalyser as SpikingNetworkAnalyserPySpike
        except:
            return
        if isinstance(self.pyspike_analyser, SpikingNetworkAnalyserPySpike):
            setattr(self.pyspike_analyser, attr, val)

    def __setattr__(self, attr, val):
        if hasattr(SpikingNetworkAnalyser, attr):
            # If it is a common attribute:
            SpikingNetworkAnalyserBase.__setattr__(self, attr, val)
            if attr != "gid":
                self._setattr_to_elephant_analyser(attr, val)
                self._setattr_to_pyspike_analyser(attr, val)
            return
        else:
            try:
                from tvb_multiscale.tvb_elephant.spiking_network_analyser \
                    import SpikingNetworkAnalyser as SpikingNetworkAnalyserElephant
                assert hasattr(SpikingNetworkAnalyserElephant, attr)
                if not isinstance(self.elephant_analyser, SpikingNetworkAnalyserElephant):
                    self.elephant_analyser = SpikingNetworkAnalyserElephant().from_instance(self)
                setattr(self.elephant_analyser, attr, val)
                return
            except:
                from tvb_multiscale.tvb_pyspike.spiking_network_analyser \
                    import SpikingNetworkAnalyser as SpikingNetworkAnalyserPySpike
                if hasattr(SpikingNetworkAnalyserPySpike, attr):
                    if not isinstance(self.pyspike_analyser, SpikingNetworkAnalyserPySpike):
                        self.pyspike_analyser = SpikingNetworkAnalyserPySpike().from_instance(self)
                    setattr(self.pyspike_analyser, attr, val)
                    return
        # Add a new common attribute:
        SpikingNetworkAnalyserBase.__setattr__(self, attr, val)
        if attr != "gid":
            self._setattr_to_elephant_analyser(attr, val)
            self._setattr_to_pyspike_analyser(attr, val)

    def _regions_generator(self, reg_device_or_data, population_sizes=[],
                           computation_methods=[lambda x, **kwargs: x], computations_kwargs=[{}],
                           data_method=lambda x, **kwargs: x, data_kwargs={}):
        """A generator method to loop over the regions devices of a populations' DeviceSet or the respective data,
           in order to apply a list of computations methods.
           Arguments:
            - reg_device_or_data: a region recording Device or a dictionary of data read from such a device
            - population_sizes=[]: the number of neurons of this population, or a list of such numbers, one per region.
                                  If reg_device_or_data is a Device, it will be overwritten by the data loading method.
            - computations_methods: a list of SpikingNetworkAnalyserBase class' "compute_" methods,
                                    to perform the desired computations.
                                    Default: [lambda x, **kwargs: x],
                                    i.e., comprising a method that just returns the input data.
            - computations_kwargs: a list of keyword arguments dictionaries.
                                   It len(computations_kwargs) >= len(computations_methods) to be computed,
                                   each method to be computed will get its own kwargs dict.
                                   If len(computations_kwargs) > 0 but < len(computations_methods) to be computed,
                                   all methods will get computations_kwargs[0] kwargs dict.
                                   If len(computations_kwargs) == 0,
                                   an empty kwargs dict is passed to all computations' methods
            - data_method: a SpikingNetworkAnalyserBase class' "get_" method
                           to load data from recording Device or a file (TODO).
                           Default = lambda x, **kwargs, assuming that reg_device_or_data already comprises of the data.
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data_method.

        """
        computation_methods = ensure_list(computation_methods)
        computations_kwargs = ensure_list(computations_kwargs)
        population_sizes = ensure_list(population_sizes)
        # For every region Device or the respective data stored in a pandas.Series...
        for i_reg, (reg_label, reg_data) in enumerate(reg_device_or_data.items()):
            # ...load a dictionary of data, its name, and optionally the number of neurons...
            data = data_method(reg_data, **data_kwargs)
            # ...get the number of neurons...
            population_size = data.get("number_of_neurons",
                                       self._get_safely_population_size(i_reg, population_sizes))
            # ...initialize an outputs' OrderedDict...
            outputs = OrderedDict()
            # ...we have to return the data as well...
            if self.return_data:
                # ...populate the outputs' dict with the name of the data...
                outputs["data_name"] = data["data_name"]
                # ...and the data itself...
                outputs[outputs["data_name"]] = data[data["data_name"]]
            # ...now loop for every computation method to be applied...:
            for i_comput, computation_method in enumerate(computation_methods):
                # ...set the data for this computation...
                this_data = data[data["data_name"]]
                # ...but this is a method applied on spikes...
                if computation_method.__name__.find("rate") > -1:
                    # ...check whether a Spikes Train (Elephant/neo) has already being computed,
                    # to not repeat this computation...
                    this_data = outputs.get(self.spikes_train_name, this_data)
                if computation_method.__name__.find("spikes_co") > -1:
                    # ...check whether a BinnedSpikesTrain (Elephant/neo) has already being computed,
                    # to not repeat this computation...
                    this_data = outputs.get(self.binned_spikes_trains_name, this_data)
                elif computation_method.__name__.find("spikes") > -1:
                    # ...check whether a collection of SpikesTrain (PySpike) has already being computed,
                    # to not repeat this computation...
                    this_data = outputs.get(self.spikes_trains_name, this_data)
                # ...and eventually call the computation and collect the results:
                outputs.update(computation_method(this_data, number_of_neurons=population_size,
                                                  **self._get_safely_computation_kwargs(i_comput, computations_kwargs)))
            # ...finally yield the region label and the the outputs dictionary:
            yield reg_label, outputs

    def _populations_generator(self, pop_device_or_data, populations_sizes=[],
                               computation_methods=[lambda x, **kwargs: x], computation_kwargs=[{}],
                               data_method=lambda x, **kwargs: x, data_kwargs={}):
        """A generator method to loop over a set of populations' DeviceSets or the respective data,
           in order to apply a list of computations methods.
           Arguments:
            - pop_device_or_data: a populations' DeviceSet or a Series of dictionaries of data read from such devices
            - population_sizes=[]: list of numbers of neurons per population,
                                  If pop_device_or_data is a DeviceSet,
                                  it will be overwritten by the data loading method.
            - computations_methods: a list of SpikingNetworkAnalyserBase class' "compute_" methods,
                                    to perform the desired computations.
                                    Default: [lambda x, **kwargs: x],
                                    i.e., comprising a method that just returns the input data.
            - computations_kwargs: a list of keyword arguments dictionaries.
                                   It len(computations_kwargs) >= len(computations_methods) to be computed,
                                   each method to be computed will get its own kwargs dict.
                                   If len(computations_kwargs) > 0 but < len(computations_methods) to be computed,
                                   all methods will get computations_kwargs[0] kwargs dict.
                                   If len(computations_kwargs) == 0,
                                   an empty kwargs dict is passed to all computations' methods
            - data_method: a SpikingNetworkAnalyserBase class' "get_" method
                           to load data from recording Device or a file (TODO).
                           Default = lambda x, **kwargs,
                           assuming that pop_device_or_data already comprises of loaded data.
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data_method.

        """
        # Initialize the name of the data for this set of computations:
        data_name = None
        populations_sizes = ensure_list(populations_sizes)
        # For every populations' DeviceSet or the respective data stored in a pandas.Series:
        for i_pop, (pop_label, pop_data) in enumerate(pop_device_or_data.items()):
            # ...initialize the results' OrderedDict...
            pop_results = OrderedDict()
            # ...for every region Device or data...
            for reg_label, reg_results in \
                    self._regions_generator(pop_data, _get_safely_list_item(i_pop, populations_sizes, 1),
                                            computation_methods, computation_kwargs,
                                            data_method, data_kwargs):
                # ...get the data_name if it exists....
                data_name = reg_results.pop("data_name", data_name)
                # ...if this is the first population's DeviceSet...
                if len(pop_results) == 0:
                    # ...initialize the results' dictionary
                    for res_type in reg_results.keys():
                        pop_results[res_type] = Series(name=pop_label, dtype='object')
                # ...and loop to get every region wise result of computation...
                for res_type, res in reg_results.items():
                    pop_results[res_type][reg_label] = res
            # ...if we need to return xarray.DataArray instances of homogeneous results...
            if self.force_homogeneous_results:
                # ...loop for every result...
                for res_type, res in pop_results.items():
                    # ...and if it is a pandas.Series of xarray.DataArray instances...
                    if isinstance(res[0], DataArray):
                        # ...concatenate them:
                        pop_results[res_type] = \
                            concatenate_heterogeneous_DataArrays(res, "Region",
                                                                 name=pop_label, fill_value=np.nan)
            if data_name is not None:
                pop_results["data_name"] = data_name
            # Finally, yield the population's label and results:
            yield pop_label, pop_results

    def _compute_results_from_population_generator(self, results, population_generator):
        """Method to compute and collect the results of the computations performed
           by the _populations_generator() method
           Arguments:
            - results: a collections.OrderedDict to be populated with results
            - populations_generator: the population generator
           Returns:
            the populated results
        """
        data_name = None
        # For every population's DeviceSet
        for pop_label, pop_results in population_generator:
            data_name = pop_results.pop("data_name", data_name)
            # ...and for every result...
            for res_name in pop_results.keys():
                # ...initialize if this is a new result...
                if res_name not in results.keys():
                    results[res_name] = Series(name=res_name, dtype='object')
                # ...get the result...
                results[res_name][pop_label] = pop_results[res_name]
        if data_name is not None:
            results["data_name"] = data_name
        return results

    def _prepare_results(self, results_names, computations_methods):
        """A method to prepare the results OrderedDict
           based on the user defined results_names list and
           the list of computations' methods to be performed.
           Arguments:
            -results_names: a list of desired names (strings) for the results of the computations,
                            to be given to the results' pandas.Series or xarray.DataArray instances.
            -computations_methods: the methods to be computed
           Returns:
            an OrderedDict with key value pairs as follows:
            "res_name" (taken from the computation method name):
                an empty pandas.Series with a name taken either from the input results_names or same as res_name
        """
        results = OrderedDict()
        for i_res, computation_method in enumerate(computations_methods):
            res_name = self._get_comput_res_type(computation_method)
            try:
                results[res_name] = Series(name=results_names[i_res], dtype='object')
            except:
                results[res_name] = Series(name=res_name, dtype='object')
        return results

    def _get_safely_computation_kwargs(self, i_computation, computations_kwargs):
        """A method to safely get computations specific keyword arguments, no matter if the user has provided them.
           Arguments:
            - i_computation: increasing index (integer) of the method to be computed
            - computations_kwargs: a list of keyword arguments dictionaries.
                                   It len(computations_kwargs) >= number of methods to be computed,
                                   each method to be computed will get its own kwargs dict.
                                   If len(computations_kwargs) > 0 but < number of methods to be computed,
                                   all methods will get computations_kwargs[0] kwargs dict.
                                   If len(computations_kwargs) == 0,
                                   an empty kwargs dict is passed to all computations' methods
           Returns:
            - computation_kwargs: a keyword arguments' dictionary
        """
        return _get_safely_list_item(i_computation, computations_kwargs, {})

    def _get_safely_population_size(self, i_pop_or_reg, populations_sizes):
        """A method to safely get population size, no matter if the user has provided them.
           Arguments:
            - i_pop_or_reg: increasing index (integer) of the population or region
            - populations_sizes: a list of populations' sizes
                                   It len(populations_sizes) >= number of populations,
                                   each population gets its own size.
                                   If len(populations_sizes) > 0 but < number of populations ,
                                   all populations will get populations_sizes[0].
                                   If len(populations_sizes) == 0,
                                   all populations will get the size = 1
           Returns:
            - populations_size: a number (integer) or a list of numbers, one per region
        """
        return _get_safely_list_item(i_pop_or_reg, populations_sizes, 1)

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
        # If the input results is still a pandas.Series:
        if not isinstance(results, DataArray):
            # ...concatenate to get a homogeneous xarray.DataArray instance:
            results = \
                concatenate_heterogeneous_DataArrays(results, concat_dim_name, name=name, data_keys=data_keys,
                                                     fill_value=fill_value, transpose_dims=transpose_dims, **kwargs)

        # ...while there are more than 4 dimensions in the data...
        while results.ndim > 4:
            # ...keep averaging the last dimension...
            results = results.mean(axis=-1)

        # ...if we have a connectivity...
        if self.connectivity:
            # ...create a TVB TimeSeriesRegion instance from the results' xarray.DataArray instance......
            time_series = TimeSeriesRegion(data=results, connectivity=self.connectivity)
        else:
            # ...create a TVB TimeSeries instance from the results' xarray.DataArray instance......
            time_series = TimeSeries(results)
        time_series.configure()

        # ...and return it...
        return time_series

    def convert_results_to_output_type(self, results, dims_order):
        """This method converts the results' output pandas.Series of xarray.DataArray instances to
           a homogeneous xarray.DataArray via concatenation or even to a TVB TimeSeries instance for time series data,
           according to the flag force_homogeneous_results and the time_series_output_type attribute.
           Arguments:
            - results: collections.OrderedDict of all populations' Devices' results pandas.Series
            - dims_order: the desired order of dimensions for the output arrays
           Returns:
            - the results collections.OrderedDict with the results' data types modified.
        """
        dims_order_no_population = list(dims_order)  # This is for homogeneous results in xarray.DataArray
        dims_order_no_population.remove("Population")  # This is for heterogeneous results in pandas.Series
        # For every results...
        for res_name, result in results.items():
            homogeneous = False
            # ...if it is not empty and if it consists of xarray.DataArray instances...
            if len(result) and isinstance(result[0], DataArray):
                # these are the desired dimensions for a homogeneous result:
                transpose_dims = \
                    get_ordered_dimensions(list(result[0].dims) + ["Population"], dims_order)
                # ...if the output type for time series data is TVB TimeSeries instances...
                if self.time_series_output_type.upper() == "TVB" \
                        and result[0].ndim <= 3 \
                            and "Time" in result[0].dims:
                    # ...put them in a TVB TimeSeries instance...:
                    results[res_name] = \
                        self.convert_to_TVB_TimeSeries(result, concat_dim_name="Population",
                                                       fill_value=np.nan, transpose_dims=transpose_dims)
                    homogeneous = True
                # ...or if we require homogeneous results...
                elif self.force_homogeneous_results:
                    # ...concatenate the results of the different populations...
                    results[res_name] = \
                        concatenate_heterogeneous_DataArrays(result, concat_dim_name="Population",
                                                             fill_value=np.nan, transpose_dims=transpose_dims)
                    homogeneous = True
                if not homogeneous:
                    # If the results are to be pandas.Series...
                    for pop_label, pop_res in result.items():
                        # ...of heterogeneous xarray.DataArray instances...
                        if isinstance(pop_res, DataArray):
                            # ...transpose the dimensions -without Population- to the desired order:
                            transpose_dims = \
                                get_ordered_dimensions(list(pop_res[0].dims), dims_order_no_population)
                            results[res_name][pop_label] = pop_res.transpose(*transpose_dims)
        return results

    def _compute_spikes_correlations_from_spikes_trains(self, spikes_trains,
                                                        pop_labels, pop_reg_labels, all_regions_labels,
                                                        computations_methods=[], computations_kwargs=[],
                                                        bin_kwargs={}, results_names=[]):
        """A method to compute several spikes' correlations' measures on the a list of Spikes Trains.
           Arguments:
            - spikes_trains: a list of Spikes Trains.
            - population_sizes=[]: list of numbers of neurons per region in such a population,
                                  If pop_device_or_data is a DeviceSet,
                                  it will be overwritten by the data loading method.
            - pop_labels: a list of all the populations' labels
            - pop_reg_labels: a list of tuples (Population_i, Region_j) one for each combination
            - all_reg_labels: a list of all regions' labels
            - computations_methods: a list of SpikingNetworkAnalyserBase class' "compute_" methods,
                                    to perform the desired computations.
                                    Default: [lambda x, **kwargs: x],
                                    i.e., comprising a method that just returns the input data.
            - computations_kwargs: a list of keyword arguments dictionaries.
                                   It len(computations_kwargs) >= len(computations_methods) to be computed,
                                   each method to be computed will get its own kwargs dict.
                                   If len(computations_kwargs) > 0 but < len(computations_methods) to be computed,
                                   all methods will get computations_kwargs[0] kwargs dict.
                                   If len(computations_kwargs) == 0,
                                   an empty kwargs dict is passed to all computations' methods
            - bin_kwargs={}: a dictionary of potential keyword arguments to be passed
                             to the method computing Binned Spikes' Trains.
            - results_names=[]: a list of names (strings) to be given to the pandas.Series or xarray.DataArray instances
                                of the results of each measure.
        """
        n_spikes_trains = len(spikes_trains)
        spikes_trains = self.compute_binned_spikes_trains(spikes_trains, **bin_kwargs)
        computations_methods = ensure_list(computations_methods)
        computations_kwargs = ensure_list(computations_kwargs)
        results_names = ensure_list(results_names)
        if len(computations_methods) == 0:
            # This is the default method to be computed and its default name:
            computations_methods = [self.compute_spikes_correlation_coefficient]
            results_names = ["Populations' Correlation Coefficient"]
        # Prepare the results...
        if self.force_homogeneous_results:
            # ...either in a (Population_i, Population_j, Region_i, Region_j) xarray.DataArray:
            dims, coords = cross_dimensions_and_coordinates_MultiIndex(["Population", "Region"],
                                                                       pop_labels, all_regions_labels)
            n_cross_dims = len(pop_labels) * len(all_regions_labels)
            dummy = np.nan * np.ones((n_cross_dims, n_cross_dims))
        else:
            # ...or in a (Population_i-Region_i, Population_j-Region_j) xarray.DataArray (product pandas.MultiIndex):
            dummy = np.nan * np.ones((n_spikes_trains, n_spikes_trains))
            dims = ["Population_i-Region_i", "Population_j-Region_j"]
            coords = {dims[0]: pop_reg_labels, dims[1]: pop_reg_labels}
        results = OrderedDict()
        # Loop over all the different correlation computations' methods...:
        for i_comput, computation_method in enumerate(computations_methods):
            # ...prepare the DataArray:
            res_name = self._get_comput_res_type(computation_method)
            try:
                output_res_name = results_names[i_comput]
            except:
                output_res_name = res_name
            results[res_name] = DataArray(dummy, name=output_res_name, dims=dims, coords=coords)
            # ...compute....
            corr = computation_method(spikes_trains,
                                      self._get_safely_computation_kwargs(i_comput, computations_kwargs))
            # ...unpack the results...
            spikes_trains = corr.get(self.binned_spikes_trains_name, spikes_trains)
            corr = corr[res_name]
            if self.force_homogeneous_results:
                # ...reshape the result for the case of a (Population_i, Population_j, Region_i, Region_j) output:
                results[res_name].loc[pop_reg_labels, pop_reg_labels] = corr
                # ...unstack the product pandas.MultiIndex:
                results[res_name] = results[res_name].unstack(dims)
                temp_dims = list(results[res_name].dims)
                # ...put population's variables in front of regions:
                results[res_name] = results[res_name].transpose(*tuple(temp_dims[0::2] + temp_dims[1::2]))
            else:
                results[res_name] = DataArray(corr, name=results[res_name].name, dims=dims, coords=coords)
        return results

    def compute_rate(self, spikes, number_of_neurons=1, duration=None, **kwargs):
        """A method to compute rate from an input of spikes' events or spikes' times as:
           number_of_spikes/total_duration * 1000 in Hz.
           Arguments:
            - spikes: a Spike Train or
                      an array of spikes' times or a dict with a key-value pair of "times" and spikes' times array
            - number_of_neurons=1: the number (integer) of neurons
            - duration: Default=None, in which case it is computed by start_time and end_time
            - kwargs
           Returns:
            - a dictionary of the following key-value pair:
             "rate": xarray.DataArray(rate).squeeze()
        """
        res_type = self._get_comput_res_type()
        spikes_times = np.array(self._get_spikes_times_from_spikes_events(spikes))
        if self.elephant_analyser is not None:
            return self.elephant_analyser.compute_rate(spikes, number_of_neurons, duration, **kwargs)
        return self._compute_rate_base(spikes_times, res_type, number_of_neurons, duration, **kwargs)

    def compute_spikes_measures(self, spikes_data, populations_sizes=[],
                                computations_methods=[], computations_kwargs=[], results_names=[],
                                data_method=lambda x, **kwargs: x, data_kwargs={},
                                return_spikes_trains=False):
        """A method to compute several spikes' measures together on the same spikes' data.
           Arguments:
            - spikes_data: a pandas.Series of spikes' recording devices or of data loaded from such devices.
            - population_sizes=[]: list of numbers of neurons per region in such a population,
                                  If pop_device_or_data is a DeviceSet,
                                  it will be overwritten by the data loading method.
            - computations_methods: a list of SpikingNetworkAnalyserBase class' "compute_" methods,
                                    to perform the desired computations.
                                    Default: [lambda x, **kwargs: x],
                                    i.e., comprising a method that just returns the input data.
            - computations_kwargs: a list of keyword arguments dictionaries.
                                   It len(computations_kwargs) >= len(computations_methods) to be computed,
                                   each method to be computed will get its own kwargs dict.
                                   If len(computations_kwargs) > 0 but < len(computations_methods) to be computed,
                                   all methods will get computations_kwargs[0] kwargs dict.
                                   If len(computations_kwargs) == 0,
                                   an empty kwargs dict is passed to all computations' methods
            - data_method: a SpikingNetworkAnalyserBase class' "get_" method
                           to load data from recording Device or a file (TODO).
                           Default = lambda x, **kwargs, assuming that spikes_data already comprises of loadedd data.
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data_method.
            - return_spikes_trains=False: flag to return the Spikes Trains used in the computations or not,
                                          if it is a string, it will be the name of the returned spikes' train result,
                                          default name is taken from spikes_train_name attribute
        """
        computations_methods = ensure_list(computations_methods)
        computations_kwargs = ensure_list(computations_kwargs)
        results_names = ensure_list(results_names)
        if len(computations_methods) == 0:
            computations_methods = [self.compute_mean_rate, self.compute_mean_rate_time_series]
            results_names = ["Mean Populations' Spikes' Rates",
                             "Mean Populations' Spikes' Rates' Time Series"]
        # Prepare the results dictionary:
        results = self._prepare_results(results_names, computations_methods)
        if return_spikes_trains:
            if not isinstance(return_spikes_trains, string_types):
                return_spikes_trains = self.spikes_train_name
            # Add the spikes trains to the results dicts if required:
            results[return_spikes_trains] = Series(name="Spikes Trains", dtype='object')
        if len(spikes_data):
            # If there are any spikes" data or recording devices, proceed to computation and collect the results...
            results = \
                self._compute_results_from_population_generator(results,
                                                                self._populations_generator(spikes_data,
                                                                                            populations_sizes,
                                                                                            computations_methods,
                                                                                            computations_kwargs,
                                                                                            data_method,
                                                                                            data_kwargs)
                                                                )
            # Remove the spikes_rates_by_neuron false results name for the per neuron computations:
            # TODO: find a more elegant solution to remove this entry from results' dict
            false_res_name = self._get_comput_res_type(self.compute_spikes_rates_by_neuron)
            if false_res_name in results.keys():
                del results[false_res_name]
            # Convert the results to the desired output type
            results = self.convert_results_to_output_type(results, ["Time", "Population", "Region", "Neuron"])
        return results

    def compute_spikes_correlations(self, spikes_data, populations=[], regions=[],
                                    computations_methods=[], computations_kwargs=[], results_names=[],
                                    bin_kwargs={}, data_method=lambda x, **kwargs: x, data_kwargs={}):
        """A method to compute several spikes' correlations' measures
           on the same spikes' data or recording Devices.
           Arguments:
            - spikes_data: Spike recording populations' DeviceSets or data loaded from them in a pandas.Series.
            - population[]: list of populations' labels to be included in the computations. By default all of them.
            - regions[]: list of regions' labels to be included in the computations. By default all of them.
            - corrs_methods: a list of SpikingNetworkAnalyserBase class' "compute_" methods,
                             to perform the desired correlations' computations.
                             It defaults to computing correlation_coefficient.
            - corrs_kwargs: a list of keyword arguments dictionaries.
                            It len(corrs_kwargs) >= len(corrs_methods) to be computed,
                            each method to be computed will get its own kwargs dict.
                            If len(corrs_kwargs) > 0 but < len(corrs_methods) to be computed,
                            all methods will get corrs_kwargs[0] kwargs dict.
                            If len(corrs_kwargs) == 0,
                            an empty kwargs dict is passed to all correlation computations' methods
            - results_names=[]: a list of names (strings) to be given to the pandas.Series or xarray.DataArray instances
                                of the results of each measure.
            - bin_kwargs={}: a dictionary of potential keyword arguments to be passed
                             to the method computing Binned Spikes' Trains.
            - data_method: a SpikingNetworkAnalyserBase class' "get_" method
                           to load data from recording Device or a file (TODO).
                           Default = lambda x, **kwargs, assuming that spikes_data already comprises of loaded data.
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data_method.

        """
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
        # For every population (selected)...
        for pop_label in populations:
            # ...get the label and DeviceSet or data pandas.Series...
            pop_spikes = spikes_data[pop_label]
            pop_labels.append(pop_label)
            if self.return_data:
                # ...if we also return the data loaded...:
                data_to_return = Series(name=pop_label, dtype='object')
            # ...and for every region Device or data...
            for reg_label, reg_spikes in pop_spikes.items():
                if not regions or reg_label in regions:
                    # ...if no selection of regions is required, or if this region is selected...
                    pop_reg_labels.append((pop_label, reg_label))
                    if reg_label not in all_regions_labels:
                        # ...and the region label
                        all_regions_labels.append(reg_label)
                    # ...load the spikes' data...
                    spikes = data_method(reg_spikes, **data_kwargs)
                    data_name = spikes["data_name"]
                    # ...compute the respective Spikes Train...
                    spikes_trains.append(self._compute_spikes_train(spikes[data_name]))
                    if self.return_data:
                        data_to_return[reg_label] = spikes[data_name]
            if self.return_data:
                results["data_name"] = data_name
                if data_name not in results.keys():
                    results[data_name] = Series(name=data_name, dtype='object')
                # load the spikes' data to the results to be returned...
                results[data_name][pop_label] = data_to_return
        # ...eventually compute the results and add it to the output dict...
        results.update(
            self._compute_spikes_correlations_from_spikes_trains(spikes_trains,
                                                                 pop_labels, pop_reg_labels, all_regions_labels,
                                                                 computations_methods, computations_kwargs,
                                                                 bin_kwargs, results_names))
        return results

    def compute_spikes_rates_and_correlations(self, spikes_data, populations_sizes=[],
                                              rates_methods=[], rates_kwargs=[{}], rate_results_names=[],
                                              corrs_methods=[], corrs_kwargs=[{}], corrs_results_names=[],
                                              bin_kwargs={}, data_method=lambda x, **kwargs: x, data_kwargs={}):
        """A method to compute several spikes' rates and correlations' measures
           on the same spikes' data or recording Devices.
           Arguments:
            - spikes_data: Spike recording populations' DeviceSets or data loaded from them in a pandas.Series.
            - population_sizes=[]: list of numbers of neurons per region in such a population,
                                   If pop_device_or_data is a DeviceSet,
                                   it will be overwritten by the data loading method.
            - rates_methods: a list of SpikingNetworkAnalyserBase class' "compute_" methods,
                              to perform the desired rate computations.
                              It defaults to computing mean_rate and mean_rate_time_series.
            - rates_kwargs: a list of keyword arguments dictionaries for the rate computation methods.
                            It len(rates_kwargs) >= len(rates_methods) to be computed,
                            each method to be computed will get its own kwargs dict.
                            If len(rates_kwargs) > 0 but < len(rates_methods) to be computed,
                            all methods will get rates_kwargs[0] kwargs dict.
                            If len(rates_kwargs) == 0,
                            an empty kwargs dict is passed to all rate computations' methods
            - rate_results_names=[]: a list of names (strings) to be given to the pandas.Series
                                     or xarray.DataArray instances of the rate results of each measure.
            - corrs_methods: a list of SpikingNetworkAnalyserBase class' "compute_" methods,
                             to perform the desired correlations' computations.
                             It defaults to computing correlation_coefficient.
            - corrs_kwargs: a list of keyword arguments dictionaries.
                            It len(corrs_kwargs) >= len(corrs_methods) to be computed,
                            each method to be computed will get its own kwargs dict.
                            If len(corrs_kwargs) > 0 but < len(corrs_methods) to be computed,
                            all methods will get corrs_kwargs[0] kwargs dict.
                            If len(corrs_kwargs) == 0,
                            an empty kwargs dict is passed to all correlation computations' methods
            - bin_kwargs={}: a dictionary of potential keyword arguments to be passed
                             to the method computing Binned Spikes' Trains.
            - data_method: a SpikingNetworkAnalyserBase class' "get_" method
                           to load data from recording Device or a file (TODO).
                           Default = lambda x, **kwargs, assuming that spikes_data already comprises of loaded data.
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data_method.

        """
        # Compute first the spikes' rate results:
        results = self.compute_spikes_measures(spikes_data, populations_sizes,
                                               rates_methods, rates_kwargs, rate_results_names,
                                               data_method, data_kwargs, return_spikes_trains=self.spikes_train_name)
        # Loop over the resulting Spikes Trains and prepare for the correlations' computations:
        pop_labels = []
        all_regions_labels = []
        pop_reg_labels = []
        spikes_trains = []
        for pop_label, pop_spikes_trains in results[self.spikes_train_name].items():
            pop_labels.append(pop_label)
            for reg_label, reg_spikes_trains in pop_spikes_trains.items():
                pop_reg_labels.append((pop_label, reg_label))
                if reg_label not in all_regions_labels:
                    all_regions_labels.append(reg_label)
                spikes_trains.append(reg_spikes_trains)
        # Compute the correlations' results and add them to the results' dictionary:
        results.update(
            self._compute_spikes_correlations_from_spikes_trains(spikes_trains,
                                                                 pop_labels, pop_reg_labels, all_regions_labels,
                                                                 corrs_methods, corrs_kwargs,
                                                                 bin_kwargs, corrs_results_names))
        return results

    def compute_continuous_data_measures(self, data, populations_sizes=[], variables=None,
                                         computations_methods=[], computations_kwargs=[], results_names=[],
                                         data_method=lambda x, **kwargs: x, data_kwargs={}):
        """A method to compute several continuous time variables' measures together on the same data.
           Arguments:
            - data: a pandas.Series of recording devices or of data loaded from such devices.
            - population_sizes=[]: list of numbers of neurons per region in such a population,
                                  If pop_device_or_data is a DeviceSet,
                                  it will be overwritten by the data loading method.
            - variables=None: a selection of continuous time variables to include in the computation. By default, all.
            - computations_methods: a list of SpikingNetworkAnalyserBase class' "compute_" methods,
                                    to perform the desired computations. By default, the mean_field_time_series.
            - computations_kwargs: a list of keyword arguments dictionaries.
                                   It len(computations_kwargs) >= len(computations_methods) to be computed,
                                   each method to be computed will get its own kwargs dict.
                                   If len(computations_kwargs) > 0 but < len(computations_methods) to be computed,
                                   all methods will get computations_kwargs[0] kwargs dict.
                                   If len(computations_kwargs) == 0,
                                   an empty kwargs dict is passed to all computations' methods
            - results_names=[]: a list of names (strings) to be given to the pandas.Series or xarray.DataArray instances
                                of the results of each measure.
            - data_method: a SpikingNetworkAnalyserBase class' "get_" method
                           to load data from recording Device or a file (TODO).
                           Default = lambda x, **kwargs, assuming that data already comprises of loaded data.
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data_method.
        """
        device_dims = data_kwargs.get("dims_names", ["Time", "Variable"])
        if variables:
            data_kwargs["variables"] = variables
        if len(computations_methods) == 0:
            # The default method to be computed and the default name.
            computations_methods = [self.compute_mean_field_time_series]
            results_names = ["Populations' Mean Field Time Series"]
        # Prepare the results dictionary.
        results = self._prepare_results(results_names, computations_methods)
        if len(data):
            # If there are any data, compute the results:
            results = \
                self._compute_results_from_population_generator(results,
                                                                self._populations_generator(data,
                                                                                           populations_sizes,
                                                                                           computations_methods,
                                                                                           computations_kwargs,
                                                                                           data_method,
                                                                                           data_kwargs)
                                                                )
            results = self.convert_results_to_output_type(results, device_dims + ["Region", "Population", "Neuron"])
        return results

    def compute_spikeNet_rates(self, populations_devices=None, regions=None,
                               computations_kwargs={}, data_kwargs={},
                               return_spikes_trains=False, return_devices=False):
        """A method to compute mean (across time) spikes' rates per neuron
           from data loaded by the spikes' recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - computations_kwargs: keyword arguments dictionary to be passed to the computation method
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_spikes_trains=False: flag to return the Spikes Trains used in the computations or not
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all spikes' recording devices of the SpikingNetwork
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            # Compute if there are any such devices:
            computations_kwargs["rate_method"] = self.compute_rate
            if return_spikes_trains:
                return_spikes_trains = self.spikes_train_name
            results = self.compute_spikes_measures(spikes_devices, [],
                                                   [self.compute_spikes_rates_by_neuron], [computations_kwargs],
                                                   ["Neurons' Spikes' Rates"],
                                                   self.get_spikes_times_by_neuron_from_device, data_kwargs,
                                                   return_spikes_trains)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return results, spikes_devices
        else:
            return results

    def compute_spikeNet_rates_time_series(self, populations_devices=None, regions=None,
                                           computations_kwargs={}, data_kwargs={},
                                           return_spikes_trains=False, return_devices=False):
        """A method to compute mean population spikes' rate time series
           from data loaded by the spikes' recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - computations_kwargs: keyword arguments dictionary to be passed to the computation method
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_spikes_trains=False: flag to return the Spikes Trains used in the computations or not
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all spikes' recording devices of the SpikingNetwork
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            # Compute if there are any such devices:
            computations_kwargs["rate_method"] = self.compute_rate_time_series
            if return_spikes_trains:
                return_spikes_trains = self.spikes_train_name
            results = self.compute_spikes_measures(spikes_devices, [],
                                                   [self.compute_spikes_rates_by_neuron], [computations_kwargs],
                                                   ["Neurons' Spikes' Rates' Time Series"],
                                                   self.get_spikes_times_by_neuron_from_device, data_kwargs,
                                                   return_spikes_trains)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return results, spikes_devices
        else:
            return results

    def compute_spikeNet_mean_rates(self, populations_devices=None, regions=None,
                                    computations_kwargs={}, data_method=None, data_kwargs={},
                                    return_spikes_trains=False, return_devices=False):
        """A method to compute mean (across time and neurons) spikes' rates per population
           from data loaded by the spikes' recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - computations_kwargs: keyword arguments dictionary to be passed to the computation method
            - data_method: a SpikingNetworkAnalyserBase class' "get_" method
                           to load data from a recording Device or a file (TODO).
                           Default = get_spikes_times_from_device, loading only the spikes' times array.
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_spikes_trains=False: flag to return the Spikes Trains used in the computations or not
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all spikes' recording devices of the SpikingNetwork
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            # Compute if there are any such devices:
            if data_method is None:
                data_method = self.get_spikes_times_from_device
            if return_spikes_trains:
                return_spikes_trains = self.spikes_train_name
            results = self.compute_spikes_measures(spikes_devices, [],
                                                   [self.compute_mean_rate], [computations_kwargs],
                                                   ["Mean Populations' Spikes' Rates"],
                                                   data_method, data_kwargs, return_spikes_trains)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return results, spikes_devices
        else:
            return results

    def compute_spikeNet_mean_rates_time_series(self, populations_devices=None, regions=None,
                                                computations_kwargs={}, data_kwargs={}, data_method=None,
                                                return_spikes_trains=False, return_devices=False):
        """A method to compute mean (across neurons) spikes' rates' time series per population
           from data loaded by the spikes' recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - computations_kwargs: keyword arguments dictionary to be passed to the computation method
            - data_method: a SpikingNetworkAnalyserBase class' "get_" method
                           to load data from a recording Device or a file (TODO).
                           Default = get_spikes_times_from_device, loading only the spikes' times array.
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_spikes_trains=False: flag to return the Spikes Trains used in the computations or not
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all spikes' recording devices of the SpikingNetwork
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            # Compute if there are any such devices:
            if data_method is None:
                data_method = self.get_spikes_times_from_device
            if return_spikes_trains:
                return_spikes_trains = self.spikes_train_name
            results = self.compute_spikes_measures(spikes_devices, [],
                                                   [self.compute_mean_instant_rate], [computations_kwargs],
                                                   ["Mean Populations' Spikes' Rates' Time Series"],
                                                   data_method, data_kwargs, return_spikes_trains)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return results, spikes_devices
        else:
            return results

    def compute_spikeNet_total_rates(self, populations_devices=None, regions=None,
                                     computations_kwargs={}, data_kwargs={}, data_method=None,
                                     return_spikes_trains=False, return_devices=False):
        """A method to compute the total (summing across neurons) spikes' rates per population
           from data loaded by the spikes' recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - computations_kwargs: keyword arguments dictionary to be passed to the computation method
            - data_method: a SpikingNetworkAnalyserBase class' "get_" method
                           to load data from a recording Device or a file (TODO).
                           Default = get_spikes_times_from_device, loading only the spikes' times array.
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_spikes_trains=False: flag to return the Spikes Trains used in the computations or not
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all spikes' recording devices of the SpikingNetwork
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            # Compute if there are any such devices:
            if data_method is None:
                data_method = self.get_spikes_times_from_device
            if return_spikes_trains:
                return_spikes_trains = self.spikes_train_name
            results = self.compute_spikes_measures(spikes_devices, [],
                                                   [self.compute_total_rate], [computations_kwargs],
                                                   ["Total Populations' Spikes' Rates"],
                                                   data_method, data_kwargs, return_spikes_trains)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return results, spikes_devices
        else:
            return results

    def compute_spikeNet_total_rates_time_series(self, populations_devices=None, regions=None,
                                                 computations_kwargs={}, data_kwargs={}, data_method=None,
                                                 return_spikes_trains=False, return_devices=False):
        """A method to compute the total (summing across neurons) spikes' rates' time series per population
           from data loaded by the spikes' recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - computations_kwargs: keyword arguments dictionary to be passed to the computation method
            - data_method: a SpikingNetworkAnalyserBase class' "get_" method
                           to load data from a recording Device or a file (TODO).
                           Default = get_spikes_times_from_device, loading only the spikes' times array.
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_spikes_trains=False: flag to return the Spikes Trains used in the computations or not
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all spikes' recording devices of the SpikingNetwork
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            # Compute if there are any such devices:
            if data_method is None:
                data_method = self.get_spikes_times_from_device
            if return_spikes_trains:
                return_spikes_trains = self.spikes_train_name
            results = self.compute_spikes_measures(spikes_devices, [],
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
                                             corrs_methods=[], corrs_kwargs=[{}], results_names=[],
                                             bin_kwargs={}, data_method=None, data_kwargs={}, return_devices=False):
        """A method to compute the spikes' correlations among populations and regions
           from data loaded by the spikes' recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - corrs_methods: a list of SpikingNetworkAnalyserBase class' "compute_" methods,
                             to perform the desired correlations' computations.
                             It defaults to compute_correlation_coefficient.
            - corrs_kwargs: a list of keyword arguments dictionaries.
                            It len(corrs_kwargs) >= len(corrs_methods) to be computed,
                            each method to be computed will get its own kwargs dict.
                            If len(corrs_kwargs) > 0 but < len(corrs_methods) to be computed,
                            all methods will get corrs_kwargs[0] kwargs dict.
                            If len(corrs_kwargs) == 0,
                            an empty kwargs dict is passed to all correlation computations' methods
            - bin_kwargs={}: a dictionary of potential keyword arguments to be passed
                             to the method computing Binned Spikes' Trains.
            - data_method: a SpikingNetworkAnalyserBase class' "get_" method
                           to load data from a recording Device or a file (TODO).
                           Default = get_spikes_times_from_device, loading only the spikes' times array.
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all spikes' recording devices of the SpikingNetwork
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            # Compute if there are any such devices:
            if data_method is None:
                data_method = self.get_spikes_times_from_device
            results = self.compute_spikes_correlations(spikes_devices, computations_methods=corrs_methods,
                                                       computations_kwargs=corrs_kwargs,
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
                                                       rates_methods=[], rates_kwargs=[{}], rate_results_names=[],
                                                       corrs_methods=[], corrs_kwargs=[{}], corrs_results_names=[],
                                                       bin_kwargs={}, data_method=None, data_kwargs={},
                                                       return_devices=False):
        """A method to compute the spikes' rates and correlations among populations and regions
           from data loaded by the spikes' recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - rates_methods: a list of SpikingNetworkAnalyserBase class' "compute_" methods,
                              to perform the desired rate computations.
                              It defaults to computing compute_mean_rate and compute_mean_rate_time_series.
            - rates_kwargs: a list of keyword arguments dictionaries for the rate computation methods.
                            It len(rates_kwargs) >= len(rates_methods) to be computed,
                            each method to be computed will get its own kwargs dict.
                            If len(rates_kwargs) > 0 but < len(rates_methods) to be computed,
                            all methods will get rates_kwargs[0] kwargs dict.
                            If len(rates_kwargs) == 0,
                            an empty kwargs dict is passed to all rate computations' methods
            - rate_results_names=[]: a list of names (strings) to be given to the pandas.Series
                                     or xarray.DataArray instances of the rate results of each measure.
            - corrs_methods: a list of SpikingNetworkAnalyserBase class' "compute_" methods,
                             to perform the desired correlations' computations.
                             It defaults to compute_correlation_coefficient.
            - corrs_kwargs: a list of keyword arguments dictionaries.
                            It len(corrs_kwargs) >= len(corrs_methods) to be computed,
                            each method to be computed will get its own kwargs dict.
                            If len(corrs_kwargs) > 0 but < len(corrs_methods) to be computed,
                            all methods will get corrs_kwargs[0] kwargs dict.
                            If len(corrs_kwargs) == 0,
                            an empty kwargs dict is passed to all correlation computations' methods
            - bin_kwargs={}: a dictionary of potential keyword arguments to be passed
                             to the method computing Binned Spikes' Trains.
            - data_method: a SpikingNetworkAnalyserBase class' "get_" method
                           to load data from a recording Device or a file (TODO).
                           Default = get_spikes_times_from_device, loading only the spikes' times array.
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all spikes' recording devices of the SpikingNetwork
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            # Compute if there are any such devices:
            if data_method is None:
                data_method = self.get_spikes_times_from_device
            results = self.compute_spikes_rates_and_correlations(spikes_devices, [],
                                                                 rates_methods, rates_kwargs, rate_results_names,
                                                                 corrs_methods, corrs_kwargs, corrs_results_names,
                                                                 bin_kwargs, data_method, data_kwargs)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return results, spikes_devices
        else:
            return results

    def compute_spikeNet_synchronization(self, populations_devices=None, regions=None,
                                         comp_methods=[], computations_kwargs=[{}], data_kwargs={},
                                         return_spikes_trains=False, return_devices=False):
        """A method to compute mean (across time) spikes' rates per neuron
           from data loaded by the spikes' recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - computations_kwargs: keyword arguments dictionary to be passed to the computation method
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_spikes_trains=False: flag to return the Spikes Trains used in the computations or not
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all spikes' recording devices of the SpikingNetwork
        spikes_devices = self.spikeNet.get_spikes_devices(populations_devices=populations_devices, regions=regions)
        if len(spikes_devices):
            # Compute if there are any such devices:
            if len(comp_methods) == 0:
                comp_methods = [self.compute_spikes_sync]
            if return_spikes_trains:
                return_spikes_trains = self.spikes_trains_name
            results = self.compute_spikes_measures(spikes_devices, [],
                                                   comp_methods, computations_kwargs,
                                                   ["Populations' Spikes' Synchronization"],
                                                   self.get_spikes_times_by_neuron_from_device, data_kwargs,
                                                   return_spikes_trains)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return results, spikes_devices
        else:
            return results

    def compute_spikeNet_mean_field(self, populations_devices=None, regions=None, variables=None,
                                    computations_kwargs={}, data_kwargs={}, return_devices=False):
        """A method to compute the mean field (mean across time and neurons)
           of continuous time variables per population from data loaded by recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - computations_kwargs: keyword arguments dictionary to be passed to the computation method
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all continuous time data recording devices of the SpikingNetwork
        devices = self.spikeNet.get_continuous_time_devices(regions=regions, populations_devices=populations_devices)
        if len(devices):
            # Compute if there are any such devices:
            results = self.compute_continuous_data_measures(devices, [], variables,
                                                            [self.compute_mean_field], [computations_kwargs],
                                                            ["Populations' Mean Field"],
                                                            self.get_data_by_neuron_from_device, data_kwargs)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return results, devices
        else:
            return results

    def compute_spikeNet_total_activity(self, populations_devices=None, regions=None, variables=None,
                                        computations_kwargs={}, data_kwargs={}, return_devices=False):
        """A method to compute the total activity (mean across time and summing across neurons)
           of continuous time variables per population from data loaded by recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - computations_kwargs: keyword arguments dictionary to be passed to the computation method
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all continuous time data recording devices of the SpikingNetwork
        devices = self.spikeNet.get_continuous_time_devices(regions=regions, populations_devices=populations_devices)
        if len(devices):
            # Compute if there are any such devices:
            results = self.compute_continuous_data_measures(devices, [], variables,
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

    def compute_spikeNet_mean_field_time_series(self, populations_devices=None, regions=None, variables=None,
                                                computations_kwargs={}, data_kwargs={}, return_devices=False):
        """A method to compute the mean field (mean across neurons) time series
           of continuous time variables per population from data loaded by recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - computations_kwargs: keyword arguments dictionary to be passed to the computation method
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all continuous time data recording devices of the SpikingNetwork
        devices = self.spikeNet.get_continuous_time_devices(regions=regions, populations_devices=populations_devices)
        if len(devices):
            # Compute if there are any such devices:
            results = self.compute_continuous_data_measures(devices, [], variables,
                                                           [self.compute_mean_field_time_series], [computations_kwargs],
                                                           ["Populations' Mean Field Time Series"],
                                                           self.get_data_by_neuron_from_device, data_kwargs)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return results, devices
        else:
            return results

    def compute_spikeNet_total_activity_time_series(self, populations_devices=None, regions=None, variables=None,
                                                    computations_kwargs={}, data_kwargs={}, return_devices=False):
        """A method to compute the total activity (sum across neurons) time series
           of continuous time variables per population from data loaded by recording devices of the SpikingNetwork.
           Arguments:
            - populations_devices: a list of selected populations' spikes" recording DeviceSets labels (strings)
                                   to be included in the computation. By default all.
            - regions: a list of selected regions labels (strings) to be included in the computation. By default all.
            - computations_kwargs: keyword arguments dictionary to be passed to the computation method
            - data_kwargs={}: a dictionary of potential keyword arguments to be passed to the data loading method.
            - return_devices=False: a flag to return the DeviceSets used to load data from.
        """
        # Get all continuous time data recording devices of the SpikingNetwork
        devices = self.spikeNet.get_continuous_time_devices(regions=regions, populations_devices=populations_devices)
        if len(devices):
            # Compute if there are any such devices:
            results = self.compute_continuous_data_measures(devices, [], variables,
                                                            [self.compute_total_activity_time_series],
                                                            [computations_kwargs],
                                                            ["Populations' Total Activity Time Series"],
                                                            self.get_data_by_neuron_from_device, data_kwargs)
            if len(results) == 1:
                results = list(results.values())[0]
        else:
            results = None
        if return_devices:
            return results, devices
        else:
            return results
