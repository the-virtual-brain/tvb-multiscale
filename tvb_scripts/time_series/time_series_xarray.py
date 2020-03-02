# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
The TimeSeries datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
from copy import deepcopy
from six import string_types
import xarray as xr
import numpy as np
from tvb.datatypes import sensors, surfaces, volumes, region_mapping, connectivity
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List, Float, narray_summary_info
from tvb_scripts.utils.data_structures_utils import is_integer


def prepare_4d(data):
    if data.ndim < 2:
        raise ValueError("The data array is expected to be at least 2D!")
    if data.ndim < 4:
        if data.ndim == 2:
            data = np.expand_dims(data, 2)
        data = np.expand_dims(data, 3)
    return data


class TimeSeries(HasTraits):
    """
    Base time-series dataType.
    """

    _data = xr.DataArray([])

    _default_labels_ordering = List(
        default=("Time", "State Variable", "Space", "Mode"),
        label="Dimension Names",
        doc="""List of strings representing names of each data dimension""")

    title = Attr(str)

    @property
    def data(self):
        # Return numpy array
        return self._data.values

    @property
    def name(self):
        return self._data.name

    @property
    def shape(self):
        return self._data.shape

    @property
    def nr_dimensions(self):
        return self._data.ndim

    @property
    def number_of_dimensions(self):
        return self.nr_dimensions

    @property
    def time_length(self):
        try:
            return self._data.shape[0]
        except:
            return None

    @property
    def number_of_variables(self):
        try:
            return self.shape[1]
        except:
            return None

    @property
    def number_of_labels(self):
        try:
            return self.shape[2]
        except:
            return None

    @property
    def number_of_samples(self):
        try:
            return self.shape[3]
        except:
            return None

    @property
    def labels_ordering(self):
        return list(self._data.dims)

    @property
    def labels_dimensions(self):
        d = {}
        for key, val in zip(list(self._data.coords.keys()),
                            list([value.values for value in self._data.coords.values()])):
            d[key] = val
        return d

    @property
    def time(self):
        return self._data.coords[self._data.dims[0]].values

    # xarrays have a attrs dict with useful attributes

    @property
    def start_time(self):
        try:
            return self.time[0]
        except:
            return None

    @property
    def end_time(self):
        try:
            return self.time[-1]
        except:
            return None

    @property
    def duration(self):
        try:
            return self.end_time - self.start_time
        except:
            return None

    @property
    def sample_period(self):
        try:
            return np.mean(np.diff(self.time))
        except:
            return None

    @property
    def sample_rate(self):
        try:
            return 1.0 / self.sample_period
        except:
            return None

    @property
    def sample_period_unit(self):
        try:
            return self._data.attrs["sample_period_unit"]
        except:
            return ""

    @property
    def time_unit(self):
        return self.sample_period_unit

    @property
    def squeeze(self):
        return self._data.squeeze()

    @property
    def squeezed(self):
        return self.data.squeeze()

    @property
    def flattened(self):
        return self.data.flatten()

    def _configure_input_time(self, data, **kwargs):
        # Method to initialise time attributes
        # It gives priority to an input time vector, if any.
        # Subsequently, it considers start_time and sample_period kwargs
        sample_period = kwargs.pop("sample_period", None)
        start_time = kwargs.pop("start_time", None)
        time = kwargs.pop("time", None)
        time_length = data.shape[0]
        if time_length > 0:
            if time is None:
                if start_time is not None and sample_period is not None:
                    end_time = start_time + (time_length - 1) * sample_period
                    time = np.arange(start_time, end_time + sample_period, sample_period)
                    return time, start_time, end_time, sample_period
                else:
                    raise ValueError("Neither time vector nor start_time and/or "
                                     "sample_period are provided as input arguments!")
            else:
                assert time_length == len(time)
                start_time = time[0]
                end_time = time[-1]
                if len(time) > 1:
                    sample_period = np.mean(np.diff(time))
                    assert end_time == start_time + (time_length - 1) * sample_period
                else:
                    sample_period = None
                return time, start_time, end_time, sample_period
        else:
            # Empty data
            return None, start_time, None, sample_period

    def _configure_input_labels(self, **kwargs):
        # Method to initialise label attributes
        # It gives priority to labels_ordering,
        # i.e., labels_dimensions dict cannot ovewrite it
        # and should agree with it
        labels_ordering = list(kwargs.pop("labels_ordering", self._default_labels_ordering))
        labels_dimensions = kwargs.pop("labels_dimensions", None)
        if isinstance(labels_dimensions, dict):
            assert [key in labels_ordering for key in labels_dimensions.keys()]
        return labels_ordering, labels_dimensions

    def from_numpy(self, data, **kwargs):
        # We have to infer time and labels inputs from kwargs
        data = prepare_4d(data)
        time, start_time, end_time, sample_period = self._configure_input_time(data, **kwargs)
        labels_ordering, labels_dimensions = self._configure_input_labels(**kwargs)
        if time is not None:
            if labels_dimensions is None:
                labels_dimensions = {}
            labels_dimensions[labels_ordering[0]] = time
        self._data = xr.DataArray(data, dims=labels_ordering, coords=labels_dimensions, attrs=kwargs,
                                  name=self.__class__.__name__)

    def _configure_time(self):
        assert self.time[0] == self.start_time
        assert self.time[-1] == self.end_time
        if self.time_length > 1:
            assert self.sample_period == (self.end_time - self.start_time) / (self.time_length - 1)

    def _configure_labels(self):
        for i_dim in range(1, self.nr_dimensions):
            dim_label = self.labels_ordering[i_dim]
            val = self.labels_dimensions.get(dim_label, None)
            if val is not None:
                assert len(val) == self.shape[i_dim]
            else:
                # We set by default integer labels if no input labels are provided by the user
                self._data.coords[self._data.dims[i_dim]] = np.arange(0, self.shape[i_dim])

    def configure(self):
        # To be always used when a new object is created
        # to check that everything is set correctly
        self.title = self.name
        super(TimeSeries, self).configure()
        try:
            time_length = self.time_length
        except:
            pass  # It is ok if data is empty
        if time_length is not None and time_length > 0:
            self._configure_time()
            self._configure_labels()

    def __init__(self, data=None, **kwargs):
        super(TimeSeries, self).__init__()
        if isinstance(data, (list, tuple)):
            self.from_numpy(np.array(data), **kwargs)
        elif isinstance(data, np.ndarray):
            self.from_numpy(data, **kwargs)
        elif isinstance(data, self.__class__):
            for attr, val in data.__dict__.items():
                setattr(self, attr, val)
        else:
            # Assuming data is an input xr.DataArray() can handle,
            if isinstance(data, dict):
                # ...either as kwargs
                self._data = xr.DataArray(**data)
            else:
                # ...or as args
                # including a xr.DataArray or None
                self._data = xr.DataArray(data)
        self.configure()

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {
            "Time-series type": self.__class__.__name__,
            "Time-series name": self.title,
            "Dimensions": self.labels_ordering,
            "Time units": self.sample_period_unit,
            "Sample period": self.sample_period,
            "Length": self.time_length
        }
        summary.update(narray_summary_info(self.data))
        return summary

    def duplicate(self, **kwargs):
        # Since all labels are internal to xarray,
        # it suffices to pass a new (e.g., sliced) xarray _data as kwarg
        # for all labels to be set correctly (and confirmed by the call to configure(),
        # whereas any other attributes of TimeSeries will be copied
        duplicate = deepcopy(self)
        for attr, value in kwargs.items():
            setattr(duplicate, attr, value)
        duplicate.configure()
        return duplicate

    def _assert_array_indices(self, slice_tuple):
        if is_integer(slice_tuple) or isinstance(slice_tuple, string_types):
            return ([slice_tuple], )
        else:
            slice_list = []
            for slc in slice_tuple:
                if is_integer(slc) or isinstance(slc, string_types):
                    slice_list.append([slc])
                else:
                    slice_list.append(slc)
            return tuple(slice_list)

    def _get_index_for_slice_label(self, slice_label, slice_idx):
        return np.where(self.labels_dimensions[self.labels_ordering[slice_idx]] == slice_label)[0][0]

    def _check_for_string_slice_indices(self, current_slice, slice_idx):
        slice_label1 = current_slice.start
        slice_label2 = current_slice.stop

        if isinstance(slice_label1, string_types):
            slice_label1 = self._get_index_for_slice_label(slice_label1, slice_idx)
        if isinstance(slice_label2, string_types):
            # NOTE!: In case of a string slice, we consider stop included!
            slice_label2 = self._get_index_for_slice_label(slice_label2, slice_idx) + 1

        return slice(slice_label1, slice_label2, current_slice.step)

    def _resolve_mixted_slice(self, slice_tuple):
        slice_list = []
        for idx, current_slice in enumerate(slice_tuple):
            if isinstance(current_slice, slice):
                slice_list.append(self._check_for_string_slice_indices(current_slice, idx))
            else:
                # If not a slice, it will be an iterable:
                for i_slc, slc in enumerate(current_slice):
                    if isinstance(slc, string_types):
                        current_slice[i_slc] = self.labels_dimensions[self.labels_ordering[idx]].tolist().index(slc)
                    else:
                        current_slice[i_slc] = slc
                slice_list.append(current_slice)
        return tuple(slice_list)

    # Return a TimeSeries object
    def __getitem__(self, slice_tuple):
        slice_tuple = self._assert_array_indices(slice_tuple)
        try:
            # For integer indices
            return self.duplicate(_data=self._data[slice_tuple])
        except:
            try:
                # For label indices
                # xrarray.DataArray.loc slices along labels
                # Assuming that all dimensions without input labels
                # are configured with labels of integer indices=
                return self.duplicate(_data=self._data.loc[slice_tuple])
            except:
                # Still, for a conflicting mixture that has to be resolved
                return self.duplicate(_data=self._data[self._resolve_mixted_slice(slice_tuple)])

    def __setitem__(self, slice_tuple, values):
        slice_tuple = self._assert_array_indices(slice_tuple)
        # Mind that xarray can handle setting values both from a numpy array and/or another xarray
        if isinstance(values, self.__class__):
            values = np.array(values.data)
        try:
            # For integer indices
            self._data[slice_tuple] = values
        except:
            try:
                # For label indices
                # xrarray.DataArray.loc slices along labels
                # Assuming that all dimensions without input labels
                # are configured with labels of integer indices
                self._data.loc[slice_tuple] = values
            except:
                # Still, for a conflicting mixture that has to be resolved
                self._data[self._resolve_mixted_slice(slice_tuple)] = values

    # def __getattr__(self, attr_name):
    #     # We are here because attr_name is not an attribute of TimeSeries...
    #     try:
    #         # First try to see if this is a xarray.DataArray attribute
    #         getattr(self._data, attr_name)
    #     except:
    #         # TODO: find out if this part works, given that it is not really necessary
    #         # Now try to behave as if this was a getitem call:
    #         slice_list = [slice(None)]  # for first dimension index, i.e., time
    #         for i_dim in range(1, self.nr_dimensions):
    #             if self.labels_ordering[i_dim] in self.labels_dimensions.keys() and \
    #                     attr_name in self.labels_dimensions[self.labels_ordering[i_dim]]:
    #                 slice_list.append(attr_name)
    #                 return self._data.loc[tuple(slice_list)]
    #             else:
    #                 slice_list.append(slice(None))
    #         raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, attr_name))
    #
    # def __setattr__(self, attr_name, value):
    #     # We are here because attr_name is not an attribute of TimeSeries...
    #     try:
    #         # First try to see if this is a xarray.DataArray attribute
    #         getattr(self._data, attr_name, value)
    #     except:
    #         # TODO: find out if this part works, given that it is not really necessary
    #         # Now try to behave as if this was a setitem call:
    #         slice_list = [slice(None)]  # for first dimension index, i.e., time
    #         for i_dim in range(1, self.nr_dimensions):
    #             if self.labels_ordering[i_dim] in self.labels_dimensions.keys() and \
    #                     attr_name in self.labels_dimensions[self.labels_ordering[i_dim]]:
    #                 slice_list.append(attr_name)
    #                 self._data.loc[tuple(slice_list)] = value
    #                 return
    #             else:
    #                 slice_list.append(slice(None))
    #         raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, attr_name))

    def swapaxes(self, ax1, ax2):
        dims = list(self.dims)
        dims[ax1] = self.dims[ax2]
        dims[ax2] = self.dims[ax1]
        return self.transpose(*dims)

    def _prepare_plot_args(self, **kwargs):
        plotter = kwargs.pop("plotter", None)
        labels_ordering = self.labels_ordering
        time = labels_ordering[0]
        # Regions
        if self.shape[2] > 1:
            regions = labels_ordering[2]
        else:
            regions = None
        # Variables
        if self.shape[1] > 1:
            variables = labels_ordering[1]
        else:
            variables = None
        # Modes
        if self.shape[3] > 1:
            modes = labels_ordering[3]
        else:
            modes = None
        return time, variables, regions, modes, plotter, kwargs

    # def plot(self, **kwargs):
    #     time, variables, regions, modes, plotter, kwargs = \
    #         self._prepare_plot_args(**kwargs)
    #     robust = kwargs.pop("robust", True)
    #     cmap = kwargs.pop("cmap", "jet")
    #     figure_name = kwargs.pop("figname", "%s plot" % self.title)
    #     output = self._data.plot(x=time,         # Time
    #                              y=regions,      # Regions
    #                              col=variables,  # Variables
    #                              row=modes,      # Modes/Samples/Populations etc
    #                              robust=robust, cmap=cmap, **kwargs)
    #     # TODO: Something better than this temporary hack for base_plotter functionality
    #     if plotter is not None:
    #         plotter._save_figure(figure_name=figure_name)
    #         plotter._check_show()
    #     return output

    def plot_timeseries(self, **kwargs):
        if kwargs.pop("per_variable", False):
            outputs = []
            figure_name = kwargs.pop("figure_name", "%s time series plot" % self.title)
            for var in self.labels_dimensions[self.labels_ordering[1]]:
                kwargs["figure_name"] = figure_name + " - %s" % var
                outputs.append(self[:, var].plot_timeseries(**kwargs))
            return outputs
        time, variables, regions, modes, plotter, kwargs = \
            self._prepare_plot_args(**kwargs)
        if self.shape[3] == 1 or variables is None or regions is None or modes is None:
            if variables is None:
                data = self._data[:, 0]
                data.name = self.labels_dimensions[self.labels_ordering[1]][0]
            else:
                data = self._data
            col_wrap = kwargs.pop("col_wrap", self.shape[3])  # Row per variable
            figure_name = kwargs.pop("figure_name", "%s time series plot" % self.title)
            output = data.plot.line(x=time,       # Time
                                    y=variables,  # Variables
                                    hue=regions,  # Regions
                                    col=modes,    # Modes/Samples/Populations etc
                                    col_wrap=col_wrap, **kwargs)
            # TODO: Something better than this temporary hack for base_plotter functionality
            if plotter is not None:
                plotter.base._save_figure(figure_name=figure_name)
                plotter.base._check_show()
            return output
        else:
            if plotter is not None:
                kwargs["plotter"] = plotter
            return self.plot_raster(**kwargs)

    def plot_raster(self, **kwargs):
        if kwargs.pop("per_variable", False):
            outputs = []
            figure_name = kwargs.pop("figure_name", "%s raster plot" % self.title)
            for var in self.labels_dimensions[self.labels_ordering[1]]:
                kwargs["figure_name"] = figure_name + " - %s" % var
                outputs.append(self[:, var].plot_raster(**kwargs))
            return outputs
        time, variables, regions, modes, plotter, kwargs = \
            self._prepare_plot_args(**kwargs)
        figure_name = kwargs.pop("figure_name", "%s raster plot" % self.title)
        data = xr.DataArray(self._data)
        for i_var, var in enumerate(self.labels_dimensions[self.labels_ordering[1]]):
            # Remove mean
            data[:, i_var] -= data[:, i_var].mean()
            # Compute approximate range for this variable
            amplitude = 0.9 * (data[:, i_var].max() - data[:, i_var].min())
            # Add the step on y axis for this variable and for each Region's data
            for i_region in range(self.shape[2]):
                data[:, i_var, i_region] += amplitude * i_region
        if self.shape[3] > 1 and regions is not None and modes is not None:
            hue = "%s - %s" % (regions, modes)
            data = data.stack({hue: (regions, modes)})
        else:
            hue = regions or modes
        col_wrap = kwargs.pop("col_wrap", self.shape[1])  # All variables in columns
        output = data.plot(x=time,         # Time
                           hue=hue,        # Regions and/or Modes/Samples/Populations etc
                           col=variables,  # Variables
                           yincrease=False, col_wrap=col_wrap, **kwargs)
        # TODO: Something better than this temporary hack for base_plotter functionality
        if plotter is not None:
            plotter.base._save_figure(figure_name=figure_name)
            plotter.base._check_show()
        return output


class SensorsTSBase(TimeSeries):

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(SensorsTSBase, self).summary_info()
        summary.update({"Source Sensors": self.sensors.title})
        return summary


class TimeSeriesEEG(SensorsTSBase):
    """ A time series associated with a set of EEG sensors. """

    sensors = Attr(field_type=sensors.SensorsEEG)
    _default_labels_ordering = List(of=str, default=("Time", "1", "EEG Sensor", "1"))


class TimeSeriesMEG(SensorsTSBase):
    """ A time series associated with a set of MEG sensors. """

    sensors = Attr(field_type=sensors.SensorsMEG)
    _default_labels_ordering = List(of=str, default=("Time", "1", "MEG Sensor", "1"))


class TimeSeriesSEEG(SensorsTSBase):
    """ A time series associated with a set of Internal sensors. """

    sensors = Attr(field_type=sensors.SensorsInternal)
    _default_labels_ordering = List(of=str, default=("Time", "1", "sEEG Sensor", "1"))


class TimeSeriesRegion(TimeSeries):
    """ A time-series associated with the regions of a connectivity. """

    connectivity = Attr(field_type=connectivity.Connectivity)
    region_mapping_volume = Attr(field_type=region_mapping.RegionVolumeMapping, required=False)
    region_mapping = Attr(field_type=region_mapping.RegionMapping, required=False)
    _default_labels_ordering = List(of=str, default=("Time", "State Variable", "Region", "Mode"))

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesRegion, self).summary_info()
        summary.update({
            "Source Connectivity": self.connectivity.title,
            "Region Mapping": self.region_mapping.title if self.region_mapping else "None",
            "Region Mapping Volume": (self.region_mapping_volume.title
                                      if self.region_mapping_volume else "None")
        })
        return summary


class TimeSeriesSurface(TimeSeries):
    """ A time-series associated with a Surface. """

    surface = Attr(field_type=surfaces.CorticalSurface)
    _default_labels_ordering = List(of=str, default=("Time", "State Variable", "Vertex", "Mode"))

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesSurface, self).summary_info()
        summary.update({"Source Surface": self.surface.title})
        return summary


class TimeSeriesVolume(TimeSeries):
    """ A time-series associated with a Volume. """

    volume = Attr(field_type=volumes.Volume)
    _default_labels_ordering = List(of=str, default=("Time", "X", "Y", "Z"))

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesVolume, self).summary_info()
        summary.update({"Source Volume": self.volume.title})
        return summary
