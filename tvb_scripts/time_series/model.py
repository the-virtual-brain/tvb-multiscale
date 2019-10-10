# -*- coding: utf-8 -*-

from six import string_types
from enum import Enum
from copy import deepcopy
import numpy

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_scripts.utils.log_error_utils import initialize_logger, warning
from tvb_scripts.utils.data_structures_utils import monopolar_to_bipolar
from tvb.datatypes.time_series import TimeSeries as TimeSeriesTVB
from tvb.datatypes.time_series import TimeSeriesRegion as TimeSeriesRegionTVB
from tvb.datatypes.time_series import TimeSeriesEEG as TimeSeriesEEGTVB
from tvb.datatypes.time_series import TimeSeriesMEG as TimeSeriesMEGTVB
from tvb.datatypes.time_series import TimeSeriesSEEG as TimeSeriesSEEGTVB
from tvb.datatypes.time_series import TimeSeriesSurface as TimeSeriesSurfaceTVB
from tvb.datatypes.time_series import TimeSeriesVolume as TimeSeriesVolumeTVB
from tvb.datatypes.sensors import Sensors, SensorsEEG, SensorsMEG, SensorsInternal


class TimeSeriesDimensions(Enum):
    TIME = "Time"
    VARIABLES = "State Variables"
    SPACE = "Space"
    SAMPLES = "Samples"


LABELS_ORDERING = [TimeSeriesDimensions.TIME,
                   TimeSeriesDimensions.VARIABLES,
                   TimeSeriesDimensions.SPACE,
                   TimeSeriesDimensions.SAMPLES]


class PossibleVariables(Enum):
    LFP = "lfp"
    SOURCE = "source"
    SENSORS = "sensors"
    EEG = "eeg"
    MEEG = "meeg"
    SEEG = "seeg"


def prepare_4D(data, logger=initialize_logger(__name__)):
    if data.ndim < 2:
        logger.error("The data array is expected to be at least 2D!")
        raise ValueError
    if data.ndim < 4:
        if data.ndim == 2:
            data = numpy.expand_dims(data, 2)
        data = numpy.expand_dims(data, 3)
    return data


class TimeSeries(TimeSeriesTVB):

    logger = initialize_logger(__name__)

    # labels_dimensions = {"space": numpy.array([]), "variables": numpy.array([])}

    def __init__(self, data, **kwargs):
        super(TimeSeries, self).__init__(**kwargs)
        self.data = prepare_4D(data, self.logger)
        self.time = kwargs.pop("time", None)
        if self.time is not None:
            self.start_time = float(kwargs.pop("start_time", self.time[0]))
            self.sample_period = float(kwargs.pop("sample_period", numpy.mean(numpy.diff(self.time))))
        self.configure()

    def duplicate(self, **kwargs):
        duplicate = deepcopy(self)
        for attr, value in kwargs.items():
            setattr(duplicate, attr, value)
        return duplicate.configure()

    def _get_index_of_state_variable(self, sv_label):
        try:
            sv_index = numpy.where(self.variables_labels == sv_label)[0][0]
        except KeyError:
            self.logger.error("There are no state variables defined for this instance. Its shape is: %s",
                              self.data.shape)
            raise
        except IndexError:
            self.logger.error("Cannot access index of state variable label: %s. Existing state variables: %s" % (
                sv_label, self.variables_labels))
            raise
        return sv_index

    def get_state_variable(self, sv_label):
        sv_data = self.data[:, self._get_index_of_state_variable(sv_label), :, :]
        subspace_labels_dimensions = deepcopy(self.labels_dimensions)
        subspace_labels_dimensions[self.labels_ordering[1]] = [sv_label]
        if sv_data.ndim == 3:
            sv_data = numpy.expand_dims(sv_data, 1)
        return self.duplicate(data=sv_data, labels_dimensions=subspace_labels_dimensions)

    def _get_indices_for_labels(self, list_of_labels):
        list_of_indices_for_labels = []
        for label in list_of_labels:
            try:
                space_index = numpy.where(self.space_labels == label)[0][0]
            except ValueError:
                self.logger.error("Cannot access index of space label: %s. Existing space labels: %s" %
                                  (label, self.space_labels))
                raise
            list_of_indices_for_labels.append(space_index)
        return list_of_indices_for_labels

    def get_subspace_by_index(self, list_of_index, **kwargs):
        self._check_space_indices(list_of_index)
        subspace_data = self.data[:, :, list_of_index, :]
        subspace_labels_dimensions = deepcopy(self.labels_dimensions)
        subspace_labels_dimensions[self.labels_ordering[2]] = self.space_labels[list_of_index].tolist()
        if subspace_data.ndim == 3:
            subspace_data = numpy.expand_dims(subspace_data, 2)
        return self.duplicate(data=subspace_data, labels_dimensions=subspace_labels_dimensions, **kwargs)

    def get_subspace_by_labels(self, list_of_labels):
        list_of_indices_for_labels = self._get_indices_for_labels(list_of_labels)
        return self.get_subspace_by_index(list_of_indices_for_labels)

    def __getattr__(self, attr_name):
        if self.labels_ordering[1] in self.labels_dimensions.keys():
            if attr_name in self.variables_labels:
                return self.get_state_variable(attr_name)
        if (self.labels_ordering[2] in self.labels_dimensions.keys()):
            if attr_name in self.space_labels:
                return self.get_subspace_by_labels([attr_name])
        try:
            return getattr(self, attr_name)
        except:
            # Hack to avoid stupid error messages when searching for __ attributes in numpy.array() call...
            # TODO: something better? Maybe not needed if we never do something like numpy.array(time_series)
            if attr_name.find("__") < 0:
                self.logger.error(
                    "Attribute %s is not defined for this instance! You can use the following labels: "
                    "state_variables = %s and space = %s" %
                    (attr_name, self.variables_labels, self.space_labels))
            raise AttributeError

    def _get_index_for_slice_label(self, slice_label, slice_idx):
        if slice_idx == 1:
            return self._get_indices_for_labels([slice_label])[0]
        if slice_idx == 2:
            return self._get_index_of_state_variable(slice_label)

    def _check_for_string_slice_indices(self, current_slice, slice_idx):
        slice_label1 = current_slice.start
        slice_label2 = current_slice.stop

        if isinstance(slice_label1, string_types):
            slice_label1 = self._get_index_for_slice_label(slice_label1, slice_idx)
        if isinstance(slice_label2, string_types):
            slice_label2 = self._get_index_for_slice_label(slice_label2, slice_idx)

        return slice(slice_label1, slice_label2, current_slice.step)

    def _get_string_slice_index(self, current_slice_string, slice_idx):
        return self._get_index_for_slice_label(current_slice_string, slice_idx)

    def __getitem__(self, slice_tuple):
        slice_list = []
        for idx, current_slice in enumerate(slice_tuple):
            if isinstance(current_slice, slice):
                slice_list.append(self._check_for_string_slice_indices(current_slice, idx))
            else:
                if isinstance(current_slice, string_types):
                    slice_list.append(self._get_string_slice_index(current_slice, idx))
                else:
                    slice_list.append(current_slice)

        return self.data[tuple(slice_list)]

    @property
    def shape(self):
        return self.data.shape

    @property
    def time_length(self):
        return self.data.shape[0]

    @property
    def number_of_labels(self):
        return self.data.shape[1]

    @property
    def number_of_variables(self):
        return self.data.shape[2]

    @property
    def number_of_samples(self):
        return self.data.shape[3]

    @property
    def end_time(self):
        return self.start_time + (self.time_length - 1) * self.sample_period

    @property
    def duration(self):
        return self.end_time - self.start_time

    def configure_time(self):
        self.time = numpy.arange(self.start_time, self.end_time + self.sample_period, self.sample_period)
        return self

    @property
    def time_unit(self):
        return self.sample_period_unit

    @property
    def sample_rate(self):
        if len(self.sample_period_unit) > 0 and self.sample_period_unit[0] == "m":
            return 1000.0/self.sample_period
        return 1.0/self.sample_period

    @property
    def space_labels(self):
        try:
            return numpy.array(self.get_space_labels())
        except:
            return numpy.array(self.labels_dimensions.get(self.labels_ordering[2], []))

    @property
    def variables_labels(self):
        return numpy.array(self.labels_dimensions.get(self.labels_ordering[1], []))

    @property
    def number_of_dimensions(self):
        return self.nr_dimensions

    @property
    def squeezed(self):
        return numpy.squeeze(self.data)

    def _check_space_indices(self, list_of_index):
        for index in list_of_index:
            if index < 0 or index > self.data.shape[1]:
                self.logger.error("Some of the given indices are out of space range: [0, %s]",
                                  self.data.shape[1])
                raise IndexError

    def _get_time_unit_for_index(self, time_index):
        return self.start_time + time_index * self.sample_period

    def _get_index_for_time_unit(self, time_unit):
        return int((time_unit - self.start_time) / self.sample_period)

    def get_time_window(self, index_start, index_end, **kwargs):
        if index_start < 0 or index_end > self.data.shape[0]:
            self.logger.error("The time indices are outside time series interval: [%s, %s]" %
                              (0, self.data.shape[0]))
            raise IndexError
        subtime_data = self.data[index_start:index_end, :, :, :]
        if subtime_data.ndim == 3:
            subtime_data = numpy.expand_dims(subtime_data, 0)
        return self.duplicate(data=subtime_data, start_time=self._get_time_unit_for_index(index_start),  **kwargs)

    def get_time_window_by_units(self, unit_start, unit_end, **kwargs):
        end_time = self.end_time
        if unit_start < self.start_time or unit_end > end_time:
            self.logger.error("The time units are outside time series interval: [%s, %s]" %
                              (self.start_time, end_time))
            raise ValueError
        index_start = self._get_index_for_time_unit(unit_start)
        index_end = self._get_index_for_time_unit(unit_end)
        return self.get_time_window(index_start, index_end)

    def decimate_time(self, new_sample_period, **kwargs):
        if new_sample_period % self.sample_period != 0:
            self.logger.error("Cannot decimate time if new time step is not a multiple of the old time step")
            raise ValueError

        index_step = int(new_sample_period / self.sample_period)
        time_data = self.data[::index_step, :, :, :]
        return self.duplicate(data=time_data, sample_period=new_sample_period, **kwargs)

    def get_sample_window(self, index_start, index_end, **kwargs):
        subsample_data = self.data[:, :, :, index_start:index_end]
        if subsample_data.ndim == 3:
            subsample_data = numpy.expand_dims(subsample_data, 3)
        return self.duplicate(data=subsample_data, **kwargs)

    def configure(self):
        super(TimeSeries, self).configure()
        self.configure_time()
        return self


class TimeSeriesBrain(TimeSeries):

    def __init__(self, data, **kwargs):
        super(TimeSeriesBrain, self).__init__(data, **kwargs)

    def get_source(self):
        if self.labels_ordering[1] not in self.labels_dimensions.keys():
            self.logger.error("No state variables are defined for this instance!")
            raise ValueError
        if PossibleVariables.SOURCE.value in self.variables_labels:
            return self.get_state_variable(PossibleVariables.SOURCE.value)

    @property
    def brain_labels(self):
        return self.space_labels


class TimeSeriesRegion(TimeSeriesBrain, TimeSeriesRegionTVB):

    def __init__(self, data, **kwargs):
        super(TimeSeriesRegion, self).__init__(data, **kwargs)
        if self.labels_ordering is None:
            self.labels_ordering = LABELS_ORDERING
            self.labels_ordering[1] = "Region"
        if len(self.title) == 0:
            self.title = "Region Time Series"

    @property
    def region_labels(self):
        return self.space_labels


class TimeSeriesSurface(TimeSeriesBrain, TimeSeriesSurfaceTVB):
    def __init__(self, data, **kwargs):
        super(TimeSeriesSurface, self).__init__(data, **kwargs)
        if self.labels_ordering is None:
            self.labels_ordering = LABELS_ORDERING
            self.labels_ordering[1] = "Vertex"
        if len(self.title) == 0:
            self.title = "Surface Time Series"

    @property
    def surface_labels(self):
        return self.space_labels


class TimeSeriesVolume(TimeSeries, TimeSeriesVolumeTVB):
    def __init__(self, data, **kwargs):
        super(TimeSeriesVolume, self).__init__(data, **kwargs)
        if self.labels_ordering is None:
            self.labels_ordering = ["Time", "X", "Y", "Z"]
        if len(self.title) == 0:
            self.title = "Volume Time Series"

    @property
    def volume_labels(self):
        return self.space_labels


class TimeSeriesSensors(TimeSeries):

    def __init__(self, data, **kwargs):
        super(TimeSeriesSensors, self).__init__(data,  **kwargs)
        self.sensors = kwargs.get("sensors", None)
        if isinstance(self.sensors, Sensors.__class__):
            if self.labels_ordering is None:
                self.labels_ordering = LABELS_ORDERING
                self.labels_ordering[1] = "%s sensor" % self.sensors.sensors_type
        if len(self.title) == 0:
            self.title = "Sensor Time Series"

    @property
    def sensor_labels(self):
        return self.space_labels

    def get_bipolar(self, **kwargs):
        bipolar_labels, bipolar_inds = monopolar_to_bipolar(self.space_labels)
        data = self.data[:, :, bipolar_inds[0]] - self.data[:, :, bipolar_inds[1]]
        bipolar_labels_dimensions = deepcopy(self.labels_dimensions)
        bipolar_labels_dimensions[self.labels_ordering[2]] = list(bipolar_labels)
        return self.duplicate(data=data, labels_dimensions=bipolar_labels_dimensions, **kwargs)


class TimeSeriesEEG(TimeSeriesSensors, TimeSeriesEEGTVB):

    def __init__(self, data, **kwargs):
        super(TimeSeriesEEG, self).__init__(data,  **kwargs)
        if isinstance(self.sensors, Sensors) and not isinstance(self.sensors, SensorsEEG):
            warning("Creating %s with sensors of type %s!" %
                    (self.__class__.__name__, self.sensors.__class__.__name__))
        if len(self.title) == 0:
            self.title = "EEG Time Series"

    @property
    def EEGsensor_labels(self):
        return self.space_labels


class TimeSeriesMEG(TimeSeriesSensors, TimeSeriesMEGTVB):
    def __init__(self, data, **kwargs):
        super(TimeSeriesMEG, self).__init__(data,  **kwargs)
        if isinstance(self.sensors, Sensors) and not isinstance(self.sensors, SensorsMEG):
            warning("Creating %s with sensors of type %s!" %
                    (self.__class__.__name__, self.sensors.__class__.__name__))
        if len(self.title) == 0:
            self.title = "MEG Time Series"

    @property
    def MEGsensor_labels(self):
        return self.space_labels


class TimeSeriesSEEG(TimeSeriesSensors, TimeSeriesSEEGTVB):
    def __init__(self, data, **kwargs):
        super(TimeSeriesSEEG, self).__init__(data,  **kwargs)
        if isinstance(self.sensors, Sensors) and not isinstance(self.sensors, SensorsInternal):
            warning("Creating %s with sensors of type %s!" %
                    (self.__class__.__name__, self.sensors.__class__.__name__))
        if len(self.title) == 0:
            self.title = "SEEG Time Series"

    @property
    def SEEGsensor_labels(self):
        return self.space_labels


if __name__ == "__main__":

    kwargs = {"data": numpy.ones((4, 2, 10, 1)), "start_time": 0.0,
              "labels_dimensions": {LABELS_ORDERING[1]: ["x", "y"]}}
    ts = TimeSeriesRegion(**kwargs)
    tsy = ts.y
    print(tsy.squeezed)
