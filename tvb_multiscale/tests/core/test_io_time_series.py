# coding=utf-8
import os
import shutil

import numpy
import pytest
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion, TimeSeriesDimensions, PossibleVariables
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesRegionXarray
from tvb_multiscale.core.tvb.io.h5_writer import H5Writer, h5
from tvb.datatypes.connectivity import Connectivity

PATH = os.path.join('..', 'TEST_OUTPUT', 'time_series_io')
TEST_CLASSES = [TimeSeriesRegion, TimeSeriesRegionXarray]


def _prepare_dummy_time_series(dim):
    if dim == 1:
        data = numpy.array([1, 2, 3, 4, 5])
    elif dim == 2:
        data = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])[:, numpy.newaxis, :]
    elif dim == 3:
        data = numpy.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]],
                            [[3, 4, 5], [6, 7, 8], [9, 0, 1], [2, 3, 4]],
                            [[5, 6, 7], [8, 9, 0], [1, 2, 3], [4, 5, 6]]])
    else:
        data = numpy.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]],
                             [[3, 4, 5, 6], [7, 8, 9, 0], [1, 2, 3, 4]],
                             [[5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]],
                             [[7, 8, 9, 0], [1, 2, 3, 4], [5, 6, 7, 8]]],
                            [[[9, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 0]],
                             [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]],
                             [[3, 4, 5, 6], [7, 8, 9, 0], [1, 2, 3, 4]],
                             [[5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]]],
                            [[[7, 8, 9, 0], [1, 2, 3, 4], [5, 6, 7, 8]],
                             [[9, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 0]],
                             [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]],
                             [[3, 4, 5, 6], [7, 8, 9, 0], [1, 2, 3, 4]]]]).reshape((3, 3, 4, 4))
    start_time = 0
    sample_period = 0.01
    sample_period_unit = "ms"

    return data, start_time, sample_period, sample_period_unit


def _prepare_connectivity():
    connectivity = Connectivity(weights=numpy.array([[0.0, 2.0, 3.0, 4.0],
                                                     [2.0, 0.0, 2.0, 3.0],
                                                     [3.0, 2.0, 0.0, 1.0],
                                                     [4.0, 3.0, 1.0, 0.0]]),
                                tract_lengths=numpy.array([[0.0, 2.0, 3.0, 4.0],
                                                           [2.0, 0.0, 2.0, 3.0],
                                                           [3.0, 2.0, 0.0, 1.0],
                                                           [4.0, 3.0, 1.0, 0.0]]),
                                region_labels=numpy.array(["a", "b", "c", "d"]),
                                centres=numpy.array([1.0, 2.0, 3.0, 4.0]),
                                areas=numpy.array([1.0, 2.0, 3.0, 4.0]))
    connectivity.configure()
    return connectivity


def teardown_function():
    if os.path.exists(PATH):
        shutil.rmtree(PATH)
    else:
        os.makedirs(PATH)
    return str(PATH)


class TestIOTimeSeries:

    @pytest.mark.parametrize('datatype', TEST_CLASSES)
    def test_timeseries_4D(self, datatype):
        path = teardown_function()
        writer = H5Writer()
        data, start_time, sample_period, sample_period_unit = _prepare_dummy_time_series(4)
        ts = \
            datatype(data=data,
                     labels_dimensions={"Region": ["r1", "r2", "r3", "r4"],
                                        TimeSeriesDimensions.VARIABLES.value: [
                                            PossibleVariables.X.value, PossibleVariables.Y.value,
                                            PossibleVariables.Z.value]},
                     start_time=start_time, sample_period=sample_period,
                     sample_period_unit=sample_period_unit,
                     connectivity=_prepare_connectivity())

        if isinstance(ts, TimeSeriesRegionXarray):
            ts_write = TimeSeriesRegion(data=ts.data, connectivity=ts.connectivity, sample_period=sample_period)
            ts_write.initialze_from_xarray_DataArray(ts._data)
        else:
            ts_write = ts
        path = writer.write_tvb_to_h5(ts_write, path, recursive=True)

        ts_read = TimeSeriesRegion.from_tvb_instance(h5.load(path, with_references=True))
        if isinstance(ts, TimeSeriesRegionXarray):
            ts2 = TimeSeriesRegionXarray(ts_read)
        else:
            ts2 = ts_read

        assert numpy.max(numpy.abs(ts.data - ts2.data)) < 1e-6
        assert numpy.all(ts.variables_labels == ts2.variables_labels)
        assert numpy.all(ts.space_labels == ts2.space_labels)
        assert numpy.abs(ts.start_time - ts2.start_time) < 1e-6
        assert numpy.abs(ts.sample_period - ts2.sample_period) < 1e-6
        assert numpy.all(ts.sample_period_unit == ts2.sample_period_unit)
