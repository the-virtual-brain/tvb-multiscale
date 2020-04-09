# coding=utf-8
import numpy
import pytest
from tvb_scripts.datatypes.time_series import TimeSeries, TimeSeriesDimensions, PossibleVariables


class TestTimeseries(object):
    data_1D = numpy.array([1, 2, 3, 4, 5])
    data_2D = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    data_3D = numpy.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]],
                           [[3, 4, 5], [6, 7, 8], [9, 0, 1], [2, 3, 4]],
                           [[5, 6, 7], [8, 9, 0], [1, 2, 3], [4, 5, 6]]])

    data_4D = numpy.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]],
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
                            [[3, 4, 5, 6], [7, 8, 9, 0], [1, 2, 3, 4]]]])
    start_time = 0
    sample_period = 0.01
    sample_period_unit = "ms"

    def test_timeseries_1D_definition(self):
        with pytest.raises(ValueError):
            TimeSeries(self.data_1D, labels_dimensions={}, start_time=0,
                       sample_period=0.01, sample_period_unit="ms")

    def test_timeseries_2D(self):
        ts_from_2D = TimeSeries(self.data_2D, labels_dimensions={TimeSeriesDimensions.SPACE.value:
                                                                  numpy.array(["r1", "r2", "r3"])},
                                start_time=self.start_time, sample_period=self.sample_period,
                                sample_period_unit=self.sample_period_unit)
        assert ts_from_2D.data.ndim == 4
        assert ts_from_2D.data.shape[2] == ts_from_2D.data.shape[3] == 1
        assert ts_from_2D.data.shape == (3, 3, 1, 1)

        assert ts_from_2D.end_time == 0.02
        assert all(ts_from_2D.time == numpy.array([0, 0.01, 0.02]))

        with pytest.raises(KeyError):
            ts_from_2D.get_variables_by_label("")

        ts_r2r3 = ts_from_2D.get_subspace_by_labels(["r2", "r3"])
        assert ts_r2r3.data.ndim == 4
        assert ts_r2r3.data.shape == (3, 2, 1, 1)

        ts_r2 = ts_from_2D.get_subspace_by_labels(["r2"])
        assert ts_r2.data.ndim == 4
        assert ts_r2.data.shape == (3, 1, 1, 1)
        assert ts_r2.labels_dimensions[TimeSeriesDimensions.SPACE.value] == numpy.array(["r2"])
        assert ts_r2.get_subspace_by_labels(["r2"]).labels_dimensions[TimeSeriesDimensions.SPACE.value] == \
               numpy.array(["r2"])

        with pytest.raises(IndexError):
            ts_r2.get_subspace_by_labels(["r1"])

        ts_r2r3_idx = ts_from_2D.get_subspace_by_index([1, 2])
        assert ts_r2r3_idx.data.ndim == 4
        assert ts_r2r3_idx.data.shape == (3, 2, 1, 1)

        ts_r2_idx = ts_r2r3_idx.get_subspace_by_index([0])
        assert ts_r2_idx.data.ndim == 4
        assert ts_r2_idx.data.shape == (3, 1, 1, 1)
        assert ts_r2_idx.labels_dimensions[TimeSeriesDimensions.SPACE.value] == numpy.array(["r2"])
        assert ts_r2_idx.get_subspace_by_index([0]).labels_dimensions[TimeSeriesDimensions.SPACE.value] == \
               numpy.array(["r2"])

        with pytest.raises(IndexError):
            ts_r2_idx.get_subspace_by_index([2])

        assert ts_r2r3_idx.data.all() == ts_r2r3.data.all()
        assert all(ts_r2_idx.data == ts_r2.data)

        ts_time_window = ts_from_2D.get_time_window(1, 2)
        assert ts_time_window.data.ndim == 4
        assert ts_time_window.data.shape == (1, 3, 1, 1)
        assert numpy.array_equal(ts_time_window.labels_dimensions[TimeSeriesDimensions.SPACE.value],
                                 numpy.array(["r1", "r2", "r3"]))
        assert ts_time_window.start_time == 0.01

        ts_time_window_units = ts_from_2D.get_time_window_by_units(0.01, 0.02)
        assert ts_time_window_units.data.ndim == 4
        assert ts_time_window_units.data.shape == (1, 3, 1, 1)
        assert numpy.array_equal(ts_time_window_units.labels_dimensions[TimeSeriesDimensions.SPACE.value],
                                 numpy.array(["r1", "r2", "r3"]))
        assert ts_time_window_units.start_time == 0.01

        with pytest.raises(IndexError):
            ts_from_2D.get_time_window(2, 4)

        with pytest.raises(ValueError):
            ts_from_2D.get_time_window_by_units(0, 0.025)

        with pytest.raises(AttributeError):
            ts_from_2D.lfp

    def test_timeseries_3D(self):
        ts_3D = TimeSeries(self.data_3D,
                           labels_dimensions={TimeSeriesDimensions.SPACE.value: numpy.array([]),
                                             TimeSeriesDimensions.VARIABLES.value: numpy.array([])},
                           start_time=self.start_time, sample_period=self.sample_period,
                           sample_period_unit=self.sample_period_unit)
        assert ts_3D.data.ndim == 4
        assert ts_3D.data.shape[3] == 1

    def test_timeseries_data_access(self):
        ts = TimeSeries(self.data_3D,
                        labels_dimensions={TimeSeriesDimensions.SPACE.value: numpy.array(["r1", "r2", "r3", "r4"]),
                                           TimeSeriesDimensions.VARIABLES.value: numpy.array(["sv1", "sv2", "sv3"])},
                        start_time=self.start_time, sample_period=self.sample_period,
                        sample_period_unit=self.sample_period_unit)
        assert isinstance(ts.r1, TimeSeries)
        assert ts.r1.data.shape == (3, 1, 3, 1)

        assert isinstance(ts.sv1, TimeSeries)
        assert ts.sv1.data.shape == (3, 4, 1, 1)

        with pytest.raises(AttributeError):
            ts.r9

        with pytest.raises(AttributeError):
            ts.sv0

        assert ts[:, :, :, :].shape == ts.data.shape
        assert ts[1:, :, :, :].shape == ts.data[1:, :, :, :].shape
        assert ts[1:2, :, :, :].shape == ts.data[1:2, :, :, :].shape
        assert ts[1, :, :, :].shape == ts.data[1, :, :, :].shape

        assert ts[:, 1:, :, :].shape == ts.data[:, 1:, :, :].shape
        assert ts[:, :1, :, :].shape == ts.data[:, :1, :, :].shape
        assert ts[:, 1:3, :, :].shape == ts.data[:, 1:3, :, :].shape
        assert ts[:, 1, :, :].shape == ts.data[:, 1, :, :].shape

        assert ts[:, "r2":, :, :].shape == ts.data[:, 1:, :, :].shape
        assert ts[:, :"r2", :, :].shape == ts.data[:, :1, :, :].shape
        assert ts[:, "r2", :, :].shape == ts.data[:, 1, :, :].shape
        assert ts[:, "r2":"r4", :, :].shape == ts.data[:, 1:3, :, :].shape

        assert ts[1:2, "r2":"r4", :, :].shape == ts.data[1:2, 1:3, :, :].shape
        assert ts[1, "r2":"r4", :, :].shape == ts.data[1, 1:3, :, :].shape

        assert ts[:, :, 1:, :].shape == ts.data[:, :, 1:, :].shape
        assert ts[:, :, :1, :].shape == ts.data[:, :, :1, :].shape
        assert ts[:, :, 0:2, :].shape == ts.data[:, :, 0:2, :].shape
        assert ts[:, :, 2, :].shape == ts.data[:, :, 2, :].shape

        assert ts[:, :, "sv2":, :].shape == ts.data[:, :, 1:, :].shape
        assert ts[:, :, :"sv2", :].shape == ts.data[:, :, :1, :].shape
        assert ts[:, :, "sv1":"sv3", :].shape == ts.data[:, :, 0:2, :].shape
        assert ts[:, :, "sv3", :].shape == ts.data[:, :, 2, :].shape

        assert ts[1:2, :, "sv2":, :].shape == ts.data[1:2, :, 1:, :].shape
        assert ts[1:2, :, :"sv2", :].shape == ts.data[1:2, :, :1, :].shape
        assert ts[1:2, :, "sv1":"sv3", :].shape == ts.data[1:2, :, 0:2, :].shape
        assert ts[1:2, :, "sv3", :].shape == ts.data[1:2, :, 2, :].shape
        assert ts[2, :, "sv3", :].shape == ts.data[2, :, 2, :].shape

        assert ts[2, 0:3, "sv3", :].shape == ts.data[2, 0:3, 2, :].shape
        assert ts[2, "r1":"r4", "sv3", :].shape == ts.data[2, 0:3, 2, :].shape
        assert ts[0:2, "r1":"r4", "sv3", :].shape == ts.data[0:2, 0:3, 2, :].shape
        assert ts[0:2, :"r2", "sv3", :].shape == ts.data[0:2, :1, 2, :].shape
        assert ts[0:2, "r2":, "sv3", :].shape == ts.data[0:2, 1:, 2, :].shape
        assert ts[0:2, "r1", "sv3", :].shape == ts.data[0:2, 0, 2, :].shape

        assert all(ts[0:2, "r1", "sv3", :] == ts.data[0:2, 0, 2, :])
        assert ts[0:2, "r1":"r3", "sv3", :].all() == ts.data[0:2, 0:2, 2, :].all()
        assert ts[0:2, "r1":"r3", :"sv2", :].all() == ts.data[0:2, 0:2, :1, :].all()
        assert ts[2, "r1":"r3", :"sv2", :].all() == ts.data[2, 0:2, :1, :].all()
        assert ts[2, "r3", "sv2", :].all() == ts.data[2, 2, 1, :].all()
        assert ts[2, "r3", "sv2", 0] == ts.data[2, 2, 1, 0]

        # IndexError because of [0][0] on numpy array in TS._get_index_of_state_variable
        with pytest.raises(IndexError):
            ts[:, :, "sv0", :]

        with pytest.raises(IndexError):
            ts[0, "r1":"r5", :, :]

        with pytest.raises(IndexError):
            ts[0, :, 10, :]

        with pytest.raises(AttributeError):
            ts.lfp

    def test_timeseries_4D(self):
        ts_4D = TimeSeries(self.data_4D,
                           labels_dimensions={TimeSeriesDimensions.SPACE.value: numpy.array(["r1", "r2", "r3", "r4"]),
                                              TimeSeriesDimensions.VARIABLES.value: numpy.array([
                                                 PossibleVariables.X1.value, PossibleVariables.X2.value,
                                                 "sv3"])},
                           start_time=self.start_time, sample_period=self.sample_period,
                           sample_period_unit=self.sample_period_unit)
        assert ts_4D.data.shape == (3, 4, 3, 4)
        assert ts_4D.x1.data.shape == (3, 4, 1, 4)
