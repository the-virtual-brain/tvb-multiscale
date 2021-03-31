# -*- coding: utf-8 -*-

# to test all models for TVB only:
from tests.core.test_models import test_models
test_models()


# to test all models for TVB-NEST:
from tests.tvb_nest.test_models import test_models
test_models()

# ...or only the faster one:
from .launch_example import launch_example
launch_example()

# To test the spiking model builder for TVB-NEST:
from tests.tvb_nest.test_spiking_models_builder_propertes_methods import test as test_spiking_model_builder_nest
test_spiking_model_builder_nest()


# to test all models for TVB-ANNarchy:
from tests.tvb_annarchy.test_models import test_models
test_models()

# ...or only the faster one:
from .launch_example import launch_example_annarchy
launch_example_annarchy()

# To test the spiking model builder for TVB-ANNarchy:
from tests.tvb_annarchy.test_spiking_models_builder_propertes_methods import test as test_spiking_model_builder_nest
test_spiking_model_builder_nest()


# To test simulator serialization:
from tests.core.test_simulator_serialization import test_simulator_serialization
test_simulator_serialization()

# To test TimeSeries datatypes:
from tests.core.test_time_series import TimeSeries, TimeSeriesXarray, \
    test_timeseries_data_access, test_timeseries_1D_definition, \
    test_timeseries_2D, test_timeseries_3D, test_timeseries_4D

test_timeseries_1D_definition(TimeSeriesXarray)
test_timeseries_2D(TimeSeriesXarray)
test_timeseries_3D(TimeSeriesXarray)
test_timeseries_data_access(TimeSeriesXarray)
test_timeseries_4D(TimeSeriesXarray)

test_timeseries_1D_definition(TimeSeries)
test_timeseries_2D(TimeSeries)
test_timeseries_3D(TimeSeries)
test_timeseries_data_access(TimeSeries)
test_timeseries_4D(TimeSeries)

from tests.core.test_time_series_objects import test_time_series_region_object
test_time_series_region_object()

# ...and their io:
from tests.core.test_io_time_series import test_timeseries_4D, TimeSeriesRegionXarray
test_timeseries_4D()
test_timeseries_4D(TimeSeriesRegionXarray)
