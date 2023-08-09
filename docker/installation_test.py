# -*- coding: utf-8 -*-

# to test all models for TVB only:
try:
    from tvb_multiscale.tests.core import test_models
    test_models()
except:
    print("Failed to run core test_models!")


# to test all models for TVB-NEST:
try:
    from tvb_multiscale.tests.tvb_nest.test_models import test_models
    test_models()
except:
    print("Failed to run tvb_nest test_models!")

# ...or only the faster one:
try:
    from .launch_example import launch_example_nest
    launch_example_nest()
except:
    print("Failed to run launch_example_nest!")

# To test the spiking model builder for TVB-NEST:
try:
    from tvb_multiscale.tests.tvb_nest.test_spiking_models_builder_properties_methods import test as test_spiking_model_builder_nest
    test_spiking_model_builder_nest()
except:
    print("Failed to run tvb_nest test_spiking_model_builder_nest!")

# to test all models for TVB-ANNarchy:
try:
    from tvb_multiscale.tests.tvb_annarchy.test_models import test_models
    test_models()
except:
    print("Failed to run tvb_annarchy test_models!")

# ...or only the faster one:
try:
    from .launch_example import launch_example_annarchy
    launch_example_annarchy()
except:
    print("Failed to run launch_example_annarchy!")

# To test the spiking model builder for TVB-ANNarchy:
try:
    from tvb_multiscale.tests.tvb_annarchy.test_spiking_models_builder_properties_methods import test as test_spiking_model_builder_nest
    test_spiking_model_builder_nest()
except:
    print("Failed to run tvb_annarchy test_spiking_model_builder_nest!")

# To test simulator serialization:
try:
    from tvb_multiscale.tests.core import test_simulator_serialization
    test_simulator_serialization()
except:
    print("Failed to run test_simulator_serialization!")

# To test TimeSeries datatypes:
try:
    from tvb_multiscale.tests.core.test_time_series_objects import test_time_series_region_object
    test_time_series_region_object()
except:
    print("Failed to run test_time_series_region_object!")

# ...and their io:
try:
    from tvb_multiscale.tests.core import test_timeseries_4D, TimeSeriesRegionXarray
    test_timeseries_4D()
except:
    print("Failed to run tests.core.test_io_time_series.test_timeseries_4D for TimeSeriesRegionXarray!")
