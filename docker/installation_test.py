# -*- coding: utf-8 -*-
import warnings


# to test all models for TVB only:
try:
    from tvb_multiscale.tests.core import test_models
    test_models()
except Exception as e:
    warnings.warn("Failed to run core test_models with error!:\n%s" % str(e))


# to test all models for TVB-NEST:
try:
    from tvb_multiscale.tests.tvb_nest.test_models import test_models
    test_models()
except Exception as e:
    warnings.warn("Failed to run TVB-NEST test_models with error!:\n%s" % str(e))

# ...or only the faster one:
try:
    from .launch_example import launch_example_nest
    launch_example_nest()
except Exception as e:
    warnings.warn("Failed to run TVB-NEST launch_example_nest with error!:\n%s" % str(e))

# To test the spiking model builder for TVB-NEST:
try:
    from tvb_multiscale.tests.tvb_nest.test_spiking_models_builder_properties_methods \
        import test as test_spiking_model_builder_nest
    test_spiking_model_builder_nest()
except Exception as e:
    warnings.warn("Failed to run TVB-NEST test_spiking_model_builder_nest with error!:\n%s" % str(e))

# to test all models for TVB-ANNarchy:
try:
    from tvb_multiscale.tests.tvb_annarchy.test_models import test_models
    test_models()
except Exception as e:
    warnings.warn("Failed to run TVB-ANNarchy test_models with error!:\n%s" % str(e))

# ...or only the faster one:
try:
    from .launch_example import launch_example_annarchy
    launch_example_annarchy()
except Exception as e:
    warnings.warn("Failed to run TVB-ANNarchy launch_example_annarchy with error!:\n%s" % str(e))

# To test the spiking model builder for TVB-ANNarchy:
try:
    from tvb_multiscale.tests.tvb_annarchy.test_spiking_models_builder_properties_methods \
        import test as test_spiking_model_builder_annarchy
    test_spiking_model_builder_annarchy()
except Exception as e:
    warnings.warn("Failed to run TVB-ANNarchy test_spiking_model_builder_nest with error!:\n%s" % str(e))

# to test all models for TVB-ANNarchy:
try:
    from tvb_multiscale.tests.tvb_netpyne.test_models import test_models
    test_models()
except Exception as e:
    warnings.warn("Failed to run TVB-NetPyNE test_models with error!:\n%s" % str(e))

# ...or only the faster one:
try:
    from .launch_example import launch_example_netpyne
    launch_example_netpyne()
except Exception as e:
    warnings.warn("Failed to run TVB-NetPyNE launch_example_netpyne with error!:\n%s" % str(e))

# To test the spiking model builder for TVB-NetPyNE:
try:
    from tvb_multiscale.tests.tvb_netpyne.test_spiking_models_builder_properties_methods \
        import test as test_spiking_model_builder_netpyne
    test_spiking_model_builder_annarchy()
except Exception as e:
    warnings.warn("Failed to run TVB-NetPyNE test_spiking_model_builder_nest with error!:\n%s" % str(e))


# To test simulator serialization:
try:
    from tvb_multiscale.tests.core import test_simulator_serialization
    test_simulator_serialization()
except Exception as e:
    warnings.warn("Failed to run core test_simulator_serialization with error!:\n%s" % str(e))

# To test TimeSeries datatypes:
try:
    from tvb_multiscale.tests.core.test_time_series_objects import test_time_series_region_object
    test_time_series_region_object()
except Exception as e:
    warnings.warn("Failed to run core test_time_series_region_object with error!:\n%s" % str(e))

# ...and their io:
try:
    from tvb_multiscale.tests.core import test_timeseries_4D, TimeSeriesRegionXarray
    test_timeseries_4D()
except Exception as e:
    warnings.warn("Failed to run core test_timeseries_4D with error!:\n%s" % str(e))
