# coding=utf-8

from tvb_scripts.datatypes.time_series import *

from tvb.config.init.datatypes_registry import REGISTRY, populate_datatypes_registry
from tvb.adapters.datatypes.h5.time_series_h5 import *
from tvb.adapters.datatypes.db.time_series import *


populate_datatypes_registry()


REGISTRY.register_datatype(TimeSeries, TimeSeriesH5, TimeSeriesIndex)
REGISTRY.register_datatype(TimeSeriesRegion, TimeSeriesRegionH5, TimeSeriesRegionIndex)
REGISTRY.register_datatype(TimeSeriesSurface, TimeSeriesSurfaceH5, TimeSeriesSurfaceIndex)
REGISTRY.register_datatype(TimeSeriesVolume, TimeSeriesVolumeH5, TimeSeriesVolumeIndex)
REGISTRY.register_datatype(TimeSeriesEEG, TimeSeriesEEGH5, TimeSeriesEEGIndex)
REGISTRY.register_datatype(TimeSeriesMEG, TimeSeriesMEGH5, TimeSeriesMEGIndex)
REGISTRY.register_datatype(TimeSeriesSEEG, TimeSeriesSEEGH5, TimeSeriesSEEGIndex)
