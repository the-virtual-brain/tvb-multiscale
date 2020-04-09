# -*- coding: utf-8 -*-

import os

import numpy
from six import string_types
from tvb.datatypes.projections import ProjectionMatrix
from tvb.datatypes.region_mapping import RegionMapping, RegionVolumeMapping
from tvb.datatypes.structural import StructuralMRI
from tvb.simulator.plot.utils.log_error_utils import raise_value_error

from tvb_scripts.datatypes.connectivity import ConnectivityH5Field
from tvb_scripts.datatypes.sensors import SensorsH5Field, SensorTypes, Sensors
from tvb_scripts.datatypes.surface import SurfaceH5Field, Surface
from tvb_scripts.datatypes.time_series import TimeSeries
from tvb_scripts.datatypes.time_series_xarray import TimeSeries as TimeSeriesXarray
from tvb_scripts.io.h5_writer_base import H5WriterBase
from tvb_scripts.utils.file_utils import write_metadata

KEY_NODES = "Number_of_nodes"
KEY_SENSORS = "Number_of_sensors"
KEY_MAX = "Max_value"
KEY_MIN = "Min_value"
KEY_CHANNELS = "Number_of_channels"
KEY_SV = "Number_of_state_variables"
KEY_STEPS = "Number_of_steps"
KEY_SAMPLING = "Sampling_period"
KEY_START = "Start_time"


class H5Writer(H5WriterBase):

    # TODO: write variants.
    def write_connectivity(self, connectivity, path=None, h5_file=None, close_file=True):
        """
        :param connectivity: Connectivity object to be written in H5
        :param path: H5 path to be written
        """
        h5_file, path = self._open_file("Connectivity", path, h5_file)

        h5_file.create_dataset(ConnectivityH5Field.WEIGHTS, data=connectivity.weights)
        h5_file.create_dataset(ConnectivityH5Field.TRACTS, data=connectivity.tract_lengths)
        h5_file.create_dataset(ConnectivityH5Field.CENTERS, data=connectivity.centres)
        h5_file.create_dataset(ConnectivityH5Field.REGION_LABELS,
                               data=numpy.array([numpy.string_(label) for label in connectivity.region_labels]))
        h5_file.create_dataset(ConnectivityH5Field.ORIENTATIONS, data=connectivity.orientations)
        h5_file.create_dataset(ConnectivityH5Field.HEMISPHERES, data=connectivity.hemispheres)
        h5_file.create_dataset(ConnectivityH5Field.AREAS, data=connectivity.areas)

        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("Connectivity"))
        h5_file.attrs.create("Number_of_regions", numpy.string_(connectivity.number_of_regions))

        if hasattr(connectivity, "normalized_weights"):
            h5_file.create_dataset("normalized_weights/" + ConnectivityH5Field.WEIGHTS,
                                   data=connectivity.normalized_weights)

        self._close_file(h5_file, close_file)

        self._log_success("Connectivity", path)

        return h5_file

    def write_sensors(self, sensors, projection=None, path=None, h5_file=None, close_file=True):
        """
        :param sensors: Sensors object to write in H5
        :param path: H5 path to be written
        """
        if isinstance(sensors, Sensors):
            h5_file, path = self._open_file("Sensors", path, h5_file)

            h5_file.create_dataset(SensorsH5Field.LABELS,
                                   data=numpy.array([numpy.string_(label) for label in sensors.labels]))
            h5_file.create_dataset(SensorsH5Field.LOCATIONS, data=sensors.locations)
            h5_file.create_dataset(SensorsH5Field.ORIENTATIONS, data=sensors.orientations)

            if sensors.sensors_type in [SensorTypes.TYPE_SEEG.value, SensorTypes.TYPE_INTERNAL.value]:
                h5_file.create_dataset("ElectrodeLabels",
                                       data=numpy.array([numpy.string_(label) for label in sensors.channel_labels]))
                h5_file.create_dataset("ElectrodeIndices", data=sensors.channel_inds)

            if isinstance(projection, ProjectionMatrix):
                projection = projection.projection_data
            elif not isinstance(projection, numpy.ndarray):
                projection = numpy.array([])
            projection_dataset = h5_file.create_dataset(SensorsH5Field.PROJECTION_MATRIX, data=projection)
            if projection.size > 0:
                projection_dataset.attrs.create("Max", numpy.string_(projection.max()))
                projection_dataset.attrs.create("Min", numpy.string_(projection.min()))

            h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("Sensors"))
            h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, numpy.string_(sensors.__class__.__name__))
            h5_file.attrs.create("Number_of_sensors", numpy.string_(sensors.number_of_sensors))
            h5_file.attrs.create("Sensors_subtype", numpy.string_(sensors.sensors_type))
            h5_file.attrs.create("name", numpy.string_(sensors.name))

            self._close_file(h5_file, close_file)

            self._log_success("Sensors", path)

        return h5_file

    def write_surface(self, surface, path=None, h5_file=None, close_file=True):
        """
        :param surface: Surface object to write in H5
        :param path: H5 path to be written
        """
        if isinstance(surface, Surface):
            h5_file, path = self._open_file("Surface", path, h5_file)

            h5_file.create_dataset(SurfaceH5Field.VERTICES, data=surface.vertices)
            h5_file.create_dataset(SurfaceH5Field.TRIANGLES, data=surface.triangles)
            h5_file.create_dataset(SurfaceH5Field.VERTEX_NORMALS, data=surface.vertex_normals)
            h5_file.create_dataset(SurfaceH5Field.TRIANGLE_NORMALS, data=surface.triangle_normals)

            h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("Surface"))
            h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, numpy.string_(surface.__class__.__name__))
            h5_file.attrs.create("Surface_subtype", numpy.string_(surface.surface_type))
            h5_file.attrs.create("Number_of_triangles", surface.triangles.shape[0])
            h5_file.attrs.create("Number_of_vertices", surface.vertices.shape[0])
            if hasattr(surface, "vox2ras") and surface.vox2ras is not None:
                h5_file.create_dataset("Voxel_to_ras_matrix", data=surface.vox2ras)

            self._close_file(h5_file, close_file)

            self._log_success("Surface", path)

        return h5_file

    def write_region_mapping(self, region_mapping, n_regions, subtype="Cortical",
                             path=None, h5_file=None, close_file=True):
        """
            :param region_mapping: region_mapping array to write in H5
            :param path: H5 path to be written
        """
        if isinstance(region_mapping, RegionMapping):
            h5_file, path = self._open_file("%s RegionMapping" % subtype, path, h5_file)

            h5_file.create_dataset("data", data=region_mapping.array_data)

            data_length = len(region_mapping.array_data)
            h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("RegionMapping"))
            h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, numpy.string_(region_mapping.__class__.__name__))
            h5_file.attrs.create("RegionMapping_subtype", numpy.string_(subtype))
            h5_file.attrs.create("Connectivity_parcel", numpy.string_("Connectivity-%d" % n_regions))
            h5_file.attrs.create("Surface_parcel", numpy.string_("Surface-%s-%d" % (subtype.capitalize(), data_length)))
            h5_file.attrs.create("Length_data", data_length)
            self._close_file(h5_file, close_file)
            self._log_success("%s RegionMapping" % subtype, path)

        return h5_file

    def write_volume(self, volume, vol_type, n_regions,
                     path=None, h5_file=None, close_file=True):
        """
            :param t1: t1 array to write in H5
            :param path: H5 path to be written
        """
        if isinstance(volume, (RegionVolumeMapping, StructuralMRI)):
            shape = volume.array_data.shape
            if len(shape) < 3:
                shape = (0, 0, 0)
            h5_file, path = self._open_file("%s VolumeMapping" % vol_type, path, h5_file)
            h5_file.create_dataset("data", data=volume.array_data)
            h5_file.attrs.create("Connectivity_parcel", numpy.string_("Connectivity-%d" % n_regions))
            h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("VolumeData"))
            h5_file.attrs.create("VolumeMapping_subtype", numpy.string_(volume.__class__.__name__))
            h5_file.attrs.create("Length_x", numpy.string_(shape[0]))
            h5_file.attrs.create("Length_y", numpy.string_(shape[1]))
            h5_file.attrs.create("Length_z", numpy.string_(shape[2]))
            h5_file.attrs.create("Type", numpy.string_(vol_type.upper()))
            self._close_file(h5_file, close_file)
            self._log_success("%s VolumeMapping" % vol_type, path)

        return h5_file

    def write_t1(self, t1, n_regions,
                 path=None, h5_file=None, close_file=True):
        if isinstance(t1, StructuralMRI):
            h5_file = self.write_volume(t1, "STRUCTURAL", n_regions, path, h5_file, close_file)

        return h5_file

    def write_volume_mapping(self, volume_mapping, n_regions,
                             path=None, h5_file=None, close_file=True):
        if isinstance(volume_mapping, RegionVolumeMapping):
            h5_file = self.write_volume(volume_mapping, "MAPPING", n_regions, path, h5_file, close_file)

        return h5_file

    def write_head(self, head, path=None):
        """
        :param head: Head object to be written
        :param path: path to datatypes folder
        """
        if path is None:
            path = head.folderpath
        self.logger.info("Starting to write Head folder: %s" % path)

        if not (os.path.isdir(path)):
            os.mkdir(path)

        n_regions = head.connectivity.number_of_regions
        self.write_connectivity(head.connectivity, os.path.join(path, "Connectivity.h5"))
        self.write_surface(head.cortical_surface, os.path.join(path, "CorticalSurface.h5"))
        self.write_region_mapping(head.cortical_region_mapping, n_regions, "Cortical",
                                  os.path.join(path, "CorticalRegionMapping.h5"))
        if head.subcortical_surface is not None:
            self.write_surface(head.subcortical_surface, os.path.join(path, "SubcorticalSurface.h5"))
            self.write_region_mapping(head.subcortical_region_mapping, n_regions, "Subcortical",
                                      os.path.join(path, "SubcorticalRegionMapping.h5"))
        self.write_volume_mapping(head.region_volume_mapping, n_regions, os.path.join(path, "VolumeMapping.h5"))
        self.write_t1(head.t1, n_regions, os.path.join(path, "StructuralMRI.h5"))
        for s_type, sensors_set in head.sensors.items():
            for sensor, projection in sensors_set.items():
                self.write_sensors(sensor, projection.
                                   os.path.join(path, "%s.h5" % sensor.name.replace(" ", "")))

        self._log_success("Head folder", path)

    def write_dictionary(self, dictionary, path, h5_file=None, close_file=True):
        """
        :param dictionary: dictionary to write in H5
        :param path: H5 path to be written
        """
        h5_file, path = self._open_file("Dictionary", path, h5_file)
        self._write_dictionary_to_group(dictionary, h5_file)
        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("Dictionary"))
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, numpy.string_(dictionary.__class__.__name__))
        self._close_file(h5_file, close_file)
        self._log_success("Dictionary", path)
        return h5_file

    def write_list_of_dictionaries(self, list_of_dicts, path=None, h5_file=None, close_file=True):
        h5_file, path = self._open_file("List of dictionaries", path, h5_file)
        for idict, dictionary in enumerate(list_of_dicts):
            idict_str = str(idict)
            h5_file.create_group(idict_str)
            self._write_dictionary_to_group(dictionary, h5_file[idict_str])
        h5_file.attrs.create(self.H5_TYPE_ATTRIBUTE, numpy.string_("List of dictionaries"))
        h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, numpy.string_("list"))
        self._close_file(h5_file, close_file)
        self._log_success("List of dictionaries", path)
        return h5_file

    def write_ts(self, raw_data, sampling_period, path=None, h5_file=None, close_file=True):
        h5_file, path = self._open_file("TimeSeries", path, h5_file)
        write_metadata({self.H5_TYPE_ATTRIBUTE: "TimeSeries"}, h5_file,
                       self.H5_DATE_ATTRIBUTE, self.H5_VERSION_ATTRIBUTE)
        if isinstance(raw_data, (TimeSeries, TimeSeriesXarray)):
            h5_file.attrs.create(self.H5_SUBTYPE_ATTRIBUTE, numpy.string_(raw_data.__class__.__name__))
            if len(raw_data.shape) == 4 and str(raw_data.data.dtype)[0] == "f":
                h5_file.create_dataset("data", data=raw_data.data)
                h5_file.create_dataset("time", data=raw_data.time)
                try:
                    h5_file.create_dataset("dimensions_labels",
                                           data=numpy.array([numpy.string_(label)
                                                             for label in raw_data.labels_ordering]))
                except:
                    pass
                for i_dim, dim_label in enumerate(raw_data.labels_ordering[1:]):
                    try:
                        labels = raw_data.labels_dimensions[dim_label]
                        if isinstance(labels[0], string_types):
                            h5_file.create_dataset("%s" % dim_label,
                                                   data=numpy.array([numpy.string_(label) for label in labels]))
                        else:
                            h5_file.create_dataset("%s" % dim_label, data=labels)
                    except:
                        pass
                h5_file.attrs.create("sample_period_unit", numpy.string_(raw_data.sample_period_unit))
                h5_file.attrs.create("title", numpy.string_(raw_data.title))
                write_metadata({KEY_MAX: raw_data.data.max(), KEY_MIN: raw_data.data.min(),
                                KEY_STEPS: raw_data.data.shape[0], KEY_CHANNELS: raw_data.data.shape[1],
                                KEY_SV: 1, KEY_SAMPLING: raw_data.sample_period,
                                KEY_START: raw_data.start_time}, h5_file,
                               self.H5_DATE_ATTRIBUTE, self.H5_VERSION_ATTRIBUTE, "data")
                # If this is a TimeSeriesRegion try to write all of the structures below:
                if hasattr(raw_data, "connectivity"):
                    h5_file.create_group("connectivity")
                    self.write_connectivity(raw_data.connectivity,
                                            path=path, h5_file=h5_file["connectivity"], close_file=False)
                if hasattr(raw_data, "region_mapping"):
                    h5_file.create_group("region_mapping")
                    self.write_region_mapping(raw_data.region_mapping,
                                              raw_data.connectivity.number_of_regions,
                                              subtype="Cortical", path=path,
                                              h5_file=h5_file["region_mapping"], close_file=False)
                if hasattr(raw_data, "region_mapping_volume"):
                    h5_file.create_group("volume_mapping")
                    self.write_volume_mapping(raw_data.region_mapping_volume,
                                              raw_data.connectivity.number_of_regions,
                                              path=path, h5_file=h5_file["volume_mapping"], close_file=False)

                # If this is a TimeSeriesSensors try to write the sensors:
                if hasattr(raw_data, "sensors"):
                    h5_file.create_group("sensors")
                    self.write_sensors(raw_data.sensors, path=path, h5_file=h5_file["sensors"], close_file=False)

                # If this is a TimeSeriesSurface try to write the surface:
                if hasattr(raw_data, "surface"):
                    h5_file.create_group("surface")
                    self.write_surface(raw_data.surface, path=path, h5_file=h5_file["surface"], close_file=False)

                # If this is a TimeSeriesVolume try to write the volume:
                if hasattr(raw_data, "volume"):
                    h5_file.create_group("volume")
                    self.write_volume(raw_data.volume, path=path, h5_file=h5_file["volume"], close_file=False)

            else:
                raise_value_error("Invalid TS data. 4D (time, nodes) numpy.ndarray of floats expected")
        else:
            h5_file.attrs.create("time_series_type", "TimeSeries")
            if isinstance(raw_data, dict):
                for data in raw_data:
                    if len(raw_data[data].shape) == 2 and str(raw_data[data].dtype)[0] == "f":
                        h5_file.create_dataset(data, data=raw_data[data])
                        write_metadata({KEY_MAX: raw_data[data].max(), KEY_MIN: raw_data[data].min(),
                                        KEY_STEPS: raw_data[data].shape[0], KEY_CHANNELS: raw_data[data].shape[1],
                                        KEY_SV: 1, KEY_SAMPLING: sampling_period, KEY_START: 0.0}, h5_file,
                                       self.H5_DATE_ATTRIBUTE, self.H5_VERSION_ATTRIBUTE, data)
                    else:
                        raise_value_error("Invalid TS data. 2D (time, nodes) numpy.ndarray of floats expected")
            elif isinstance(raw_data, numpy.ndarray):
                if len(raw_data.shape) != 2 and str(raw_data.dtype)[0] != "f":
                    h5_file.create_dataset("data", data=raw_data)
                    write_metadata({KEY_MAX: raw_data.max(), KEY_MIN: raw_data.min(), KEY_STEPS: raw_data.shape[0],
                                    KEY_CHANNELS: raw_data.shape[1], KEY_SV: 1, KEY_SAMPLING: sampling_period,
                                    KEY_START: 0.0}, h5_file, self.H5_DATE_ATTRIBUTE, self.H5_VERSION_ATTRIBUTE, "data")
                else:
                    raise_value_error("Invalid TS data. 2D (time, nodes) numpy.ndarray of floats expected")
            else:
                raise_value_error("Invalid TS data. TimeSeries, "
                                  "dictionary or 2D (time, nodes) numpy.ndarray of floats expected")
        self._close_file(h5_file, close_file)
        self._log_success("TimeSeries", path)
        return h5_file

    def write_timeseries(self, timeseries, path=None, h5_file=None, close_file=True):
        return self.write_ts(timeseries, timeseries.sample_period, path, h5_file, close_file)
