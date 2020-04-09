# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
from tvb.datatypes.head import Head
from tvb.datatypes.projections import ProjectionMatrix
from tvb.datatypes.region_mapping import RegionMapping, RegionVolumeMapping
from tvb.datatypes.structural import StructuralMRI

from tvb_scripts.datatypes.connectivity import Connectivity, ConnectivityH5Field
from tvb_scripts.datatypes.sensors import \
    Sensors, SensorsDict, SensorsH5Field, SensorTypesToProjectionDict
from tvb_scripts.datatypes.surface import Surface, SurfaceDict, SurfaceH5Field
from tvb_scripts.datatypes.time_series import TimeSeriesDict, TimeSeries
from tvb_scripts.datatypes.time_series_xarray import TimeSeries as XarrayTimeSeries
from tvb_scripts.datatypes.time_series_xarray import TimeSeriesDict as XarrayTimeSeriesDict
from tvb_scripts.io.h5_reader_base import *

RegionMappingDict = {
    RegionMapping.__name__: RegionMapping,
    RegionVolumeMapping.__name__: RegionVolumeMapping,
}


class H5Reader(H5ReaderBase):
    connectivity_filename = "Connectivity.h5"
    cortical_surface_filename = "CorticalSurface.h5"
    subcortical_surface_filename = "SubcorticalSurface.h5"
    cortical_region_mapping_filename = "RegionMapping.h5"
    subcortical_region_mapping_filename = "RegionMappingSubcortical.h5"
    volume_mapping_filename = "VolumeMapping.h5"
    structural_mri_filename = "StructuralMRI.h5"
    sensors_filename_prefix = "Sensors"
    sensors_filename_separator = "_"

    def read_connectivity(self, path=None, h5_file=None, close_file=True):
        """
        :param path: Path towards a custom Connectivity H5 file
        :return: Connectivity object
        """
        h5_file = self._open_file("Connectivity", path, h5_file)

        weights = h5_file[ConnectivityH5Field.WEIGHTS][()]
        try:
            tract_lengths = h5_file[ConnectivityH5Field.TRACTS][()]
        except:
            tract_lengths = np.array([])
        try:
            region_centres = h5_file[ConnectivityH5Field.CENTERS][()]
        except:
            region_centres = np.array([])
        try:
            region_labels = np.array([label.decode("UTF-8")
                                      for label in h5_file[ConnectivityH5Field.REGION_LABELS][()]])
        except:
            region_labels = np.array([])
        try:
            orientations = h5_file[ConnectivityH5Field.ORIENTATIONS][()]
        except:
            orientations = np.array([])
        try:
            hemispheres = h5_file[ConnectivityH5Field.HEMISPHERES][()]
        except:
            hemispheres = np.array([])
        try:
            areas = h5_file[ConnectivityH5Field.AREAS][()]
        except:
            areas = np.array([])

        self._close_file(h5_file, close_file)

        conn = Connectivity(filepath=path, weights=weights, tract_lengths=tract_lengths,
                            region_labels=region_labels, centres=region_centres,
                            hemispheres=hemispheres, orientations=orientations, areas=areas)
        conn.configure()

        self._log_success("Connectivity", path)

        return conn

    def read_surface(self, path=None, h5_file=None, close_file=True):
        """
        :param path: Path towards a custom Surface H5 file
        :return: Surface object
        """
        h5_file = self._open_file("Surface", path, h5_file)

        vertices = h5_file[SurfaceH5Field.VERTICES][()]
        triangles = h5_file[SurfaceH5Field.TRIANGLES][()]
        try:
            vertex_normals = h5_file[SurfaceH5Field.VERTEX_NORMALS][()]
        except:
            vertex_normals = np.array([])
        try:
            triangle_normals = h5_file[SurfaceH5Field.TRIANGLE_NORMALS][()]
        except:
            triangle_normals = np.array([])
        try:
            vox2ras = h5_file[SurfaceH5Field.VOX2RAS][()]
        except:
            vox2ras = None

        surface_subtype = (h5_file.attrs.get(self.H5_TYPE_ATTRIBUTE, "")).decode("UTF-8")
        s_class_name = (h5_file.attrs.get(self.H5_SUBTYPE_ATTRIBUTE, "")).decode("UTF-8")
        self._close_file(h5_file, close_file)

        surface = \
            SurfaceDict.get(s_class_name, Surface)(
                filepath=path, surface_subtype=surface_subtype,
                vertices=vertices, triangles=triangles,
                vertex_normals=vertex_normals, triangle_normals=triangle_normals, vox2ras=vox2ras)
        surface.configure()

        self._log_success("Surface", path)

        return surface

    def read_sensors(self, path=None):
        """
        :param path: Path towards a custom datatypes folder
        :return: 3 lists with all sensors from Path by type
        """
        sensors = OrderedDict()

        self.logger.info("Starting to read all Sensors from: %s" % path)

        all_head_files = os.listdir(path)
        for head_file in all_head_files:
            str_head_file = str(head_file)
            if not str_head_file.startswith(self.sensors_filename_prefix):
                continue
            name = str_head_file.split(".")[0]
            sensor, projection = \
                self.read_sensors_of_type(name, os.path.join(path, head_file))
            sensors_set = sensors.get(sensor.sensors_type, OrderedDict())
            sensors_set.update({sensor: projection})
            sensors[sensor.sensors_type] = sensors_set

        self._log_success("all Sensors", path)

        return sensors

    def read_sensors_of_type(self, name="", path=None, h5_file=None, close_file=True):
        """
        :param
            sensors_file: Path towards a custom Sensors H5 file
            s_type: Senors s_type
        :return: Sensors object
        """
        h5_file = self._open_file("Sensors", path, h5_file)

        locations = h5_file[SensorsH5Field.LOCATIONS][()]
        try:
            labels = np.array([label.decode("UTF-8")
                               for label in h5_file[SensorsH5Field.LABELS][()]])
        except:
            labels = np.array([])
        try:
            orientations = h5_file[SensorsH5Field.ORIENTATIONS][()]
        except:
            orientations = np.array([])
        name = h5_file.attrs.get("name", name)
        s_type = (h5_file.attrs.get("Sensors_subtype", "")).decode("UTF-8")
        s_class_name = (h5_file.attrs.get(self.H5_SUBTYPE_ATTRIBUTE, "")).decode("UTF-8")
        if SensorsH5Field.PROJECTION_MATRIX in h5_file:
            proj_matrix = h5_file[SensorsH5Field.PROJECTION_MATRIX][()]
            projection = SensorTypesToProjectionDict.get(s_type, ProjectionMatrix())()
            projection.projection_data = proj_matrix
        else:
            projection = None

        self._close_file(h5_file, close_file)

        sensors = \
            SensorsDict.get(s_class_name, Sensors)(
                filepath=path, name=name,
                labels=labels, locations=locations,
                orientations=orientations, sensors_type=s_type)
        sensors.configure()

        self._log_success("Sensors", path)

        return sensors, projection

    def read_mapping(self, mapping_type, connectivity=None,
                     path=None, h5_file=None, close_file=True):
        """
                :param path: Path towards a custom Mapping H5 file
                :return: volume mapping in a numpy array
                """
        h5_file = self._open_file(mapping_type, path, h5_file)

        array_data = h5_file['data'][()]

        vm = RegionMappingDict[mapping_type]()
        vm.array_data = array_data

        if connectivity is not None:
            vm.connectivity = connectivity

        self._close_file(h5_file, close_file)

        self._log_success(mapping_type, path)

        return vm

    def read_region_mapping(self, connectivity=None, surface=None, path=None, h5_file=None, close_file=True):
        """
        :param path: Path towards a custom RegionMapping H5 file
        :return: region mapping in a numpy array
        """
        rm = self.read_mapping("RegionMapping", connectivity, path, h5_file, close_file)
        if surface is not None:
            rm.surface = surface
        return rm

    def read_volume_mapping(self, connectivity=None, volume=None, path=None, h5_file=None, close_file=True):
        """
        :param path: Path towards a custom VolumeMapping H5 file
        :return: volume mapping in a numpy array
        """
        vm = self.read_mapping("RegionVolumeMapping", connectivity, path, h5_file, close_file)
        if volume is not None:
            vm.volume = volume
        return vm

    def read_volume(self, path=None, h5_file=None, close_file=True):
        """
        :param path: Path towards a custom StructuralMRI H5 file
        :return: structural MRI in a numpy array
        """
        h5_file = self._open_file("StructuralMRI", path, h5_file)

        data = h5_file['data'][()]

        t1 = StructuralMRI()
        t1.array_data = data

        self._close_file(h5_file, close_file)

        self._log_success("StructuralMRI", path)

        return t1

    def read_head(self, path, atlas="default"):
        """
        :param path: Path towards a custom datatypes folder
        :return: Head object
        """
        self.logger.info("Starting to read Head from: %s" % path)
        conn = \
            self.read_connectivity(os.path.join(path, self.connectivity_filename))
        cort_srf = \
            self.read_surface(os.path.join(path, self.cortical_surface_filename))
        subcort_srf = \
            self.read_surface(os.path.join(path, self.subcortical_surface_filename))
        cort_rm = \
            self.read_region_mapping(conn, cort_srf, os.path.join(path, self.cortical_region_mapping_filename))
        if cort_rm is not None:
            cort_rm.connectivity = conn._tvb
            if cort_srf is not None:
                cort_rm.surface = cort_srf._tvb
        subcort_rm = \
            self.read_region_mapping(conn, subcort_srf, os.path.join(path, self.subcortical_region_mapping_filename))
        t1 = \
            self.read_volume(os.path.join(path, self.structural_mri_filename))
        vm = \
            self.read_volume_mapping(conn, t1, os.path.join(path, self.volume_mapping_filename))
        sensors = self.read_sensors(path)

        if len(atlas) > 0:
            name = atlas
        else:
            name = path

        head = Head(conn, sensors, cort_srf, subcort_srf, cort_rm, subcort_rm, vm, t1, name, path)

        self.logger.info("Successfully read Head from: %s" % path)

        return head

    def read_ts(self, path, h5_file=None, close_file=True):
        """
        :param path: Path towards a valid TimeSeries H5 file
        :return: Timeseries data and time in 2 numpy arrays
        """
        h5_file = self._open_file("TimeSeries", path, h5_file)

        data = h5_file['data'][()]
        total_time = float(h5_file.attrs["Simulated_period"][0])
        nr_of_steps = int(h5_file["data"].attrs["Number_of_steps"][0])
        start_time = float(h5_file["data"].attrs["Start_time"][0])
        time = np.linspace(start_time, total_time, nr_of_steps)

        self._close_file(h5_file, close_file)

        self.logger.info("First Channel sv sum: " + str(np.sum(data[:, 0])))

        self._log_success("TimeSeries", path)

        return time, data

    def read_timeseries(self, path, time_series=TimeSeries, time_series_dict=TimeSeriesDict, h5_file=None,
                        close_file=True):
        """
        :param path: Path towards a valid TimeSeries H5 file
        :return: Timeseries data and time in 2 numpy arrays
        """
        h5_file = self._open_file("TimeSeries", path, h5_file)

        data = h5_file['data'][()]

        ts_type = (h5_file.attrs.get(self.H5_SUBTYPE_ATTRIBUTE)).decode("UTF-8")

        time_series_class = time_series_dict.get(ts_type, time_series)

        ts_kwargs = {}
        labels_dimensions = {}
        try:
            time = h5_file['time'][()]
            ts_kwargs["time"] = time
            ts_kwargs["sample_period"] = float(np.mean(np.diff(time)))
        except:
            pass
        try:
            labels_ordering = [label.decode("UTF-8") for label in (h5_file['dimensions_labels'][()]).tolist()]
        except:
            try:
                labels_ordering = time_series_class.labels_ordering.default
            except:
                labels_ordering = time_series_class._default_labels_ordering.default
        for i_dim, dim_label in enumerate(labels_ordering[1:]):
            try:
                labels = h5_file['%s' % dim_label][()]
                if isinstance(labels[0], np.string_):
                    labels_dimensions.update({dim_label: np.array([label.decode("UTF-8") for label in labels])})
                else:
                    labels_dimensions.update({dim_label: labels})
            except:
                pass
        if len(labels_dimensions) > 0:
            ts_kwargs["labels_dimensions"] = labels_dimensions
        time_unit = (h5_file.attrs.get("sample_period_unit")).decode("UTF-8")
        if len(time_unit) > 0:
            ts_kwargs["sample_period_unit"] = time_unit
        title = (h5_file.attrs.get("title", "")).decode("UTF-8")
        if len(title) > 0:
            ts_kwargs["title"] = title
        # If this is a TimeSeriesRegion try to write all of the structures below:
        try:
            ts_kwargs["connectivity"] = self.read_connectivity(h5_file=h5_file["connectivity"], close_file=False)
        except:
            pass
        try:
            ts_kwargs["region_mapping_volume"] = \
                self.read_volume_mapping(connectivity=ts_kwargs.get("connectivity", None),
                                         h5_file=h5_file["volume_mapping"], close_file=False)
        except:
            pass
        try:
            ts_kwargs["region_mapping"] = \
                self.read_region_mapping(connectivity=ts_kwargs.get("connectivity", None),
                                         h5_file=h5_file["region_mapping"], close_file=False)
        except:
            pass
        # If this is a TimeSeriesSensors try to write the sensors:
        try:
            ts_kwargs["sensors"] = \
                self.read_sensors_of_type(h5_file=h5_file["sensors"], close_file=False)
        except:
            pass
        # If this is a TimeSeriesSurface try to write the sensors:
        try:
            ts_kwargs["surface"] = \
                self.read_surface(h5_file=h5_file["surface"], close_file=False)
        except:
            pass
        # If this is a TimeSeriesVolume try to write the sensors:
        try:
            ts_kwargs["volume"] = \
                self.read_volume(h5_file=h5_file["volume"], close_file=False)
        except:
            pass

        self._close_file(h5_file, close_file)

        self.logger.info("First Channel sv sum: " + str(np.sum(data[:, 0])))
        self._log_success("TimeSeries", path)

        return time_series_class(data, labels_ordering=labels_ordering, **ts_kwargs)

    def read_time_series(self, path, h5_file=None, close_file=True):
        return self.read_timeseries(path, TimeSeries, TimeSeriesDict, h5_file, close_file)

    def read_xarray_time_series(self, path, h5_file=None, close_file=True):
        return self.read_timeseries(path, XarrayTimeSeries, XarrayTimeSeriesDict, h5_file, close_file)

    def read_dictionary(self, path=None, h5_file=None, type=None, close_file=True):
        """
        :param path: Path towards a dictionary H5 file
        :return: dict
        """
        h5_file = self._open_file("Dictionary", path, h5_file)
        dictionary = H5GroupHandlers().read_dictionary_from_group(h5_file, type)
        self._close_file(h5_file, close_file)
        self._log_success("Dictionary", path)
        return dictionary

    def read_list_of_dicts(self, path=None, h5_file=None, type=None, close_file=True):
        h5_file = self._open_file("List of dictionaries", path, h5_file)
        list_of_dicts = []
        id = 0
        h5_group_handlers = H5GroupHandlers()
        while 1:
            try:
                dict_group = h5_file[str(id)]
            except:
                break
            list_of_dicts.append(h5_group_handlers.read_dictionary_from_group(dict_group, type))
            id += 1
        self._close_file(h5_file, close_file)
        self._log_success("List of dictionaries", path)
        return list_of_dicts
