# -*- coding: utf-8 -*-

"""
Read tvb-scipts entities from TVB format (tvb_data module) and data-structures
"""
import os
from collections import OrderedDict
from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_scripts.utils.log_error_utils import initialize_logger, raise_value_error
from tvb_scripts.utils.data_structures_utils import ensure_list
from tvb_scripts.head.model.surface import CorticalSurface, SubcorticalSurface
from tvb_scripts.head.model.sensors import Sensors, SensorTypesToClassesDict, SensorTypes, SensorTypesToProjectionDict
from tvb_scripts.head.model.connectivity import Connectivity
from tvb_scripts.head.model.head import Head
from tvb.datatypes import region_mapping, structural
from tvb.datatypes.projections import ProjectionMatrix


class TVBReader(object):
    logger = initialize_logger(__name__)

    def read_connectivity(self, path):
        if os.path.isfile(path):
            conn = Connectivity().from_tvb_file(path)
            conn.file_path = path
            conn.configure()
            return conn
        else:
            raise_value_error(("\n No Connectivity file found at path %s!" % str(path)))

    def read_cortical_surface(self, path, surface_class):
        if os.path.isfile(path):
            surf = surface_class().from_tvb_file(path)
            surf.configure()
            return surf
        else:
            self.logger.warning("\nNo %s Surface file found at path %s!" %
                                (surface_class().surface_subtype, str(path)))
            return None

    def read_region_mapping(self, path):
        if os.path.isfile(path):
            return region_mapping.RegionMapping.from_file(path)
        else:
            self.logger.warning("\nNo Region Mapping file found at path %s!" % str(path))
            return None

    def read_volume_mapping(self, path):
        if os.path.isfile(path):
            return region_mapping.RegionVolumeMapping.from_file(path)
        else:
            self.logger.warning("\nNo Volume Mapping file found at path %s!" % str(path))
            return None

    def read_t1(self, path):
        if os.path.isfile(path):
            return structural.StructuralMRI.from_file(path)
        else:
            self.logger.warning("\nNo Structural MRI file found at path %s!" % str(path))
            return None

    def read_multiple_sensors_and_projections(self, sensors_files, root_folder, s_type, atlas=""):
        sensors_set = OrderedDict()
        if isinstance(sensors_files, (list, tuple)):
            if isinstance(sensors_files, tuple):
                sensors_files = [sensors_files]
            for s_files in sensors_files:
                sensors, projection = \
                    self.read_sensors_and_projection(s_files, root_folder, s_type, atlas)
                sensors_set[sensors] = projection
        return sensors_set

    def read_sensors_and_projection(self, filename, root_folder, s_type, atlas=""):

        def get_sensors_name(sensors_file, name):
            if len(sensors_file) > 1:
                gain_file = sensors_file[1]
            else:
                gain_file = ""
            return name + gain_file.replace(".txt", "").replace(name, "")

        filename = ensure_list(filename)
        path = os.path.join(root_folder, filename[0])
        if os.path.isfile(path):
            sensors = \
                SensorTypesToClassesDict.get(s_type, Sensors)().from_tvb_file(path)
            sensors.configure()
            if len(filename) > 1:
                projection = self.read_projection(os.path.join(root_folder, atlas, filename[1]), s_type)
            else:
                projection = None
            sensors.name = get_sensors_name(filename, sensors._ui_name)
            sensors.configure()
            return sensors, projection
        else:
            self.logger.warning("\nNo Sensor file found at path %s!" % str(path))
            return None

    def read_projection(self, path, projection_type):
        if os.path.isfile(path):
            return SensorTypesToProjectionDict.get(projection_type, ProjectionMatrix).from_file(path)
        else:
            self.logger.warning("\nNo Projection Matrix file found at path %s!" % str(path))
            return None

    def read_head(self, root_folder, name='', atlas="default",
                  connectivity_file="connectivity.zip",
                  cortical_surface_file="surface_cort.zip",
                  subcortical_surface_file="surface_subcort.zip",
                  cortical_region_mapping_file="region_mapping_cort.txt",
                  subcortical_region_mapping_file="region_mapping_subcort.txt",
                  eeg_sensors_files=[("eeg_brainstorm_65.txt", "gain_matrix_eeg_65_surface_16k.npy")],
                  meg_sensors_files=[("meg_brainstorm_276.txt", "gain_matrix_meg_276_surface_16k.npy")],
                  seeg_sensors_files=[("seeg_xyz.txt", "seeg_dipole_gain.txt"),
                                      ("seeg_xyz.txt", "seeg_distance_gain.txt"),
                                      ("seeg_xyz.txt", "seeg_regions_distance_gain.txt"),
                                      ("seeg_588.txt", "gain_matrix_seeg_588_surface_16k.npy")],
                  vm_file="aparc+aseg.nii.gz", t1_file="T1.nii.gz"):

        conn = self.read_connectivity(os.path.join(root_folder, atlas, connectivity_file))
        cort_srf = \
            self.read_cortical_surface(os.path.join(root_folder, cortical_surface_file), CorticalSurface)
        cort_rm = self.read_region_mapping(os.path.join(root_folder, atlas, cortical_region_mapping_file))
        if cort_rm is not None:
            cort_rm.connectivity = conn._tvb
            if cort_srf is not None:
                cort_rm.surface = cort_srf._tvb
        subcort_srf = \
            self.read_cortical_surface(os.path.join(root_folder, subcortical_surface_file), SubcorticalSurface)
        subcort_rm = self.read_region_mapping(os.path.join(root_folder, atlas, subcortical_region_mapping_file))
        if subcort_rm is not None:
            subcort_rm.connectivity = conn._tvb
            if subcort_srf is not None:
                subcort_rm.surface = subcort_srf._tvb
        vm = self.read_volume_mapping(os.path.join(root_folder, atlas, vm_file))
        t1 = self.read_t1(os.path.join(root_folder, t1_file))
        sensors = OrderedDict()
        sensors[SensorTypes.TYPE_EEG.value] = \
            self.read_multiple_sensors_and_projections(eeg_sensors_files, root_folder,
                                                       SensorTypes.TYPE_EEG.value, atlas)
        sensors[SensorTypes.TYPE_MEG.value] = \
            self.read_multiple_sensors_and_projections(meg_sensors_files, root_folder,
                                                       SensorTypes.TYPE_MEG.value, atlas)
        sensors[SensorTypes.TYPE_SEEG.value] = \
            self.read_multiple_sensors_and_projections(seeg_sensors_files, root_folder,
                                                       SensorTypes.TYPE_SEEG.value, atlas)
        if len(name) == 0:
            name = atlas
        return Head(conn, sensors, cort_srf, subcort_srf, cort_rm, subcort_rm, vm, t1, name)
