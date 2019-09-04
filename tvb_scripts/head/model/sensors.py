# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
from tvb_scripts.utils.log_error_utils import warning
from tvb_scripts.utils.data_structures_utils import ensure_list, \
    labels_to_inds, monopolar_to_bipolar, split_string_text_numbers
from tvb.datatypes.sensors import Sensors as TVBSensors
from tvb.datatypes.sensors import SensorsEEG as TVBSensorsEEG
from tvb.datatypes.sensors import SensorsMEG as TVBSensorsMEG
from tvb.datatypes.sensors import SensorsInternal as TVBSensorsInternal
from tvb.datatypes.sensors import EEG_POLYMORPHIC_IDENTITY, MEG_POLYMORPHIC_IDENTITY, INTERNAL_POLYMORPHIC_IDENTITY
from tvb.datatypes.projections import ProjectionSurfaceEEG, ProjectionSurfaceMEG, ProjectionSurfaceSEEG


class SensorTypes(Enum):
    TYPE_EEG = EEG_POLYMORPHIC_IDENTITY
    TYPE_MEG = MEG_POLYMORPHIC_IDENTITY
    TYPE_INTERNAL = INTERNAL_POLYMORPHIC_IDENTITY
    TYPE_SEEG = "SEEG"


SensorTypesNames = [getattr(SensorTypes, stype).value for stype in SensorTypes.__members__]

SensorTypesToProjectionDict = {"EEG": ProjectionSurfaceEEG,
                               "MEG": ProjectionSurfaceMEG,
                               "SEEG": ProjectionSurfaceSEEG,
                               "Internal": ProjectionSurfaceSEEG}


class Sensors(object):
    name = ""
    file_path = ""
    _tvb = None

    def __init__(self, **kwargs):
        self.name = kwargs.pop("name", "")
        self.file_path = kwargs.pop("file_path", "")
        tvb_sensors = kwargs.pop("tvb_sensors", TVBSensors())
        if isinstance(tvb_sensors, TVBSensors):
            for attr, value in kwargs.items():
                try:
                    if len(value):
                        setattr(tvb_sensors, attr, value)
                except:
                    warning("Failed to set attribute %s to TVB sensors!" % attr)
            self._tvb = tvb_sensors
            self._tvb.number_of_sensors = self.locations.shape[0]
            if len(self.name) == 0:
                self.name = self._tvb.sensors_type

    def __getattr__(self, attr):
        return getattr(self._tvb, attr)

    def from_tvb_instance(self, instance):
        self._tvb = instance
        return self

    def from_tvb_file(self, filepath):
        self._tvb = TVBSensors.from_file(filepath, self._tvb)
        self.file_path = filepath
        return self

    def configure(self, remove_leading_zeros_from_labels=False):
        if isinstance(self._tvb, Sensors):
            if len(self._tvb.labels) > 0:
                if remove_leading_zeros_from_labels:
                    self.remove_leading_zeros_from_labels()
            self._tvb.configure()

    def sensor_label_to_index(self, labels):
        indexes = []
        for label in ensure_list(labels):
            indexes.append(np.where([np.array(lbl) == np.array(label) for lbl in self._tvb.labels])[0][0])
        if isinstance(labels, (list, tuple)) or len(indexes) > 1:
            return indexes
        else:
            return indexes[0]

    def get_sensors_inds_by_sensors_labels(self, lbls):
        # Make sure that the labels are not bipolar:
        lbls = [label.split("-")[0] for label in ensure_list(lbls)]
        return labels_to_inds(self._tvb.labels, lbls)

    def remove_leading_zeros_from_labels(self):
        labels = []
        for label in self._tvb.labels:
            splitLabel = split_string_text_numbers(label)[0]
            n_lbls = len(splitLabel)
            if n_lbls > 0:
                elec_name = splitLabel[0]
                if n_lbls > 1:
                    sensor_ind = splitLabel[1]
                    labels.append(elec_name + sensor_ind.lstrip("0"))
                else:
                    labels.append(elec_name)
            else:
                labels.append(label)
        self._tvb.labels = np.array(labels)

    def get_bipolar_sensors(self, sensors_inds=None):
        if sensors_inds is None:
            sensors_inds = range(self._tvb.number_of_sensors)
        return monopolar_to_bipolar(self._tvb.labels, sensors_inds)


class SensorsEEG(Sensors):
    _tvb = None

    def __init__(self, **kwargs):
        tvb_sensors = kwargs.pop("tvb_sensors", TVBSensorsEEG())
        super(SensorsEEG, self).__init__(tvb_sensors=tvb_sensors, **kwargs)

    def from_tvb_file(self, filepath):
        self._tvb = TVBSensorsEEG.from_file(filepath, self._tvb)
        return self


class SensorsMEG(Sensors):
    _tvb = None

    def __init__(self, **kwargs):
        tvb_sensors = kwargs.pop("tvb_sensors", TVBSensorsMEG())
        super(SensorsMEG, self).__init__(tvb_sensors=tvb_sensors, **kwargs)

    def from_tvb_file(self, filepath):
        self._tvb = TVBSensorsMEG.from_file(filepath, self._tvb)
        self.file_path = filepath
        return self


class SensorsInternal(Sensors):
    _tvb = None
    elec_labels = np.array([])
    elec_inds = np.array([])

    def __init__(self, **kwargs):
        tvb_sensors = kwargs.pop("tvb_sensors", TVBSensorsInternal())
        self.elec_labels = kwargs.pop("elec_labels", np.array([]))
        self.elec_inds = kwargs.pop("elec_inds", np.array([]))
        super(SensorsInternal, self).__init__(tvb_sensors=tvb_sensors, **kwargs)

    def from_tvb_file(self, filepath):
        self._tvb = TVBSensorsInternal.from_file(filepath, self._tvb)
        self.file_path = filepath
        return self

    @property
    def number_of_electrodes(self):
        return len(self.elec_labels)

    @property
    def channel_labels(self):
        return self.elec_labels

    @property
    def channel_inds(self):
        return self.elec_inds

    def configure(self):
        super(SensorsInternal, self).configure()
        if isinstance(self._tvb, Sensors):
            if self._tvb.number_of_sensors > 0:
                self.elec_labels, self.elec_inds = self.group_sensors_to_electrodes()
            else:
                self.elec_labels = np.array([])
                self.elec_inds = np.array([])

    def get_elecs_inds_by_elecs_labels(self, lbls):
        return labels_to_inds(self.elec_labels, lbls)

    def get_sensors_inds_by_elec_labels(self, lbls):
        elec_inds = self.get_elecs_inds_by_elecs_labels(lbls)
        sensors_inds = []
        for ind in elec_inds:
            sensors_inds += self.elec_inds[ind]
        return np.unique(sensors_inds)

    def group_sensors_to_electrodes(self, labels=None):
        if self.sensors_type == SensorTypes.TYPE_SEEG:
            if labels is None:
                labels = self._tvb.labels
            sensor_names = np.array(split_string_text_numbers(labels))
            elec_labels = np.unique(sensor_names[:, 0])
            elec_inds = []
            for chlbl in elec_labels:
                elec_inds.append(np.where(sensor_names[:, 0] == chlbl)[0])
            return np.array(elec_labels), np.array(elec_inds)
        else:
            warning("No multisensor electrodes for %s sensors!" % self.sensors_type)
            return self.elec_labels, self.elec_inds

    def get_bipolar_elecs(self, elecs):
        try:
            bipolar_sensors_lbls = []
            bipolar_sensors_inds = []
            for elec_ind in elecs:
                curr_inds, curr_lbls = self.get_bipolar_sensors(sensors_inds=self.elec_inds[elec_ind])
                bipolar_sensors_inds.append(curr_inds)
                bipolar_sensors_lbls.append(curr_lbls)
        except:
            elecs_inds = self.get_elecs_inds_by_elecs_labels(elecs)
            bipolar_sensors_inds, bipolar_sensors_lbls = self.get_bipolar_elecs(elecs_inds)
        return bipolar_sensors_inds, bipolar_sensors_lbls


class SensorsSEEG(SensorsInternal):
    sensors_type = SensorTypes.TYPE_SEEG.value
    _ui_name = sensors_type + " Sensors"

    def __init__(self, **kwargs):
        tvb_sensors = kwargs.pop("tvb_sensors", TVBSensorsInternal())
        super(SensorsInternal, self).__init__(tvb_sensors=tvb_sensors, **kwargs)
        self.sensors_type = SensorTypes.TYPE_SEEG.value
        self._ui_name = self.sensors_type + " Sensors"


SensorTypesToClassesDict = {"EEG": SensorsEEG,
                            "MEG": SensorsMEG,
                            "SEEG": SensorsSEEG,
                            "Internal": SensorsInternal}
