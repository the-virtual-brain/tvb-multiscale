# coding=utf-8
from enum import Enum

import numpy as np
from tvb.simulator.plot.utils.data_structures_utils import ensure_list, labels_to_inds, split_string_text_numbers, \
    monopolar_to_bipolar
from tvb.simulator.plot.utils.log_error_utils import warning

from tvb_scripts.datatypes.base import BaseModel

from tvb.basic.neotraits.api import Attr, NArray
from tvb.datatypes.sensors import Sensors as TVBSensors
from tvb.datatypes.sensors import SensorsEEG as TVBSensorsEEG
from tvb.datatypes.sensors import SensorsMEG as TVBSensorsMEG
from tvb.datatypes.sensors import SensorsInternal as TVBSensorsInternal
from tvb.datatypes.sensors import EEG_POLYMORPHIC_IDENTITY, MEG_POLYMORPHIC_IDENTITY, \
    INTERNAL_POLYMORPHIC_IDENTITY
from tvb.datatypes.projections import \
    ProjectionSurfaceEEG, ProjectionSurfaceMEG, ProjectionSurfaceSEEG


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


class SensorsH5Field(object):
    PROJECTION_MATRIX = "projection_matrix"
    LABELS = "labels"
    LOCATIONS = "locations"
    ORIENTATIONS = "orientations"


class Sensors(TVBSensors, BaseModel):
    name = Attr(
        field_type=str,
        label="Sensors' name",
        default='', required=False,
        doc="Sensors' name")

    def configure(self, remove_leading_zeros_from_labels=False):
        if len(self.labels) > 0:
                if remove_leading_zeros_from_labels:
                    self.remove_leading_zeros_from_labels()
        self.configure()

    def sensor_label_to_index(self, labels):
        return self.labels2inds(self.labels, labels)

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
            sensors_inds = range(self.number_of_sensors)
        return monopolar_to_bipolar(self.labels, sensors_inds)


class SensorsEEG(Sensors, TVBSensorsEEG):
   pass


class SensorsMEG(Sensors, TVBSensorsMEG):
    pass


class SensorsInternal(Sensors, TVBSensorsInternal):
    elec_labels = NArray(
        dtype=np.str,
        label="Electrodes' labels", default=None, required=False,
        doc="""Labels of electrodes.""")

    elec_inds = NArray(
        dtype=np.int,
        label="Electrodes' indices", default=None, required=False,
        doc="""Indices of electrodes.""")

    @property
    def number_of_electrodes(self):
        if self.elec_labels is None:
            return 0
        else:
            return len(self.elec_labels)

    @property
    def channel_labels(self):
        return self.elec_labels

    @property
    def channel_inds(self):
        return self.elec_inds

    def configure(self):
        super(SensorsInternal, self).configure()
        if self._tvb.number_of_sensors > 0:
            self.elec_labels, self.elec_inds = self.group_sensors_to_electrodes()
        else:
            self.elec_labels = None
            self.elec_inds = None

    def get_elecs_inds_by_elecs_labels(self, lbls):
        if self.elec_labels is not None:
            return labels_to_inds(self.elec_labels, lbls)
        else:
            return None

    def get_sensors_inds_by_elec_labels(self, lbls):
        elec_inds = self.get_elecs_inds_by_elecs_labels(lbls)
        if elec_inds is not None:
            sensors_inds = []
            for ind in elec_inds:
                sensors_inds += self.elec_inds[ind]
            return np.unique(sensors_inds)

    def group_sensors_to_electrodes(self, labels=None):
        if self.sensors_type == SensorTypes.TYPE_SEEG:
            if labels is None:
                labels = self.labels
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
            if self.elecs_inds is None:
                return None
            for elec_ind in elecs:
                curr_inds, curr_lbls = self.get_bipolar_sensors(sensors_inds=self.elec_inds[elec_ind])
                bipolar_sensors_inds.append(curr_inds)
                bipolar_sensors_lbls.append(curr_lbls)
        except:
            elecs_inds = self.get_elecs_inds_by_elecs_labels(elecs)
            if elecs_inds is None:
                return None
            bipolar_sensors_inds, bipolar_sensors_lbls = self.get_bipolar_elecs(elecs_inds)
        return bipolar_sensors_inds, bipolar_sensors_lbls


class SensorsSEEG(SensorsInternal):
    sensors_type = Attr(str, default=SensorTypes.TYPE_SEEG.value, required=False)


SensorsDict = {
    Sensors.__name__: Sensors,
    SensorsEEG.__name__: SensorsEEG,
    SensorsMEG.__name__: SensorsMEG,
    SensorsSEEG.__name__: SensorsSEEG,
    SensorsInternal.__name__: SensorsInternal}
