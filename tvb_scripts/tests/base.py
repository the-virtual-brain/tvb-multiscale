# -*- coding: utf-8 -*-

import os

import numpy
from tvb.datatypes.head import Head
from tvb.simulator.plot.config import Config

from tvb_scripts.datatypes.connectivity import Connectivity
from tvb_scripts.datatypes.sensors import Sensors
from tvb_scripts.datatypes.surface import Surface
from tvb_scripts.io.h5_reader import H5Reader


class BaseTest(object):
    config = Config(output_base=os.path.join(os.getcwd(), "test_out"))

    dummy_connectivity = Connectivity(weights=numpy.array([[1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [3.0, 2.0, 1.0]]),
                                      tract_lengths=numpy.array([[4, 5, 6], [5, 6, 4], [6, 4, 5]]),
                                      region_labels=["a", "b", "c"],
                                      centres=numpy.array([1.0, 2.0, 3.0]))
    dummy_surface = Surface(vertices=numpy.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]]), triangles=numpy.array([[0, 1, 2]]))
    dummy_sensors = Sensors(labels=numpy.array(["sens1", "sens2"]), locations=numpy.array([[0, 0, 0], [0, 1, 0]]))

    def _prepare_dummy_head_from_dummy_attrs(self):
        return Head(self.dummy_connectivity,
                    cortical_surface=self.dummy_surface,
                    sensorsSEEG={"SensorsSEEG": self.dummy_sensors})

    def _prepare_dummy_head(self):
        reader = H5Reader()
        connectivity = reader.read_connectivity(os.path.join(self.config.input.HEAD, "Connectivity.h5"))
        cort_surface = Surface()
        seeg_sensors = Sensors(labels=numpy.array(["sens1", "sens2"]), ocations=numpy.array([[0, 0, 0], [0, 1, 0]]))
        head = Head(connectivity, cort_surface, sensorsSEEG={"SensorsSEEG": seeg_sensors})

        return head

    @classmethod
    def setup_class(cls):
        for direc in (cls.config.out.FOLDER_LOGS, cls.config.out.FOLDER_RES, cls.config.out.FOLDER_FIGURES,
                      cls.config.out.FOLDER_TEMP):
            if not os.path.exists(direc):
                os.makedirs(direc)

    @classmethod
    def teardown_class(cls):
        for direc in (cls.config.out.FOLDER_LOGS, cls.config.out.FOLDER_RES, cls.config.out.FOLDER_FIGURES,
                      cls.config.out.FOLDER_TEMP):
            for dir_file in os.listdir(direc):
                os.remove(os.path.join(os.path.abspath(direc), dir_file))
            os.removedirs(direc)
