# -*- coding: utf-8 -*-

import numpy as np

from tvb_scripts.utils.log_error_utils import warning
from tvb.datatypes.surfaces import CorticalSurface as TVBCorticalSurface


class Surface(object):
    file_path = ""
    _tvb = TVBCorticalSurface()
    vox2ras = np.array([])

    def __init__(self, **kwargs):
        self.file_path = kwargs.pop("file_path", "")
        self._tvb = kwargs.pop("tvb_surface", TVBCorticalSurface())
        self.vox2ras = kwargs.pop("vox2ras", np.array([]))
        for attr, value in kwargs.items():
            try:
                if value.any():
                    setattr(self._tvb, attr, value)
            except:
                warning("Failed to set attribute %s to TVB surface!" % attr)

    def __getattr__(self, attr):
        return getattr(self._tvb, attr)

    def from_tvb_instance(self, instance):
        self._tvb = instance
        self.vox2ras = np.array([])
        return self

    def from_tvb_file(self, filepath):
        self._tvb = TVBCorticalSurface.from_file(filepath, self._tvb)
        self.vox2ras = np.array([])
        self.file_path = filepath
        return self

    def configure(self):
        self._tvb.configure()

    def get_vertex_normals(self):
        # If there is at least 3 vertices and 1 triangle...
        if self._tvb.number_of_vertices > 2 and self._tvb.number_of_triangles > 0:
            if self._tvb.vertex_normals.shape[0] != self._tvb.number_of_vertices:
                self._tvb.vertex_normals = self._tvb.compute_vertex_normals()
        return self._tvb.vertex_normals

    def get_triangle_normals(self):
        # If there is at least 3 vertices and 1 triangle...
        if self._tvb.number_of_vertices > 2 and self._tvb.number_of_triangles > 0:
            if self._tvb.triangle_normals.shape[0] != self._tvb.number_of_triangles:
                self._tvb.triangle_normals = self._tvb.compute_triangle_normals()
        return self._tvb.triangle_normals

    def get_vertex_areas(self):
        triangle_areas = self._tvb._find_triangle_areas()
        vertex_areas = np.zeros((self._tvb.number_of_vertices,))
        for triang, vertices in enumerate(self._tvb.triangles):
            for i in range(3):
                vertex_areas[vertices[i]] += 1. / 3. * triangle_areas[triang]
        return vertex_areas

    def add_vertices_and_triangles(self, new_vertices, new_triangles,
                                   new_vertex_normals=np.array([]), new_triangle_normals=np.array([])):
        self._tvb.triangles = np.array(self._tvb.triangles.tolist() +
                                       (new_triangles + self._tvb.number_of_vertices).tolist())
        self._tvb.vertices = np.array(self._tvb.vertices.tolist() + new_vertices.tolist())
        self._tvb.vertex_normals = np.array(self._tvb.vertex_normals.tolist() + new_vertex_normals.tolist())
        self._tvb.triangle_normals = np.array(self._tvb.triangle_normals.tolist() + new_triangle_normals.tolist())
        self.get_vertex_normals()
        self.get_triangle_normals()

    def compute_surface_area(self):
        """
            This function computes the surface area
            :param: surface: input surface object
            :return: (sub)surface area, float
            """
        return np.sum(self._tvb._find_triangle_areas())


class CorticalSurface(Surface):

    @property
    def surface_subtype(self):
        return "CORTICAL"


class SubcorticalSurface(Surface):

    @property
    def surface_subtype(self):
        return "SUBCORTICAL"
