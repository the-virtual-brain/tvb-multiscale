# -*- coding: utf-8 -*-

import numpy as np
from tvb_scripts.utils.log_error_utils import warning
from tvb_scripts.utils.data_structures_utils import labels_to_inds
from tvb_scripts.utils.computations_utils import normalize_weights
from tvb.datatypes.connectivity import Connectivity as TVBConnectivity


class Connectivity(object):
    _tvb = None
    file_path = ""
    normalized_weights = np.array([])

    def __init__(self, **kwargs):
        self.file_path = kwargs.pop("file_path", "")
        self.normalized_weights = kwargs.pop("normalized_weights", np.array([]))
        # TODO: find why there is an error without the hack below...
        tvb_conn = kwargs.pop("tvb_connectivity", TVBConnectivity())
        if isinstance(tvb_conn, TVBConnectivity):
            for attr, value in kwargs.items():
                try:
                    if len(value):
                        setattr(tvb_conn, attr, value)
                except:
                    warning("Failed to set attribute %s to TVB connectivity!" % attr)
            self._tvb = tvb_conn

    def __getattr__(self, attr):
        return getattr(self._tvb, attr)

    def from_tvb_instance(self, instance):
        self._tvb = instance
        if len(self.normalized_weights) == 0:
            self.normalized_weights = normalize_weights(self._tvb.weights, remove_diagonal=True, ceil=1.0)
        return self

    def from_tvb_file(self, filepath):
        self._tvb = TVBConnectivity.from_file(filepath, self._tvb)
        if len(self.normalized_weights) == 0:
            self.normalized_weights = normalize_weights(self._tvb.weights, remove_diagonal=True, ceil=1.0)
        self.file_path = filepath
        return self

    @property
    def number_of_regions(self):
        return self._tvb.weights.shape[0]

    def configure(self):
        self._tvb.configure()
        if len(self.normalized_weights) == 0:
            self.normalized_weights = normalize_weights(self.weights, remove_diagonal=True, ceil=1.0)

    def regions_labels2inds(self, labels):
        inds = []
        for lbl in labels:
            inds.append(np.where(self.region_labels == lbl)[0][0])
        if len(inds) == 1:
            return inds[0]
        else:
            return inds

    def get_regions_inds_by_labels(self, lbls):
        return labels_to_inds(self.region_labels, lbls)

    @property
    def centers(self):
        return self._tvb.centres