# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
from tvb_scripts.utils.log_error_utils import warning
from tvb_scripts.utils.data_structures_utils import labels_to_inds
from tvb.simulator.plot.utils import normalize_weights
from tvb.datatypes.connectivity import Connectivity as TVBConnectivity


class Connectivity(TVBConnectivity):
    file_path = ""

    @staticmethod
    def from_instance(instance, **kwargs):
        result = deepcopy(instance)
        for attr, value in kwargs.items():
            try:
                if len(value):
                    setattr(result, attr, value)
            except:
                warning("Failed to set attribute %s to %s!" % (attr, instance.__class__.__name__))
        result.configure()
        return result

    # TODO: Fix this!
    # @staticmethod
    # def from_tvb_instance(instance, **kwargs):
    #     result = Connectivity()
    #     result.file_path = kwargs.pop("file_path", "")
    #     if isinstance(instance, TVBConnectivity):
    #         for attr, value in instance.__dict__:
    #             try:
    #                 if len(value):
    #                     setattr(result, attr, value)
    #             except:
    #                 warning("Failed to set attribute %s of TVB %s!" % (attr, instance.__class__.__name__))
    #     return Connectivity.from_instance(result, **kwargs)

    @staticmethod
    def from_file(filepath, **kwargs):
        result = TVBConnectivity.from_file(filepath)
        if isinstance(result, TVBConnectivity):
            raise NotImplementedError
            # return Connectivity.from_tvb_instance(result, **kwargs)
        else:
            result.centres = result.centers
            return Connectivity.from_instance(result, **kwargs)

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
        return self.centres
