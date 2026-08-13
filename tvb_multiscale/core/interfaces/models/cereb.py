# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC, abstractmethod

import numpy as np

from tvb.basic.neotraits._attr import Int, Float, NArray
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.noise import Additive

from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels
from tvb_multiscale.core.interfaces.models.default import DefaultInterfaceBuilder, DefaultTVBInterfaceBuilder, \
    DefaultTVBSpikeNetInterfaceBuilder, DefaultSpikeNetInterfaceBuilder, DefaultSpikeNetProxyNodesBuilder
from tvb_multiscale.core.interfaces.transformers.models.red_wong_wang import ElephantSpikesRateRedWongWangExc


class CerebTVBInterfaceBuilder(DefaultTVBInterfaceBuilder):

    def default_output_config(self):
        for ii in range(1):
            self._get_output_interfaces(ii)["voi"] = "R"

    def default_input_config(self):
        for ii in range(3):
            self._get_input_interfaces(ii)["voi"] = "R"


class CerebTVBtoSpikeNetTransformerBuilder(DefaultInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    N_mf = Int(field_type=int, label="N_mf", default=117,
               doc="""Number of cortical neurons projecting to mossy fibers neurons""")

    N_io = Int(field_type=int, label="N_io", default=14,
               doc="""Number of cortical neurons projecting to inferior olivary neurons""")

    def default_tvb_to_spikeNet_config(self, interfaces):
        for interface, number_of_neurons in zip(interfaces, [self.N_mf]):  # , self.N_io
            if self.model == TVBtoSpikeNetModels.SPIKES.name:
                interface["transformer_params"] = {"scale_factor": np.array([1.0]),
                                                   "number_of_neurons": np.array([number_of_neurons])}
            else:  # RATE
                if self.is_tvb_coupling_interface(interface):  # np.array([number_of_neurons])
                    interface["transformer_params"] = {"scale_factor": np.array([7.0])}
                else:
                    interface["transformer_params"] = {"scale_factor": np.array([0.01])}


class CerebSpikeNetToTVBTransformerBuilder(DefaultInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    N_grc = Int(field_type=int, label="N_mf", default=28646, doc="""Number of granule neurons""")

    N_dcgl = Int(field_type=int, label="N_odcn", default=129, doc="""Number of output dcn_cell_glut_large neurons""")

    N_io = Int(field_type=int, label="N_io", default=14, doc="""Number of inferior olivary neurons""")

    def default_spikeNet_to_tvb_config(self, interfaces):
        for interface, number_of_neurons in zip(interfaces, [self.N_grc, self.N_dcgl, self.N_io]):
            interface["transformer_params"] = {"scale_factor": np.array([1.0]) / number_of_neurons}


class CerebSpikeNetProxyNodesBuilder(DefaultSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    N_mf = Int(field_type=int, label="N_mf", default=100,
               doc="""Number of cortical neurons projecting to mossy fibers neurons""")

    N_io = Int(field_type=int, label="N_io", default=100,
               doc="""Number of cortical neurons projecting to inferior olivary neurons""")

    CC_proxy_inds = NArray(
        dtype=int,
        label="CC_proxy_inds",
        doc="""Indices of Spiking Network Cerebellar Cortices proxy nodes""",
        required=True,
        default=np.array([10])
    )

    CN_proxy_inds = NArray(
        dtype=int,
        label="CN_proxy_inds",
        doc="""Indices of Spiking Network Cerebellar Nuclei proxy nodes""",
        required=True,
        default=np.array([16])
    )

    IO_proxy_inds = NArray(
        dtype=int,
        label="IO_proxy_inds",
        doc="""Indices of Spiking Network Inferior Olivary Cortices proxy nodes""",
        required=True,
        default=np.array([0])
    )

    @property
    @abstractmethod
    def G(self):
        pass

    def _configure_global_coupling_scaling(self):
        if self.global_coupling_scaling.size == 0:
            self.global_coupling_scaling = self.tvb_coupling_a * self.G
        DefaultSpikeNetProxyNodesBuilder._configure_global_coupling_scaling(self)

    def default_tvb_to_spikeNet_config(self, interfaces):
        if self.model == TVBtoSpikeNetModels.SPIKES.name:
            for interface, number_of_neurons in zip(interfaces, [self.N_mf]):  # , self.N_io
                interface["proxy_params"] = {"number_of_neurons": number_of_neurons}


class CerebSpikeNetInterfaceBuilder(CerebSpikeNetProxyNodesBuilder, DefaultSpikeNetInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    @property
    def G(self):
        return self.tvb_simulator_serialized["model.G"]

    def default_output_config(self):
        for ii, (pop, spiking_proxy_inds) in \
                enumerate(zip(["granule_cell", "dcn_cell_glut_large", "io_cell"],
                              [self.CC_proxy_inds, self.CN_proxy_inds, self.IO_proxy_inds])):
            self._get_output_interfaces(ii)["populations"] = np.array(pop)
            self._get_output_interfaces(ii)["proxy_inds"] = np.array(spiking_proxy_inds)
        CerebSpikeNetProxyNodesBuilder.default_spikeNet_to_tvb_config(self, self.output_interfaces)

    def default_input_config(self):
        for ii, (pop, spiking_proxy_inds) in enumerate(zip(["mossy_fibers"],  # , "io_cell"
                                                           [self.CC_proxy_inds])):  # , self.IO_proxy_inds
            self._get_input_interfaces(ii)["populations"] = np.array(pop)
            self._get_input_interfaces(ii)["spiking_proxy_inds"] = np.array(spiking_proxy_inds)
        CerebSpikeNetProxyNodesBuilder.default_tvb_to_spikeNet_config(self, self.input_interfaces)


class CerebTVBSpikeNetInterfaceBuilder(CerebTVBInterfaceBuilder,
                                       CerebTVBtoSpikeNetTransformerBuilder,
                                       CerebSpikeNetToTVBTransformerBuilder,
                                       CerebSpikeNetProxyNodesBuilder,
                                                        DefaultTVBSpikeNetInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    @property
    def G(self):
        return self.tvb_model.G

    def default_output_config(self):
        CerebTVBInterfaceBuilder.default_output_config(self)
        CerebTVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)
        for ii, (pop, spiking_proxy_inds) in enumerate(zip(["mossy_fibers"],  # , "io_cell"
                                                           [self.CC_proxy_inds])):  # , self.IO_proxy_inds
            self._get_output_interfaces(ii)["populations"] = np.array(pop)
            self._get_output_interfaces(ii)["spiking_proxy_inds"] = np.array(spiking_proxy_inds)
        CerebSpikeNetInterfaceBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)

    def default_input_config(self):
        CerebTVBInterfaceBuilder.default_input_config(self)
        CerebSpikeNetToTVBTransformerBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)
        for ii, (pop, spiking_proxy_inds) in \
                enumerate(zip(["granule_cell", "dcn_cell_glut_large", "io_cell"],
                              [self.CC_proxy_inds, self.CN_proxy_inds, self.IO_proxy_inds])):
            self._get_input_interfaces(ii)["populations"] = np.array(pop)
            self._get_input_interfaces(ii)["proxy_inds"] = np.array(spiking_proxy_inds)
        CerebSpikeNetInterfaceBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)
