# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC, abstractmethod

import numpy as np

from tvb.basic.neotraits._attr import Int, Float, NArray
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.noise import Additive

from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels
from tvb_multiscale.core.interfaces.models.default import DefaultTVBInterfaceBuilder, \
    DefaultTVBSpikeNetInterfaceBuilder, DefaultTVBRemoteInterfaceBuilder, DefaultTVBTransfomerInterfaceBuilder, \
    DefaultTVBOutputTransformerInterfaceBuilder, DefaultTVBInputTransformerInterfaceBuilder, \
    DefaultSpikeNetInterfaceBuilder, DefaultSpikeNetRemoteInterfaceBuilder, \
    DefaultSpikeNetTransformerInterfaceBuilder, DefaultSpikeNetProxyNodesBuilder, DefaultInterfaceBuilder
from tvb_multiscale.core.interfaces.transformers.models import ElephantSpikesRateRedWongWangExc


class BasalGangliaIzhikevichTVBInterfaceBuilder(DefaultTVBInterfaceBuilder):

    def default_output_config(self):
        for ii in range(2):
            self._get_output_interfaces(ii)["voi"] = "S" if self.model == TVBtoSpikeNetModels.CURRENT.name else "R"

    def default_input_config(self):
        for ii in range(3):
            self._get_input_interfaces(ii)["voi"] = ("S", "R")


class BasalGangliaIzhikevichTVBtoSpikeNetTransformerBuilder(DefaultInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    N_E = Int(field_type=int, label="N_E", default=100, doc="""Number of excitatory neurons""")

    @property
    @abstractmethod
    def J_N(self):
        pass

    def default_tvb_to_spikeNet_config(self, interfaces):
        for interface in interfaces:
            if self.model == TVBtoSpikeNetModels.CURRENT.name:
                interface["transformer_params"] = {"scale_factor": self.N_E * self.J_N}
            elif self.model == TVBtoSpikeNetModels.SPIKES.name:
                interface["transformer_params"] = {"scale_factor": np.array([1.0]),
                                                   "number_of_neurons": np.array([self.N_E])}
            else:  # RATE
                interface["transformer_params"] = {"scale_factor": self.N_E * np.array([1.0])}


class BasalGangliaIzhikevichSpikeNetToTVBTransformerBuilder(DefaultInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    N_E = Int(field_type=int, label="N_E", default=100, doc="""Number of excitatory neurons""")

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True, default=0.0)

    tau_r = NArray(
        label=r":math:`\tau_R`",
        default=np.array([10., ]),
        doc="""[ms]. Excitatory population  input rate decay time constant.""")

    GPe_proxy_inds = NArray(
        dtype=int,
        label="GPe_proxy_inds",
        doc="""Indices of Spiking Network GPe proxy nodes""",
        required=True,
        default=np.array([0, 1])
    )

    GPi_proxy_inds = NArray(
        dtype=int,
        label="GPi_proxy_inds",
        doc="""Indices of Spiking Network GPi proxy nodes""",
        required=True,
        default=np.array([2, 3])
    )

    STN_proxy_inds = NArray(
        dtype=int,
        label="STN_proxy_inds",
        doc="""Indices of Spiking Network STN proxy nodes""",
        required=True,
        default=np.array([4, 5])
    )

    Thal_proxy_inds = NArray(
        dtype=int,
        label="Thal_proxy_inds",
        doc="""Indices of Spiking Network Thalamus proxy nodes""",
        required=True,
        default=np.array([8, 9])
    )

    Striatum_proxy_inds = NArray(
        dtype=int,
        label="Striatum_proxy_inds",
        doc="""Indices of Spiking Network Striatum proxy nodes""",
        required=True,
        default=np.array([6, 7])
    )

    @property
    def E_proxy_inds(self):
        return np.array(self.STN_proxy_inds.tolist() + self.Thal_proxy_inds.tolist())

    @property
    def I_proxy_inds(self):
        return np.array(self.GPe_proxy_inds.tolist() + self.GPi_proxy_inds.tolist())

    @property
    @abstractmethod
    def _dt(self):
        pass

    @property
    @abstractmethod
    def nsig(self):
        pass

    @property
    @abstractmethod
    def tau_s(self):
        pass

    @property
    @abstractmethod
    def gamma(self):
        pass

    def default_spikeNet_to_tvb_config(self, interfaces):
        for interface, number_of_proxys, scale_neurons in \
            zip(interfaces,
                [self.E_proxy_inds.size, self.I_proxy_inds.size, self.Striatum_proxy_inds.size],
                [1, 1, 2]):
            interface["transformer_model"] = ElephantSpikesRateRedWongWangExc
            interface["transformer_params"] = \
                {"scale_factor": np.array([1.0]) / self.N_E / scale_neurons,
                 "integrator": HeunStochastic(dt=self._dt, noise=Additive(nsig=self.nsig)),
                 "state": np.zeros((2, number_of_proxys)),
                 "tau_s": self.tau_s, "tau_r": self.tau_r, "gamma": self.gamma}


class BasalGangliaIzhikevichTVBRemoteInterfaceBuilder(BasalGangliaIzhikevichTVBInterfaceBuilder,
                                                      DefaultTVBRemoteInterfaceBuilder):
    pass


class BasalGangliaIzhikevichTVBOutputTransformerInterfaceBuilder(BasalGangliaIzhikevichTVBInterfaceBuilder,
                                                                 BasalGangliaIzhikevichTVBtoSpikeNetTransformerBuilder,
                                                                 DefaultTVBOutputTransformerInterfaceBuilder):

    @property
    def J_N(self):
        return self.tvb_model.J_N

    def default_output_config(self):
        BasalGangliaIzhikevichTVBInterfaceBuilder.default_output_config(self)
        BasalGangliaIzhikevichTVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config(self,
                                                                                             self.output_interfaces)


class BasalGangliaIzhikevichTVBInputTransformerInterfaceBuilder(BasalGangliaIzhikevichTVBInterfaceBuilder,
                                                                BasalGangliaIzhikevichSpikeNetToTVBTransformerBuilder,
                                                                DefaultTVBInputTransformerInterfaceBuilder):

    @property
    def _dt(self):
        if self.dt <= 0.0:
            self.dt = self.tvb_dt
        return self.dt

    @property
    def tau_s(self):
        return self.tvb_model.tau_s

    @property
    def gamma(self):
        return self.tvb_model.gamma

    def default_input_config(self):
        BasalGangliaIzhikevichTVBInterfaceBuilder.default_input_config(self)
        BasalGangliaIzhikevichSpikeNetToTVBTransformerBuilder.default_spikeNet_to_tvb_config(self,
                                                                                             self.input_interfaces)


class BasalGangliaIzhikevichTVBTransfomerInterfaceBuilder(BasalGangliaIzhikevichTVBInterfaceBuilder,
                                                          BasalGangliaIzhikevichTVBtoSpikeNetTransformerBuilder,
                                                          BasalGangliaIzhikevichSpikeNetToTVBTransformerBuilder,
                                                          DefaultTVBTransfomerInterfaceBuilder):

    @property
    def J_N(self):
        return self.tvb_model.J_N

    @property
    def tau_s(self):
        return self.tvb_model.tau_s

    @property
    def gamma(self):
        return self.tvb_model.gamma

    def default_output_config(self):
        BasalGangliaIzhikevichTVBInterfaceBuilder.default_output_config(self)
        BasalGangliaIzhikevichTVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config(self,
                                                                                             self.output_interfaces)

    def default_input_config(self):
        BasalGangliaIzhikevichTVBInterfaceBuilder.default_input_config(self)
        BasalGangliaIzhikevichSpikeNetToTVBTransformerBuilder.default_spikeNet_to_tvb_config(self,
                                                                                             self.input_interfaces)


class BasalGangliaIzhikevichSpikeNetProxyNodesBuilder(DefaultSpikeNetProxyNodesBuilder, ABC):
    __metaclass__ = ABCMeta

    GPe_proxy_inds = NArray(
        dtype=int,
        label="GPe_proxy_inds",
        doc="""Indices of Spiking Network GPe proxy nodes""",
        required=True,
        default=np.array([0, 1])
    )

    GPi_proxy_inds = NArray(
        dtype=int,
        label="GPi_proxy_inds",
        doc="""Indices of Spiking Network GPi proxy nodes""",
        required=True,
        default=np.array([2, 3])
    )

    STN_proxy_inds = NArray(
        dtype=int,
        label="STN_proxy_inds",
        doc="""Indices of Spiking Network STN proxy nodes""",
        required=True,
        default=np.array([4, 5])
    )

    Thal_proxy_inds = NArray(
        dtype=int,
        label="Thal_proxy_inds",
        doc="""Indices of Spiking Network Thalamus proxy nodes""",
        required=True,
        default=np.array([8, 9])
    )

    Striatum_proxy_inds = NArray(
        dtype=int,
        label="Striatum_proxy_inds",
        doc="""Indices of Spiking Network Striatum proxy nodes""",
        required=True,
        default=np.array([6, 7])
    )

    @property
    def E_proxy_inds(self):
        return np.array(self.STN_proxy_inds.tolist() + self.Thal_proxy_inds.tolist())

    @property
    def I_proxy_inds(self):
        return np.array(self.GPe_proxy_inds.tolist() + self.GPi_proxy_inds.tolist())

    @property
    @abstractmethod
    def G(self):
        pass

    def _configure_global_coupling_scaling(self):
        if self.global_coupling_scaling.size == 0:
            self.global_coupling_scaling = self.tvb_coupling_a * self.G
        DefaultSpikeNetProxyNodesBuilder._configure_global_coupling_scaling(self)

    def default_tvb_to_spikeNet_config(self, interfaces):
        for interface in interfaces:
            if self.model == TVBtoSpikeNetModels.SPIKES.name:
                interface["proxy_params"] = {"number_of_neurons": self.N_E}


class BasalGangliaIzhikevichSpikeNetInterfaceBuilder(BasalGangliaIzhikevichSpikeNetProxyNodesBuilder,
                                                     DefaultSpikeNetInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    @property
    def G(self):
        return self.tvb_simulator_serialized["model.G"]

    def default_output_config(self):
        for ii, (pop, spiking_proxy_inds) in \
                enumerate(zip(["E", "I", ("IdSN", "IiSN")],
                              [self.E_proxy_inds, self.I_proxy_inds, self.Striatum_proxy_inds])):
            self._get_output_interfaces(ii)["populations"] = np.array(pop)
            self._get_output_interfaces(ii)["proxy_inds"] = np.array(spiking_proxy_inds)
        BasalGangliaIzhikevichSpikeNetProxyNodesBuilder.default_spikeNet_to_tvb_config(self, self.output_interfaces)

    def default_input_config(self):
        for ii, (pop, spiking_proxy_inds) in enumerate(zip(["E", ("IdSN", "IiSN")],
                                                           [self.E_proxy_inds, self.Striatum_proxy_inds])):
            self._get_input_interfaces(ii)["populations"] = np.array(pop)
            self._get_input_interfaces(ii)["spiking_proxy_inds"] = np.array(spiking_proxy_inds)
        BasalGangliaIzhikevichSpikeNetProxyNodesBuilder.default_tvb_to_spikeNet_config(self, self.input_interfaces)


class BasalGangliaIzhikevichSpikeNetRemoteInterfaceBuilder(BasalGangliaIzhikevichSpikeNetInterfaceBuilder,
                                                           DefaultSpikeNetRemoteInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta


class BasalGangliaIzhikevichSpikeNetTransformerInterfaceBuilder(BasalGangliaIzhikevichSpikeNetInterfaceBuilder,
                                                                BasalGangliaIzhikevichSpikeNetToTVBTransformerBuilder,
                                                                BasalGangliaIzhikevichTVBtoSpikeNetTransformerBuilder,
                                                                DefaultSpikeNetTransformerInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    @property
    def _dt(self):
        if self.dt <= 0.0:
            self.dt = self.tvb_dt
        return self.dt

    @property
    def J_N(self):
        return self.tvb_simulator_serialized["model.J_N"]

    @property
    def tau_s(self):
        return self.tvb_simulator_serialized["model.tau_s"]

    @property
    def gamma(self):
        return self.tvb_simulator_serialized["model.gamma"]

    def default_output_config(self):
        BasalGangliaIzhikevichSpikeNetInterfaceBuilder.default_input_config(self)
        BasalGangliaIzhikevichSpikeNetToTVBTransformerBuilder.default_spikeNet_to_tvb_config(self)

    def default_input_config(self):
        BasalGangliaIzhikevichSpikeNetInterfaceBuilder.default_input_config(self)
        BasalGangliaIzhikevichTVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config(self)


class BasalGangliaIzhikevichTVBSpikeNetInterfaceBuilder(BasalGangliaIzhikevichTVBInterfaceBuilder,
                                                        BasalGangliaIzhikevichTVBtoSpikeNetTransformerBuilder,
                                                        BasalGangliaIzhikevichSpikeNetToTVBTransformerBuilder,
                                                        BasalGangliaIzhikevichSpikeNetProxyNodesBuilder,
                                                        DefaultTVBSpikeNetInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    @property
    def _dt(self):
        if self.dt <= 0.0:
            self.dt = self.tvb_dt
        return self.dt

    @property
    def nsig(self):
        return self.tvb_nsig

    @property
    def G(self):
        return self.tvb_model.G

    @property
    def J_N(self):
        return self.tvb_model.J_N

    @property
    def tau_s(self):
        return self.tvb_model.tau_s

    @property
    def gamma(self):
        return self.tvb_model.gamma

    def default_output_config(self):
        BasalGangliaIzhikevichTVBInterfaceBuilder.default_output_config(self)
        BasalGangliaIzhikevichTVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config(self,
                                                                                             self.output_interfaces)
        for ii, (pop, spiking_proxy_inds) in enumerate(zip(["E", ("IdSN", "IiSN")],
                                                           [self.E_proxy_inds, self.Striatum_proxy_inds])):
            self._get_output_interfaces(ii)["populations"] = np.array(pop)
            self._get_output_interfaces(ii)["spiking_proxy_inds"] = np.array(spiking_proxy_inds)
        BasalGangliaIzhikevichSpikeNetInterfaceBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)

    def default_input_config(self):
        BasalGangliaIzhikevichTVBInterfaceBuilder.default_input_config(self)
        BasalGangliaIzhikevichSpikeNetToTVBTransformerBuilder.default_spikeNet_to_tvb_config(self,
                                                                                             self.input_interfaces)
        for ii, (pop, spiking_proxy_inds) in \
                enumerate(zip(["E", "I", ("IdSN", "IiSN")],
                              [self.E_proxy_inds, self.I_proxy_inds, self.Striatum_proxy_inds])):
            self._get_input_interfaces(ii)["populations"] = np.array(pop)
            self._get_input_interfaces(ii)["proxy_inds"] = np.array(spiking_proxy_inds)
        BasalGangliaIzhikevichSpikeNetInterfaceBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)
