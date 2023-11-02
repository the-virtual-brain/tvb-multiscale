# -*- coding: utf-8 -*-

from abc import ABCMeta, ABC, abstractmethod

import numpy as np

from tvb.basic.neotraits._attr import Int, Float, NArray
from tvb.simulator.integrators import HeunStochastic, HeunDeterministic
from tvb.simulator.noise import Additive

from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels
from tvb_multiscale.core.interfaces.models.default import DefaultInterfaceBuilder, DefaultTVBInterfaceBuilder, \
    DefaultTVBSpikeNetInterfaceBuilder, DefaultSpikeNetInterfaceBuilder, DefaultSpikeNetProxyNodesBuilder
from tvb_multiscale.core.interfaces.transformers.models.red_wong_wang import \
    ElephantSpikesRateRedWongWangExc, ElephantSpikesRateRedWongWangInh


class RedWongWangExcIOTVBInterfaceBuilder(DefaultTVBInterfaceBuilder):

    def default_output_config(self):
        self._get_output_interfaces()["voi"] = "S" if self.model == TVBtoSpikeNetModels.CURRENT.name else "R"

    def default_input_config(self):
        self._get_input_interfaces()["voi"] = ("S", "R")


class RedWongWangExcIOTVBtoSpikeNetTransformerBuilder(DefaultInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    N_E = Int(field_type=int, label="N_E", default=100, doc="""Number of excitatory neurons""")

    @property
    @abstractmethod
    def J_N(self):
        pass

    def default_tvb_to_spikeNet_config(self, interfaces):
        for interface in interfaces:
            if self.model == TVBtoSpikeNetModels.CURRENT.name:
                interface["transformer_params"] = {"scale_factor": self.N_E*self.J_N}
            elif self.model == TVBtoSpikeNetModels.SPIKES.name:
                interface["transformer_params"] = {"scale_factor": np.array([1.0]),
                                                   "number_of_neurons": np.array([self.N_E])}
            else:  # RATE
                interface["transformer_params"] = {"scale_factor": self.N_E*np.array([1.0])}


class RedWongWangExcIOSpikeNetToTVBTransformerBuilder(DefaultInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    N_E = Int(field_type=int, label="N_E", default=100, doc="""Number of excitatory neurons""")

    tau_r = NArray(
        label=r":math:`\tau_R`",
        default=np.array([10., ]),
        doc="""[ms]. Excitatory population  input rate decay time constant.""")

    proxy_inds = NArray(
        dtype=int,
        label="proxy_inds",
        doc="""Indices of Spiking Network proxy nodes""",
        required=True,
    )

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True, default=0.0)

    @property
    @abstractmethod
    def _dt(self):
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
        for interface in interfaces:
            interface["transformer_model"] = ElephantSpikesRateRedWongWangExc
            interface["transformer_params"] = \
                {"scale_factor": np.array([1.0]) / self.N_E,
                 "state": np.zeros((2, len(self.proxy_inds))),
                 "tau_s": self.tau_s, "tau_r": self.tau_r, "gamma": self.gamma}


class RedWongWangExcIOSpikeNetProxyNodesBuilder(DefaultSpikeNetProxyNodesBuilder, ABC):

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


class RedWongWangExcIOSpikeNetInterfaceBuilder(RedWongWangExcIOSpikeNetProxyNodesBuilder,
                                               DefaultSpikeNetInterfaceBuilder, ABC):

    @property
    def G(self):
        return self.tvb_simulator_serialized["model.G"]

    def default_output_config(self):
        self._get_output_interfaces()["populations"] = "E"
        RedWongWangExcIOSpikeNetProxyNodesBuilder.default_spikeNet_to_tvb_config(self, self.output_interfaces)

    def default_input_config(self):
        self._get_input_interfaces()["populations"] = "E"
        RedWongWangExcIOSpikeNetProxyNodesBuilder.default_tvb_to_spikeNet_config(self, self.input_interfaces)


class RedWongWangExcIOTVBSpikeNetInterfaceBuilder(RedWongWangExcIOTVBInterfaceBuilder,
                                                  RedWongWangExcIOTVBtoSpikeNetTransformerBuilder,
                                                  RedWongWangExcIOSpikeNetToTVBTransformerBuilder,
                                                  RedWongWangExcIOSpikeNetProxyNodesBuilder,
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
        RedWongWangExcIOTVBInterfaceBuilder.default_output_config(self)
        RedWongWangExcIOTVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)
        self._get_output_interfaces()["populations"] = "E"
        RedWongWangExcIOSpikeNetProxyNodesBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)

    def default_input_config(self):
        RedWongWangExcIOTVBInterfaceBuilder.default_input_config(self)
        RedWongWangExcIOSpikeNetToTVBTransformerBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)
        self._get_input_interfaces()["populations"] = "E"
        RedWongWangExcIOSpikeNetProxyNodesBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)


class RedWongWangExcIOInhITVBInterfaceBuilder(DefaultTVBInterfaceBuilder):

    @property
    def lamda(self):
        return self.tvb_model.lamda

    def default_output_config(self):
        var = "S_e" if self.model == TVBtoSpikeNetModels.CURRENT.name else "R_e"
        self._get_output_interfaces(0)["voi"] = var
        if self.lamda > 0.0:
            self._get_output_interfaces(1)["voi"] = var

    def default_input_config(self):
        self._get_input_interfaces(0)["voi"] = ("S_e", "R_e")
        self._get_input_interfaces(1)["voi"] = ("S_i", "R_i")


class RedWongWangExcIOInhITVBtoSpikeNetTransformerBuilder(DefaultInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    N_E = Int(field_type=int, label="N_E", default=160, doc="""Number of excitatory neurons""")

    @property
    @abstractmethod
    def lamda(self):
        pass

    @property
    @abstractmethod
    def J_N(self):
        pass

    def default_tvb_to_spikeNet_config(self, interfaces):
        if self.model == TVBtoSpikeNetModels.CURRENT.name:
            transformer_params = {"scale_factor": self.J_N}
        elif self.model == TVBtoSpikeNetModels.SPIKES.name:
            transformer_params = {"scale_factor": np.array([1.0]), "number_of_neurons": np.array([self.N_E])}
        else:  # RATE
            transformer_params = {"scale_factor": self.N_E*np.array([1.0])}
        interfaces[0]["transformer_params"] = transformer_params
        if self.lamda > 0.0:
            interfaces[1]["transformer_params"] = transformer_params.copy()
            # TODO: Find an elegant way to pass this to the weight of the connectivity scaling
            interfaces[1]["transformer_params"]["scale_factor"] *= self.lamda


class RedWongWangExcIOInhISpikeNetToTVBTransformerBuilder(DefaultInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    N_E = Int(field_type=int, label="N_E", default=160, doc="""Number of excitatory neurons""")

    N_I = Int(field_type=int, label="N_I", default=40, doc="""Number of inhibitory neurons""")

    tau_re = NArray(
        label=r":math:`\tau_re`",
        default=np.array([10., ]),
        doc="""[ms]. Excitatory population input rate decay time constant.""")

    tau_ri = NArray(
        label=r":math:`\tau_ri`",
        default=np.array([10., ]),
        doc="""[ms]. Inhibitory population input rate decay time constant.""")

    proxy_inds = NArray(
        dtype=int,
        label="Indices of Spiking Network proxy nodes",
        doc="""Indices of Spiking Network proxy nodes""",
        required=True,
    )

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True, default=0.0)

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
    def tau_e(self):
        pass

    @property
    @abstractmethod
    def tau_i(self):
        pass

    @property
    @abstractmethod
    def gamma_e(self):
        pass

    @property
    @abstractmethod
    def gamma_i(self):
        pass

    def default_spikeNet_to_tvb_config(self, interfaces):
        for interface, model, N, tau_s, tau_r, gamma in \
                zip(interfaces,
                    [ElephantSpikesRateRedWongWangExc, ElephantSpikesRateRedWongWangInh],
                    [self.N_E, self.N_I],
                    [self.tau_e, self.tau_i], [self.tau_re, self.tau_ri], [self.gamma_e, self.gamma_i]):
            interface["transformer_model"] = model
            interface["transformer_params"] = \
                {"scale_factor": np.array([1.0]) / N,
                 "state": np.zeros((2, len(self.proxy_inds))),
                 "tau_s": tau_s, "tau_r": tau_r, "gamma": gamma}


class RedWongWangExcIOInhISpikeNetProxyNodesBuilder(DefaultSpikeNetProxyNodesBuilder, ABC):

    N_E = Int(field_type=int, label="Number of regions", default=160,
              doc="""The number of regions represented in this Connectivity""")

    N_I = Int(field_type=int, label="Number of regions", default=40,
              doc="""The number of regions represented in this Connectivity""")

    @property
    @abstractmethod
    def G(self):
        pass

    @property
    @abstractmethod
    def lamda(self):
        pass

    def _configure_global_coupling_scaling(self):
        if self.global_coupling_scaling.size == 0:
            self.global_coupling_scaling = self.tvb_coupling_a * self.G
        DefaultSpikeNetInterfaceBuilder._configure_global_coupling_scaling(self)

    def default_tvb_to_spikeNet_config(self, interfaces):
        if self.model == TVBtoSpikeNetModels.SPIKES.name:
            proxy_params = {"number_of_neurons": self.N_E}
        elif self.model == TVBtoSpikeNetModels.RATE.name:
            proxy_params = {"number_of_neurons": 1}
        else:
            proxy_params = {}
        interfaces[0]["proxy_params"] = proxy_params
        if np.any(self.lamda > 0.0):
            interfaces[1]["proxy_params"] = proxy_params.copy()


class RedWongWangExcIOInhISpikeNetInterfaceBuilder(RedWongWangExcIOInhISpikeNetProxyNodesBuilder,
                                                   DefaultSpikeNetInterfaceBuilder, ABC):

    @property
    def G(self):
        return self.tvb_simulator_serialized["model.G"]

    @property
    def lamda(self):
        return self.tvb_simulator_serialized["model.lamda"]

    def default_output_config(self):
        self._get_output_interfaces(0)["populations"] = "E"
        self._get_output_interfaces(1)["populations"] = "I"
        RedWongWangExcIOInhISpikeNetProxyNodesBuilder.default_spikeNet_to_tvb_config(self, self.output_interfaces)

    def default_input_config(self):
        self._get_input_interfaces()["populations"] = "E"
        if np.any(self.lamda > 0.0):
            self._get_input_interfaces(1)["populations"] = "I"
        RedWongWangExcIOInhISpikeNetProxyNodesBuilder.default_tvb_to_spikeNet_config(self, self.input_interfaces)


class RedWongWangExcIOInhITVBSpikeNetInterfaceBuilder(RedWongWangExcIOInhITVBInterfaceBuilder,
                                                      RedWongWangExcIOInhITVBtoSpikeNetTransformerBuilder,
                                                      RedWongWangExcIOInhISpikeNetToTVBTransformerBuilder,
                                                      RedWongWangExcIOInhISpikeNetProxyNodesBuilder,
                                                      DefaultTVBSpikeNetInterfaceBuilder, ABC):
    __metaclass__ = ABCMeta

    @property
    def _dt(self):
        if self.dt <= 0.0:
            self.dt = self.tvb_dt
        return self.dt

    @property
    def nsig(self):
        return self.tvb_cosimulator.integrator.noise.nsig

    @property
    def G(self):
        return self.tvb_model.G

    @property
    def lamda(self):
        return self.tvb_model.lamda

    @property
    def J_N(self):
        return self.tvb_model.J_N

    @property
    def tau_e(self):
        return self.tvb_model.tau_e

    @property
    def tau_i(self):
        return self.tvb_model.tau_i

    @property
    def gamma_e(self):
        return self.tvb_model.gamma_e

    @property
    def gamma_i(self):
        return self.tvb_model.gamma_i

    def default_output_config(self):
        RedWongWangExcIOInhITVBInterfaceBuilder.default_output_config(self)
        RedWongWangExcIOInhITVBtoSpikeNetTransformerBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)
        self._get_output_interfaces()["populations"] = "E"
        if np.any(self.lamda > 0.0):
            self._get_output_interfaces(1)["populations"] = "I"
        RedWongWangExcIOInhISpikeNetProxyNodesBuilder.default_tvb_to_spikeNet_config(self, self.output_interfaces)

    def default_input_config(self):
        RedWongWangExcIOInhITVBInterfaceBuilder.default_input_config(self)
        RedWongWangExcIOInhISpikeNetToTVBTransformerBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)
        self._get_input_interfaces(0)["populations"] = "E"
        self._get_input_interfaces(1)["populations"] = "I"
        RedWongWangExcIOInhISpikeNetProxyNodesBuilder.default_spikeNet_to_tvb_config(self, self.input_interfaces)
