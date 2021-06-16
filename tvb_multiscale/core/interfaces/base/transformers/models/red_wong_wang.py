# -*- coding: utf-8 -*-

from abc import ABCMeta

import numpy as np

from tvb.basic.neotraits._attr import NArray

from tvb_multiscale.core.interfaces.base.transformers.models.base import Integration
from tvb_multiscale.core.interfaces.base.transformers.models.elephant import \
    ElephantSpikesHistogramRate, ElephantSpikesRate


class RedWongWangExc(Integration):

    tau_s = NArray(
        label=r":math:`\tau_S`",
        default=np.array([100., ]),
        doc="""[ms]. Excitatory population NMDA decay time constant.""")

    tau_r = NArray(
        label=r":math:`\tau_R`",
        default=np.array([10., ]),
        doc="""[ms]. Excitatory population  input rate decay time constant.""")

    gamma = NArray(
        label=r":math:`\gamma`",
        default=np.array([0.641 / 1000, ]),
        doc="""Excitatory population kinetic parameter""")

    @property
    def _tau_s(self):
        return self._assert_size("tau_s")

    @property
    def _tau_r(self):
        return self._assert_size("tau_r")

    @property
    def _gamma(self):
        return self._assert_size("gamma")

    def configure(self):
        super(RedWongWangExc, self).configure()
        assert self._state.shape[0] == 2

    def dfun(self, X, coupling=0.0, input_buffer=0.0, stimulus=0.0):
        # Synaptic gating dynamics
        # dS = - (S / self.tau_s) + (1 - S) * R * self.gamma
        # dR = -(R - Rin) / tau_r
        return np.array([- (X[0] / self._tau_s) + (1 - X[0]) * X[1] * self._gamma,
                         - (X[1] - input_buffer)/self._tau_r])

    def apply_boundaries(self):
        # Apply boundaries:
        self.state = np.where(self.state < 0.0, 0.0, self.state)           # S, R >= 0.0
        self.state[0] = np.where(self.state[0] > 1.0, 1.0, self.state[0])  # S <= 1.0
        return self._state

    def transpose(self):
        self.output_buffer = self.output_buffer.T  # (time, voi, proxy) -> (proxy, voi, time)
        return self.output_buffer

    def print_str(self):
        return super(RedWongWangExc, self).print_str() + \
               "\n     - tau_s = %s" % str(self.tau_s) + \
               "\n     - tau_r = %s" % str(self.tau_r) + \
               "\n     - gamma = %s" % str(self.gamma)


class ElephantSpikesHistogramRateRedWongWangExc(ElephantSpikesHistogramRate, RedWongWangExc):

    def configure(self):
        ElephantSpikesHistogramRate.configure(self)
        RedWongWangExc.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of synaptic activity and instantaneous mean spiking rates to result."""
        return RedWongWangExc.compute(self, input_buffer=ElephantSpikesHistogramRate.compute(self))

    def print_str(self):
        return RedWongWangExc.print_str(self) + ElephantSpikesHistogramRate.print_str(self)


class ElephantSpikesRateRedWongWangExc(ElephantSpikesRate, RedWongWangExc):

    def configure(self):
        ElephantSpikesRate.configure(self)
        RedWongWangExc.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of synaptic activity and instantaneous mean spiking rates to result."""
        input_buffer = ElephantSpikesRate.compute(self)
        return RedWongWangExc.compute(self, input_buffer=input_buffer)

    def print_str(self):
        return RedWongWangExc.print_str(self) + ElephantSpikesRate.print_str(self)


class RedWongWangInh(RedWongWangExc):

    tau_s = NArray(
        label=r":math:`\tau_S`",
        default=np.array([10., ]),
        doc="""[ms]. Inhibitory population NMDA decay time constant.""")

    tau_r = NArray(
        label=r":math:`\tau_R`",
        default=np.array([10., ]),
        doc="""[ms]. Inhibitory population input rate decay time constant.""")

    gamma = NArray(
        label=r":math:`\gamma`",
        default=np.array([1.0 / 1000, ]),
        doc="""Inhibitory population kinetic parameter""")

    def dfun(self, X, coupling=0.0, input_buffer=0.0, stimulus=0.0):
        # Synaptic gating dynamics
        # dS = - (S / self.tau_s) + R * self.gamma
        # dR = -(R - Rin) / tau_r
        return np.array([- (X[0] / self._tau_s) + X[1] * self._gamma,
                         - (X[1] - input_buffer)/self._tau_r])


class ElephantSpikesHistogramRateRedWongWangInh(ElephantSpikesHistogramRate, RedWongWangInh):

    def configure(self):
        ElephantSpikesHistogramRate.configure(self)
        RedWongWangInh.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of synaptic activity and instantaneous mean spiking rates to result."""
        return RedWongWangInh.compute(self, input_buffer=ElephantSpikesHistogramRate.compute(self))

    def print_str(self):
        return RedWongWangInh.print_str(self) + ElephantSpikesHistogramRate.print_str(self)


class ElephantSpikesRateRedWongWangInh(ElephantSpikesRate, RedWongWangInh):

    def configure(self):
        ElephantSpikesRate.configure(self)
        RedWongWangInh.configure(self)

    def compute(self, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of synaptic activity and instantaneous mean spiking rates to result."""
        input_buffer = ElephantSpikesRate.compute(self)
        return RedWongWangInh.compute(self, input_buffer=input_buffer)

    def print_str(self):
        return RedWongWangInh.print_str(self) + ElephantSpikesRate.print_str(self)
