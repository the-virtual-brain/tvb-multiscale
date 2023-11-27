# -*- coding: utf-8 -*-

import numpy as np

from tvb.basic.neotraits._attr import NArray

from tvb_multiscale.core.interfaces.transformers.models.integration import Integration
from tvb_multiscale.core.interfaces.transformers.models.elephant import \
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
                         - (X[1] - np.array(input_buffer).flatten())/self._tau_r])

    def apply_boundaries(self):
        # Apply boundaries:
        self.state = np.where(self.state < 0.0, 0.0, self.state)           # S, R >= 0.0
        self.state[0] = np.where(self.state[0] > 1.0, 1.0, self.state[0])  # S <= 1.0
        return self._state


class ElephantSpikesHistogramRateRedWongWangExc(ElephantSpikesHistogramRate, RedWongWangExc):

    def configure(self):
        ElephantSpikesHistogramRate.configure(self)
        RedWongWangExc.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of synaptic activity and instantaneous mean spiking rates to result."""
        return RedWongWangExc._compute(self, ElephantSpikesHistogramRate._compute(self, input_buffer))


class ElephantSpikesRateRedWongWangExc(ElephantSpikesRate, RedWongWangExc):

    def configure(self):
        ElephantSpikesRate.configure(self)
        RedWongWangExc.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of synaptic activity and instantaneous mean spiking rates to result."""
        return RedWongWangExc._compute(self, ElephantSpikesRate._compute(self, input_buffer))


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
                         - (X[1] - np.array(input_buffer).flatten())/self._tau_r])


class ElephantSpikesHistogramRateRedWongWangInh(ElephantSpikesHistogramRate, RedWongWangInh):

    def configure(self):
        ElephantSpikesHistogramRate.configure(self)
        RedWongWangInh.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of synaptic activity and instantaneous mean spiking rates to result."""
        return RedWongWangInh._compute(self, ElephantSpikesHistogramRate._compute(self, input_buffer, *args, **kwargs))


class ElephantSpikesRateRedWongWangInh(ElephantSpikesRate, RedWongWangInh):

    def configure(self):
        ElephantSpikesRate.configure(self)
        RedWongWangInh.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        """Method for the computation on the input buffer spikes' trains' data
           for the output buffer data of synaptic activity and instantaneous mean spiking rates to result."""
        return RedWongWangInh._compute(self, ElephantSpikesRate._compute(self, input_buffer, *args, **kwargs))
