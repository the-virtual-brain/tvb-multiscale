# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, ABC
from enum import Enum

import numpy as np
from tvb.basic.neotraits._attr import Attr, Float, NArray

from tvb_multiscale.core.config import CONFIGURED
from tvb_multiscale.core.interfaces.base.transformers.models.base import \
    Transformer, LinearRate, LinearCurrent, LinearPotential, SpikesToRates
from tvb_multiscale.core.interfaces.base.transformers.models.elephant import \
    ElephantSpikesHistogram, ElephantSpikesRate, ElephantSpikesHistogramRate


class Integration(Transformer):
    __metaclass__ = ABCMeta

    """Integration class is a base class for transformers that perform some integration."""

    from tvb.simulator.integrators import Integrator, IntegratorStochastic
    _stochastic_integrator = IntegratorStochastic

    dt = Float(label="Time step",
               doc="Time step of simulation",
               required=True,
               default=0.1)

    state = NArray(
        label="State",
        doc="""Current state (originally initial condition) of state variable.""",
        required=True,
        default=np.array([[0.0]])
    )

    integrator = Attr(
        field_type=Integrator,
        label="Integration scheme",
        default=CONFIGURED.DEFAULT_TRANSFORMER_INTEGRATOR_MODEL(dt=CONFIGURED.DEFAULT_DT),
        required=True,
        doc="""A tvb.simulator.Integrator object which is
                an integration scheme with supporting attributes such as
                integration step size and noise specification for stochastic
                methods. It is used to compute the time courses of the model state
                variables.""")

    def __init__(self, **kwargs):
        from tvb.simulator.integrators import Integrator, IntegratorStochastic
        self.config = kwargs.get("config", CONFIGURED)
        self.dt = self.config.DEFAULT_DT
        self.integrator = self.config.DEFAULT_TRANSFORMER_INTEGRATOR_MODEL(dt=self.dt)
        super(Integration, self).__init__(**kwargs)

    @property
    def _state(self):
        return self._assert_size("state", dim=1)

    def configure(self):
        self.integrator.dt = self.dt
        self.integrator.configure()
        if isinstance(self.integrator, self._stochastic_integrator):
            self.integrator.noise.dt = self.dt
            self.integrator.noise.configure()
            if self.integrator.noise.ntau > 0.0:
                self.integrator.noise.configure_coloured(self.integrator.dt, self._state.shape)
            else:
                self.integrator.noise.configure_white(self.integrator.dt, self._state.shape)
        super(Integration, self).configure()

    @abstractmethod
    def dfun(self, X, coupling=0.0, local_coupling=0.0, stimulus=0.0):
        """Dynamical flow function following TVB model.dfun"""
        pass

    def compute_next_step(self, input_buffer_element):
        self.state = \
            self.integrator.scheme(self._state, self.dfun,  # X, dfun,
                                   0.0, input_buffer_element, 0.0)  # coupling, local_coupling -> input buffer, stimulus
        self.state = self.apply_boundaries()

    def apply_boundaries(self):
        return self._state

    def transpose_input(self, buffer):
        return np.transpose(buffer, (1, 2, 0))

    def transpose_output(self, buffer):
        return np.transpose(buffer, (2, 0, 1))

    def loop_integrate(self, input_buffer):
        output_buffer = []
        if input_buffer.ndim < 3:
            input_buffer = input_buffer[:, :, None]  # (proxy, time, voi)
        input_buffer = self.transpose_input(input_buffer)  # (proxy, time, voi) -> (time, voi, proxy)
        for iT in range(input_buffer.shape[0]):
            self.compute_next_step(input_buffer[iT])  # (voi, proxy)
            output_buffer.append(self._state)  # (voi, proxy)
        output_buffer = np.array(output_buffer)  # (time, voi, proxy)
        return self.transpose_output(output_buffer)   # (time, voi, proxy) -> (proxy, time, voi)

    def _compute(self, input_buffer, *args, **kwargs):
        """Method for the integration on the input buffer data for the output buffer data ot result."""
        return self.loop_integrate(np.array(input_buffer))


class LinearIntegration(Integration):

    """Integration class is an elementary implementation of a transformer that performs linear integration,
       driven by input buffer data and using TVB integrators, and optionally TVB model dfuns."""

    tau = NArray(
        label=r":math:`\tau`",
        default=np.array([10., ]),
        doc="""[ms]. Time constant.""")

    @property
    def _tau(self):
        return self._assert_size("tau", dim=1)

    def configure(self):
        Integration.configure(self)
        self._tau

    def dfun(self, X, coupling=0.0, local_coupling=0.0, stimulus=0.0, **kwargs):
        return (-X + local_coupling) / self.tau


class LinearIntegrationRate(LinearIntegration, LinearRate):

    """LinearIntegrationRate class is an elementary implementation of a transformer that performs linear integration,
       driven by input buffer (linearly transformed) rate data
       and using TVB integrators, and optionally TVB model dfuns."""

    def configure(self):
        LinearRate.configure(self)
        LinearIntegration.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        return LinearIntegration._compute(self, LinearRate._compute(self, input_buffer), *args, **kwargs)


class LinearIntegrationCurrent(LinearIntegration, LinearCurrent):

    """LinearIntegrationCurrent class is an elementary implementation of a transformer that performs linear integration,
       driven by input buffer (linearly transformed) current data
       and using TVB integrators, and optionally TVB model dfuns."""

    def configure(self):
        LinearCurrent.configure(self)
        LinearIntegration.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        return LinearIntegration._compute(self, LinearCurrent._compute(self, input_buffer), *args, **kwargs)


class LinearIntegrationPotential(LinearIntegration, LinearPotential):

    """LinearIntegrationPotential class is an elementary implementation of a transformer that performs linear integration,
       driven by input buffer (linearly transformed) potential data
       and using TVB integrators, and optionally TVB model dfuns."""

    def configure(self):
        LinearPotential.configure(self)
        LinearIntegration.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        return LinearIntegration._compute(self, LinearPotential._compute(self, input_buffer), *args, **kwargs)


class SpikesToRateLinearIntegration(SpikesToRates, LinearIntegration, ABC):
    __metaclass__ = ABCMeta

    """SpikesToRateLinearIntegration class is an elementary implementation of a transformer 
       that performs linear integration, 
       driven by input buffer (linearly transformed) rate data,
       computed from spike trains, and using TVB integrators, and optionally TVB model dfuns."""

    def configure(self):
        super().configure()
        LinearIntegration.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        return LinearIntegration._compute(self,
                                          super()._compute(input_buffer), *args, **kwargs)


class ElephantSpikesHistogramLinearIntegration(ElephantSpikesHistogram, SpikesToRateLinearIntegration):

    """ElephantSpikesHistogramLinearIntegration class is an elementary implementation of a transformer
       that performs linear integration, driven by input buffer (linearly transformed) spikes' histogram data,
       computed via the function time_histogram of elephant software,
        and using TVB integrators, and optionally TVB model dfuns."""

    def configure(self):
        SpikesToRateLinearIntegration.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        return SpikesToRateLinearIntegration._compute(self, input_buffer, *args, **kwargs)


class ElephantSpikesHistogramRateLinearIntegration(ElephantSpikesHistogramRate, SpikesToRateLinearIntegration):
    """ElephantSpikesHistogramRateLinearIntegration class is an elementary implementation of a transformer
       that performs linear integration, driven by input buffer (linearly transformed) spikes' rate data,
       computed via the function time_histogram of elephant software,
       and using TVB integrators, and optionally TVB model dfuns."""

    def configure(self):
        SpikesToRateLinearIntegration.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        return SpikesToRateLinearIntegration._compute(self, input_buffer, *args, **kwargs)


class ElephantSpikesRateLinearIntegration(ElephantSpikesRate, SpikesToRateLinearIntegration):
    """ElephantSpikesRateLinearIntegration class is an elementary implementation of a transformer
       that performs linear integration, driven by input buffer (linearly transformed) spikes' data,
       computed via the function instantaneous_rate of elephant software,
       and using TVB integrators, and optionally TVB model dfuns."""

    def configure(self):
        SpikesToRateLinearIntegration.configure(self)

    def _compute(self, input_buffer, *args, **kwargs):
        return SpikesToRateLinearIntegration._compute(self, input_buffer, *args, **kwargs)


class IntegrationTransformers(Enum):
    LINEAR_INTEGRATION_RATE = LinearIntegrationRate
    LINEAR_INTEGRATION_CURRENT = LinearIntegrationCurrent
    LINEAR_INTEGRATION_POTENTIAL = LinearIntegrationPotential
    ELEPHANT_SPIKES_HISTOGRAM_LINEAR_INTEGRATION = ElephantSpikesHistogramLinearIntegration
    ELEPHANT_SPIKES_HISTOGRAM_RATE_LINEAR_INTEGRATION = ElephantSpikesHistogramRateLinearIntegration
    ELEPHANT_SPIKES_RATE_LINEAR_INTEGRATION = ElephantSpikesRateLinearIntegration
