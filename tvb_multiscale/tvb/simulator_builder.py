# -*- coding: utf-8 -*-

from six import string_types

import numpy as np

from tvb_multiscale.config import CONFIGURED

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.cosimulator import CoSimulator
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.monitors import Raw  # , Bold  # , EEG
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


class SimulatorBuilder(object):
    cosimulation = True
    connectivity = CONFIGURED.DEFAULT_CONNECTIVITY_ZIP
    scale_connectivity_weights = "region"
    scale_connectivity_normalize = "region"
    scale_connectivity_weights_by_percentile = 95
    ceil_connectivity = 1.0
    symmetric_connectome = False
    remove_self_connections = False
    delays_flag = True
    model = ReducedWongWangExcIOInhI
    variables_of_interest = None
    integrator = HeunStochastic
    dt = 0.1
    noise_strength = 0.001
    monitors = (Raw, )
    monitor_period = 1.0
    config = CONFIGURED

    def __init__(self):
        self.config = CONFIGURED
        self.connectivity = CONFIGURED.DEFAULT_CONNECTIVITY_ZIP
        self.scale_connectivity_normalize = "region"
        self.scale_connectivity_weights_by_percentile = 95
        self.ceil_connectivity = 1.0
        self.symmetric_connectome = False
        self.delays_flag = True
        self.model = ReducedWongWangExcIOInhI
        self.integrator = HeunStochastic
        self.dt = 0.1
        self.noise_strength = 0.001
        self.monitor_period = 1.0

    def build(self, **model_params):
        # Load, normalize and configure connectivity
        if isinstance(self.connectivity, string_types):
            connectivity = Connectivity.from_file(self.connectivity)
        else:
            connectivity = self.connectivity
        # Given that
        # idelays = numpy.rint(delays / dt).astype(numpy.int32)
        # and delays = tract_lengths / speed
        minimum_tract_length = 0.1 * self.dt * connectivity.speed
        if self.remove_self_connections:
            np.fill_diagonal(connectivity.weights, 0.0)
            np.fill_diagonal(connectivity.tract_lengths, minimum_tract_length)
        if isinstance(self.scale_connectivity_weights, string_types):
            connectivity.weights = connectivity.scaled_weights(mode=self.scale_connectivity_weights)
        if self.symmetric_connectome:
            connectivity.weights = np.sqrt(connectivity.weights * connectivity.weights.T)
            connectivity.tract_lengths = np.sqrt(connectivity.tract_lengths * connectivity.tract_lengths.T)
        if self.scale_connectivity_weights_by_percentile is not None:
            connectivity.weights /= np.percentile(connectivity.weights, self.scale_connectivity_weights_by_percentile)
        if self.ceil_connectivity and self.ceil_connectivity > 0.0:
            connectivity.weights[connectivity.weights > self.ceil_connectivity] = self.ceil_connectivity
        if not self.delays_flag:
            connectivity.configure()  # to set speed
            connectivity.tract_lengths = minimum_tract_length * np.ones(connectivity.tract_lengths.shape)
        connectivity.configure()

        # Build model:
        model = self.model(**model_params)
        if self.variables_of_interest is not None:
            model.variables_of_interest = self.variables_of_interest

        # Build integrator
        integrator = self.integrator(dt=self.dt)
        integrator.noise.nsig = np.array(ensure_list(self.noise_strength))

        # Build monitors:
        monitors = []
        for monitor in self.monitors:
            monitors.append(monitor(period=self.monitor_period))
        monitors = tuple(monitors)

        # Build simulator
        simulator = CoSimulator()

        simulator._config = self.config
        simulator.connectivity = connectivity
        simulator.model = model
        simulator.integrator = integrator
        simulator.monitors = monitors

        return simulator
