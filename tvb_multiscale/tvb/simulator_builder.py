# -*- coding: utf-8 -*-

from six import string_types

import numpy as np

from tvb_multiscale.config import CONFIGURED

from tvb_scripts.utils.data_structures_utils import ensure_list

from tvb.simulator.simulator import Simulator
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.monitors import Raw  # , Bold  # , EEG
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI


class SimulatorBuilder(object):

    connectivity = CONFIGURED.DEFAULT_CONNECTIVITY_ZIP
    scale_connectivity_weights = "region"
    delays_flag = True
    model = ReducedWongWangExcIOInhI
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
        if isinstance(self.scale_connectivity_weights, string_types):
            connectivity.weights = connectivity.scaled_weights(mode=self.scale_connectivity_weights)
        if self.scale_connectivity_weights_by_percentile is not None:
            connectivity.weights /= np.percentile(connectivity.weights, self.scale_connectivity_weights_by_percentile)
        if not self.delays_flag:
            connectivity.configure()  # to set speed
            # Given that
            # idelays = numpy.rint(delays / dt).astype(numpy.int32)
            # and delays = tract_lengths / speed
            connectivity.tract_lengths = 0.1 * self.dt * connectivity.speed
        connectivity.configure()

        # Build model:
        model = self.model(**model_params)

        # Build integrator
        integrator = self.integrator(dt=self.dt)
        integrator.noise.nsig = np.array(ensure_list(self.noise_strength))

        # Build monitors:
        assert Raw in self.monitors
        monitors = []
        for monitor in self.monitors:
            monitors.append(monitor(period=self.monitor_period))
        monitors = tuple(monitors)

        # Build simulator
        simulator = Simulator()

        simulator._config = self.config
        simulator.connectivity = connectivity
        simulator.model = model
        simulator.integrator = integrator
        simulator.monitors = monitors

        return simulator
