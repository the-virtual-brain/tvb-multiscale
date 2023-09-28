# -*- coding: utf-8 -*-

from logging import Logger
from six import string_types

import numpy as np

from tvb.basic.neotraits.api import Attr, Float, NArray, List
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.coupling import Coupling
from tvb.simulator.models.base import Model
from tvb.simulator.integrators import Integrator, IntegratorStochastic
from tvb.simulator.monitors import Monitor, Raw, Bold  # , EEG
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb_multiscale.core.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.tvb.cosimulator.cosimulator import CoSimulator
from tvb_multiscale.core.tvb.cosimulator.cosimulator_serial import CoSimulatorSerial, CoSimulatorNetpyne
from tvb_multiscale.core.tvb.cosimulator.cosimulator_parallel import CoSimulatorParallel, CoSimulatorRemoteParallel


class CoSimulatorBuilder(HasTraits):

    """CoSimulatorBuilder is an opinionated builder for a TVB Simulator, adjusted for cosimulation.
       Depending on its properties set, the builder may
       - scale/normalize the connectivity weights,
       - remove time delays or not,
       - remove the self-connections or brain region nodes (diagonal of connectivity matrix)
       - set integrator (including noise and integration step),
       - set monitor (including model's variables of interest and period)
    """

    _cosimulator_type = CoSimulator

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    logger = Attr(
        label="Logger",
        field_type=Logger,
        doc="""logging.Logger instance.""",
        required=True,
        default=initialize_logger(config=CONFIGURED)
    )

    model = Attr(
        field_type=Model,
        label="Local dynamic model",
        default=CONFIGURED.DEFAULT_TVB_MODEL(),
        required=True,
        doc="""A tvb.simulator.Model object which describe the local dynamic
            equations, their parameters, and, to some extent, where connectivity
            (local and long-range) enters and which state-variables the Monitors
            monitor. By default the 'Generic2dOscillator' model is used. Read the
            Scientific documentation to learn more about this model.""")

    model_params = Attr(
        field_type=dict,
        label="Model parameters",
        default=dict(),
        required=True,
        doc="""A dictionary of model parameters""")

    connectivity = Attr(
        field_type=Connectivity,
        label="Long-range connectivity",
        default=Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP),
        required=True,
        doc="""A tvb.datatypes.Connectivity object which contains the
             structural long-range connectivity data (i.e., white-matter tracts). In
             combination with the ``Long-range coupling function`` it defines the inter-regional
             connections. These couplings undergo a time delay via signal propagation
             with a propagation speed of ``Conduction Speed``""")

    scale_connectivity_weights = Attr(label="Connectivity scaling mode",
                                      doc=""" It can be one of 'tract', 'region", or None. Default = 'region'.
                                          'tract': Scale by a value such that the maximum absolute value of a single
                                                  connection is 1.0. (Global scaling)
                                          'region': Scale by a value such that the maximum absolute value of the
                                                    cumulative input to any region is 1.0. (Global-wise scaling)""",
                                      field_type=str,
                                      default="region",
                                      required=False)

    scale_connectivity_weights_by_percentile = Float(
        label="Percentile of weights to scale",
        default=99.0,
        required=False,
        doc="""Percentile of connectivity weights to divide the whole weights' matrix for normalization. 
               Default = 95.""")

    ceil_connectivity = Float(
        label="Connectivity ceiling",
        default=0.0,
        required=False,
        doc="""Value to ceil connectivity weights to. Default = 0.0, which will not ceil connectivity""")

    symmetric_connectome = Attr(label="Symmetric connectome flag",
                                doc="""Boolean flag that if True 
                                       a symmetric connectome (weights and delays (tract_lengths)) is forced.
                                       Default = False""",
                                field_type=bool,
                                default=False,
                                required=False)

    remove_self_connections = Attr(label="Self-connections removal flag",
                                   doc="""Boolean flag that if True 
                                          self-connections of brain region nodes are forced to a zero weight.
                                          Default = True""",
                                   field_type=bool,
                                   default=False,
                                   required=True)

    delays_flag = Attr(label="Delays' flag",
                       doc="""Boolean flag that if False 
                              delays are forced to be equal to one integration time step.
                              Default = True""",
                       field_type=bool,
                       default=True,
                       required=True)

    min_tract_length = Float(
        label="Minimum tract length",
        default=0.0,
        required=True,
        doc="""Minimum tract length to be forced to the connectom. 
               If it is equal to 0.0 (default value), it will be computed as equal to the length that 
               corresponds to one integration time step, dt, given the default connectivity.speed.
               if positive, the value will be used.
               If negative, no minimum tract length will be forced.""")

    coupling = Attr(
        field_type=Coupling,
        label="Long-range coupling function",
        default=CONFIGURED.DEFAULT_TVB_COUPLING_MODEL(),
        required=True,
        doc="""The coupling function is applied to the activity propagated
            between regions by the ``Long-range connectivity`` before it enters the local
            dynamic equations of the Model. Its primary purpose is to 'rescale' the
            incoming activity to a level appropriate to Model.""")

    dt = Float(
        label="Integration-step size (ms)",
        default=0.1,
        required=True,
        doc="""The step size used by the integration routine in ms. This
            should be chosen to be small enough for the integration to be
            numerically stable. It is also necessary to consider the desired sample
            period of the Monitors, as they are restricted to being integral
            multiples of this value. The default value is set such that all built-in
            models are numerically stable with there default parameters and because
            it is consitent with Monitors using sample periods corresponding to
            powers of 2 from 128 to 4096Hz.""")

    noise_strength = NArray(
        label=":math:`D`",
        required=True,
        default=np.array([CONFIGURED.DEFAULT_NSIG]),
        doc="""The noise dispersion, it is the standard deviation of the
                distribution from which the Gaussian random variates are drawn. NOTE:
                Sensible values are typically ~<< 1% of the dynamic range of a Model's
                state variables."""
    )

    integrator = Attr(
        field_type=Integrator,
        label="Integration scheme",
        default=CONFIGURED.DEFAULT_INTEGRATOR(),
        required=True,
        doc="""A tvb.simulator.Integrator object which is
                an integration scheme with supporting attributes such as
                integration step size and noise specification for stochastic
                methods. It is used to compute the time courses of the model state
                variables.""")

    monitors = List(
        of=Monitor,
        label="Monitor(s)",
        default=(CONFIGURED.DEFAULT_MONITOR(), ),
        doc="""A tvb.simulator.Monitor or a list of tvb.simulator.Monitor
            objects that 'know' how to record relevant data from the simulation. Two
            main types exist: 1) simple, spatial and temporal, reductions (subsets
            or averages); 2) physiological measurements, such as EEG, MEG and fMRI.
            By default the Model's specified variables_of_interest are returned,
            temporally downsampled from the raw integration rate to a sample rate of
            1024Hz.""")

    monitor_period = Float(
        label="Monitor sampling period (ms)",
        default=1.0,
        doc="""For the BOLD monitor, sampling period in milliseconds must be
            an integral multiple of 500. Typical measurement interval (repetition
            time TR) is between 1-3 s. If TR is 2s, then Bold period is 2000ms.""")

    initial_conditions = NArray(
        label="Initial Conditions",
        required=False,
        doc="""Initial conditions from which the simulation will begin. By
            default, random initial conditions are provided. Needs to be the same shape
            as simulator 'history', ie, initial history function which defines the 
            minimal initial state of the network with time delays before time t=0. 
            If the number of time points in the provided array is insufficient the 
            array will be padded with random values based on the 'state_variables_range'
            attribute.""")

    simulation_length = Float(
        label="Simulation Length (ms, s, m, h)",
        default=1100.0,  # ie 1.1 second
        required=False,
        doc="""The length of a simulation (default in milliseconds).""")

    # def __init__(self, **kwargs):
    #     self.config = kwargs.get("config", CONFIGURED)
    #     init_logger = False
    #     if not isinstance(kwargs.get("logger", None), Logger):
    #         init_logger = True
    #     self.model = self.config.DEFAULT_TVB_MODEL()
    #     self.model_params = dict()
    #     self.connectivity = Connectivity.from_file(self.config.DEFAULT_CONNECTIVITY_ZIP)
    #     self.scale_connectivity_weights = "region"
    #     self.scale_connectivity_weights_by_percentile = 99.0
    #     self.ceil_connectivity = 0.0
    #     self.symmetric_connectome = False
    #     self.remove_self_connections = False
    #     self.delays_flag = True
    #     self.min_tract_length = 0.0
    #     self.coupling = self.config.DEFAULT_TVB_COUPLING_MODEL()
    #     self.dt = 0.1
    #     self.noise_strength = np.array([self.config.DEFAULT_NSIG])
    #     self.integrator = CONFIGURED.DEFAULT_INTEGRATOR()
    #     self.monitor_period = 1.0
    #     self.monitors = (CONFIGURED.DEFAULT_MONITOR(period=self.monitor_period), )
    #     self.initial_conditions = None
    #     self.simulation_length = 1100.0
    #
    #     super(CoSimulatorBuilder, self).__init__(**kwargs)
    #     if init_logger:
    #         self.logger = initialize_logger(config=self.config)

    def configure_connectivity(self):
        # Load, normalize and configure connectivity
        # Given that
        # idelays = numpy.rint(delays / dt).astype(numpy.int32)
        # and delays = tract_lengths / speed
        if self.min_tract_length < 0.0:
            # Just compute min_tract_length for other processing
            min_tract_length = self.dt * self.connectivity.speed
        else:
            if self.min_tract_length == 0.0:
                self.min_tract_length = self.dt * self.connectivity.speed.item()
            min_tract_length = self.min_tract_length
            # Force minimum tract length to connectome
            self.connectivity.tract_lengths = np.maximum(self.min_tract_length, self.connectivity.tract_lengths)
        self.connectivity.weights[np.isnan(self.connectivity.weights)] = 0.0
        if self.remove_self_connections:
            np.fill_diagonal(self.connectivity.weights, 0.0)
            np.fill_diagonal(self.connectivity.tract_lengths, min_tract_length)
        if isinstance(self.scale_connectivity_weights, string_types):
            self.connectivity.weights = self.connectivity.scaled_weights(mode=self.scale_connectivity_weights)
        if self.symmetric_connectome:
            self.connectivity.weights = np.sqrt(self.connectivity.weights * self.connectivity.weights.T)
            self.connectivity.tract_lengths = \
                np.sqrt(self.connectivity.tract_lengths * self.connectivity.tract_lengths.T)
        if self.scale_connectivity_weights_by_percentile is not None:
            self.connectivity.weights /= np.percentile(self.connectivity.weights,
                                                       self.scale_connectivity_weights_by_percentile)
        if self.ceil_connectivity and self.ceil_connectivity > 0.0:
            self.connectivity.weights[self.connectivity.weights > self.ceil_connectivity] = self.ceil_connectivity
        self.connectivity.weights[np.isnan(self.connectivity.weights)] = 0.0
        if not self.delays_flag:
            self.connectivity.configure()  # to set speed
            self.connectivity.tract_lengths = min_tract_length * np.ones(self.connectivity.tract_lengths.shape)
        self.connectivity.configure()
        return self.connectivity

    def configure_model(self, **model_params):
        # Build model:
        self.model_params.update(model_params)
        for p, pval in self.model_params.items():
            if isinstance(pval, np.ndarray):
                setattr(self.model, p, pval)
            else:
                setattr(self.model, p, np.array([pval, ]))
        return self.model

    def configure_integrator(self):
        # Build integrator
        if np.any(self.noise_strength > 0.0):
            self.integrator.noise.nsig = np.array(ensure_list(self.noise_strength))
            if self.integrator.noise.nsig.size == 1 and self.model.nvar > 1:
                self.integrator.noise.nsig = np.repeat(self.integrator.noise.nsig.size, self.model.nvar)
            if not isinstance(self.integrator, IntegratorStochastic):
                self.integrator = self.config.DEFAULT_STOCHASTIC_INTEGRATOR()
        self.integrator.dt = self.dt
        return self.integrator

    def configure_monitors(self):
        # Build monitors:
        self.monitors = list(self.monitors)
        for iM, monitor in enumerate(self.monitors):
            if isinstance(monitor, Bold):
                period = np.ceil(self.period / 500) * 500
                self.monitors[iM].period = period
            else:
                self.monitors[iM].period = self.monitor_period
        self.monitors = tuple(self.monitors)
        return self.monitors

    def configure_initial_conditions(self, simulator):
        self.initial_conditions = self.initial_conditions * np.ones((1,
                                                                     simulator.model.nvar,
                                                                     simulator.connectivity.number_of_regions,
                                                                     simulator.model.number_of_modes))
        return self.initial_conditions

    def build(self, **model_params):
        """This method will build the TVB simulator, based on the builder's properties.
           Arguments:
            - **model_params: keyword arguments to modify the default model parameters
           Returns:
            - the TVB simulator built, but not yet configured.
        """
        # Build simulator
        simulator = self._cosimulator_type(model=self.configure_model(**model_params),
                                           connectivity=self.configure_connectivity(),
                                           integrator=self.configure_integrator(),
                                           monitors=self.configure_monitors(),
                                           simulation_length=self.simulation_length)

        if self.initial_conditions is not None:
            simulator.initial_conditions = self.configure_initial_conditions(simulator)

        simulator.log.setLevel(20)

        simulator.configure()

        return simulator


class CoSimulatorParallelBuilder(CoSimulatorBuilder):

    """CoSimulatorParallelBuilder is an opinionated builder for a TVB CoSimulatorParallel,
       adjusted for parallel cosimulation.
       Depending on its properties set, the builder may
       - scale/normalize the connectivity weights,
       - remove time delays or not,
       - remove the self-connections or brain region nodes (diagonal of connectivity matrix)
       - set integrator (including noise and integration step),
       - set monitor (including model's variables of interest and period)
    """

    _cosimulator_type = CoSimulatorParallel


class CoSimulatorRemoteParallelBuilder(CoSimulatorBuilder):

    """CoSimulatorParallelBuilder is an opinionated builder for a TVB CoSimulatorParallel,
       adjusted for parallel cosimulation.
       Depending on its properties set, the builder may
       - scale/normalize the connectivity weights,
       - remove time delays or not,
       - remove the self-connections or brain region nodes (diagonal of connectivity matrix)
       - set integrator (including noise and integration step),
       - set monitor (including model's variables of interest and period)
    """

    _cosimulator_type = CoSimulatorRemoteParallel


class CoSimulatorSerialBuilder(CoSimulatorBuilder):
    """CoSimulatorSerialBuilder is an opinionated builder for a TVB CoSimulatorSerial,
       adjusted for serial cosimulation.
       Depending on its properties set, the builder may
       - scale/normalize the connectivity weights,
       - remove time delays or not,
       - remove the self-connections or brain region nodes (diagonal of connectivity matrix)
       - set integrator (including noise and integration step),
       - set monitor (including model's variables of interest and period)
    """

    _cosimulator_type = CoSimulatorSerial


class CoSimulatorNetpyneBuilder(CoSimulatorSerialBuilder):

    """CoSimulatorNetpyneBuilder is an opinionated builder for a TVB CoSimulatorNetpyne,
       adjusted for parallel cosimulation.
    """

    _cosimulator_type = CoSimulatorNetpyne
