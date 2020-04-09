# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
This is the main module of the simulator. It defines the Simulator class which
brings together all the structural and dynamic components necessary to define a
simulation and the method for running the simulation.

.. moduleauthor:: Stuart A. Knock
.. moduleauthor:: Marmaduke Woodman
.. moduleauthor:: Paula Sanz Leon

"""
import sys
import time
import math
import numpy
from tvb.basic.neotraits.api import Attr, List
from tvb.datatypes import connectivity
from tvb.simulator import models
from tvb.simulator import monitors
from tvb.simulator import integrators
from tvb.simulator.common import numpy_add_at
from tvb.simulator.history import SparseHistory
from tvb.simulator.simulator import Simulator as SimulatorTVB
from tvb_multiscale.config import CONFIGURED
from tvb_multiscale.simulator_tvb_deprecated.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.plot.utils.log_error_utils import initialize_logger, raise_value_error


LOG = initialize_logger(__name__)


class Simulator(SimulatorTVB):

    tvb_spikeNet_interface = None
    configure_spiking_simulator = None
    run_spiking_simulator = None

    model = Attr(
        field_type=models.Model,
        label="Local dynamic model",
        default=ReducedWongWangExcIOInhI(),
        required=True,
        doc="""A tvb.simulator.Model object which describes the local dynamic
            equations, their parameters, and, to some extent, where connectivity
            (local and long-range) enters and which state-variables the Monitors
            monitor. By default the 'ReducedWongWang' model is used. Read the 
            Scientific documentation to learn more about this model.""")

    monitors = List(
        of=monitors.Monitor,
        label="Monitor(s)",
        default=(monitors.Raw(),),
        doc="""A tvb.simulator.Monitor or a list of tvb.simulator.Monitor
            objects that 'know' how to record relevant data from the simulation. Two
            main types exist: 1) simple, spatial and temporal, reductions (subsets
            or averages); 2) physiological measurements, such as EEG, MEG and fMRI.
            By default the Model's specified variables_of_interest are returned,
            temporally downsampled from the raw integration rate to a sample rate of
            1024Hz.""")

    integrator = Attr(
        field_type=integrators.Integrator,
        label="Integration scheme",
        default=integrators.HeunStochastic(
            dt=0.001),
        required=True,
        doc="""A tvb.simulator.Integrator object which is
                an integration scheme with supporting attributes such as 
                integration step size and noise specification for stochastic 
                methods. It is used to compute the time courses of the model state 
                variables.""")

    connectivity = Attr(
        field_type=connectivity.Connectivity,
        label="Long-range connectivity",
        default=CONFIGURED.DEFAULT_SUBJECT["connectivity"],
        required=True,
        doc="""A tvb.datatypes.Connectivity object which contains the
        structural long-range connectivity data (i.e., white-matter tracts). In
        combination with the ``Long-range coupling function`` it defines the inter-regional
        connections. These couplings undergo a time delay via signal propagation
        with a propagation speed of ``Conduction Speed``""")

    @property
    def config(self):
        try:
            return self.tvb_spikeNet_interface.config
        except:
            return CONFIGURED

    def _configure_integrator_noise(self):
        """
        This enables having noise to be state variable specific and/or to enter
        only via specific brain structures, for example it we only want to
        consider noise as an external input entering the brain via appropriate
        thalamic nuclei.

        Support 3 possible shapes:
            1) number_of_nodes;

            2) number_of_state_variables; and

            3) (number_of_state_variables, number_of_nodes).

        """

        if self.integrator.noise.ntau > 0.0:
            LOG.warning("Colored noise is currently not supported for tvb-spikeNet simulations!\n" +
                        "Setting integrator.noise.ntau = 0.0 and configuring white noise!")
            self.integrator.noise.ntau = 0.0

        super(Simulator, self)._configure_integrator_noise()

    def bound_and_clamp(self, state):
        # If there is a state boundary...
        if self.integrator.state_variable_boundaries is not None:
            # ...use the integrator's bound_state
            self.integrator.bound_state(state)
        # If there is a state clamping...
        if self.integrator.clamped_state_variable_values is not None:
            # ...use the integrator's clamp_state
            self.integrator.clamp_state(state)

    def _update_and_bound_history(self, history):
        self.bound_and_clamp(history)
        # If there are non-state variables, they need to be updated for history:
        try:
            # Assuming that node_coupling can have a maximum number of dimensions equal to the state variables,
            # in the extreme case where all state variables are cvars as well, we set:
            node_coupling = numpy.zeros((history.shape[0], 1, history.shape[2], 1))
            for i_time in range(history.shape[1]):
                self.model.update_non_state_variables(history[:, i_time], node_coupling[:, 0], 0.0)
            self.bound_and_clamp(history)
        except:
            pass

    def _configure_history(self, initial_conditions):
        """
        Set initial conditions for the simulation using either the provided
        initial_conditions or, if none are provided, the model's initial()
        method. This method is called durin the Simulator's __init__().

        Any initial_conditions that are provided as an argument are expected
        to have dimensions 1, 2, and 3 with shapse corresponding to the number
        of state_variables, nodes and modes, respectively. If the provided
        inital_conditions are shorter in time (dim=0) than the required history
        the model's initial() method is called to make up the difference.

        """
        rng = numpy.random
        if hasattr(self.integrator, 'noise'):
            rng = self.integrator.noise.random_stream
        # Default initial conditions
        if initial_conditions is None:
            n_time, n_svar, n_node, n_mode = self.good_history_shape
            LOG.info('Preparing initial history of shape %r using model.initial()', self.good_history_shape)
            if self.surface is not None:
                n_node = self.number_of_nodes
            history = self.model.initial(self.integrator.dt, (n_time, n_svar, n_node, n_mode), rng)
        # ICs provided
        else:
            # history should be [timepoints, state_variables, nodes, modes]
            LOG.info('Using provided initial history of shape %r', initial_conditions.shape)
            n_time, n_svar, n_node, n_mode = ic_shape = initial_conditions.shape
            nr = self.connectivity.number_of_regions
            if self.surface is not None and n_node == nr:
                initial_conditions = initial_conditions[:, :, self._regmap]
                return self._configure_history(initial_conditions)
            elif ic_shape[1:] != self.good_history_shape[1:]:
                raise_value_error("Incorrect history sample shape %s, expected %s"
                                  % ic_shape[1:], self.good_history_shape[1:])
            else:
                if ic_shape[0] >= self.horizon:
                    LOG.debug("Using last %d time-steps for history.", self.horizon)
                    history = initial_conditions[-self.horizon:, :, :, :].copy()
                else:
                    LOG.debug('Padding initial conditions with model.initial')
                    history = self.model.initial(self.integrator.dt, self.good_history_shape, rng)
                    shift = self.current_step % self.horizon
                    history = numpy.roll(history, -shift, axis=0)
                    history[:ic_shape[0], :, :, :] = initial_conditions
                    history = numpy.roll(history, shift, axis=0)
                self.current_step += ic_shape[0] - 1
        # Make sure that history values are bounded,
        # and any possible non-state variables are initialized
        # based on state variable ones (but with no coupling yet...)
        self._update_and_bound_history(numpy.swapaxes(history, 0, 1))
        LOG.info('Final initial history shape is %r', history.shape)
        # create initial state from history
        self.current_state = history[self.current_step % self.horizon].copy()
        LOG.debug('initial state has shape %r' % (self.current_state.shape, ))
        if self.surface is not None and history.shape[2] > self.connectivity.number_of_regions:
            n_reg = self.connectivity.number_of_regions
            (nt, ns, _, nm), ax = history.shape, (2, 0, 1, 3)
            region_history = numpy.zeros((nt, ns, n_reg, nm))
            numpy_add_at(region_history.transpose(ax), self._regmap, history.transpose(ax))
            region_history /= numpy.bincount(self._regmap).reshape((-1, 1))
            history = region_history
        # create history query implementation
        self.history = SparseHistory(
            self.connectivity.weights,
            self.connectivity.idelays,
            self.model.cvar,
            self.model.number_of_modes
        )
        # initialize its buffer
        self.history.initialize(history)

    def configure(self, tvb_spikeNet_interface, full_configure=True):
        """Configure simulator and its components.

        The first step of configuration is to run the configure methods of all
        the Simulator's components, ie its traited attributes.

        Configuration of a Simulator primarily consists of calculating the
        attributes, etc, which depend on the combinations of the Simulator's
        traited attributes (keyword args).

        Converts delays from physical time units into integration steps
        and updates attributes that depend on combinations of the 6 inputs.

        Returns
        -------
        sim: Simulator
            The configured Simulator instance.

        """
        if full_configure:
            # When run from GUI, preconfigure is run separately, and we want to avoid running that part twice
            self.preconfigure()
            # Make sure spatialised model parameters have the right shape (number_of_nodes, 1)

        self.tvb_spikeNet_interface = tvb_spikeNet_interface

        # TODO: decide if this is really necessary...
        if self.integrator.dt >= 2 * self.tvb_spikeNet_interface.spikeNet_min_delay:
            self.integrator.dt = int(numpy.round(self.integrator.dt /
                                                 self.tvb_spikeNet_interface.spikeNet_min_delay)) * \
                                 self.tvb_spikeNet_interface.spikeNet_min_delay
        else:
            raise_value_error("TVB integration time step dt=%f "
                              "is not equal or greater than twice the Spiking Network minimum delay min_delay=%f!" %
                              (self.integrator.dt, self.tvb_spikeNet_interface.spikeNet_min_delay))

        excluded_params = ("state_variable_range", "state_variable_boundaries", "variables_of_interest",
                           "noise", "psi_table", "nerf_table", "gid")
        spatial_reshape = self.model.spatial_param_reshape
        for param in type(self.model).declarative_attrs:
            if param in excluded_params:
                continue
            # If it's a surface sim and model parameters were provided at the region level
            region_parameters = getattr(self.model, param)
            if self.surface is not None:
                if region_parameters.size == self.connectivity.number_of_regions:
                    new_parameters = region_parameters[self.surface.region_mapping].reshape(spatial_reshape)
                    setattr(self.model, param, new_parameters)
            region_parameters = getattr(self.model, param)
            if hasattr(region_parameters, "size") and region_parameters.size == self.number_of_nodes:
                new_parameters = region_parameters.reshape(spatial_reshape)
                setattr(self.model, param, new_parameters)
        # Configure spatial component of any stimuli
        self._configure_stimuli()
        # Set delays, provided in physical units, in integration steps.
        self.connectivity.set_idelays(self.integrator.dt)
        self.horizon = self.connectivity.idelays.max() + 1
        # Reshape integrator.noise.nsig, if necessary.
        if isinstance(self.integrator, integrators.IntegratorStochastic):
            self._configure_integrator_noise()

        # Configure Monitors to work with selected Model, etc...
        self._configure_monitors()

        # TODO: find out why the model instance is different in simulator and interface...
        self.tvb_spikeNet_interface.configure(self.model)

        dummy = -numpy.ones((self.connectivity.number_of_regions, ))
        dummy[self.tvb_spikeNet_interface.spiking_nodes_ids] = 0.0
        # Create TVB model parameter for SpikeNet to target
        for param in self.tvb_spikeNet_interface.spikeNet_to_tvb_params:
            setattr(self.model, param, dummy)

        # Setup Spiking Simulator configure() and Run() method
        self.configure_spiking_simulator = self.tvb_spikeNet_interface.spiking_network.configure
        self.run_spiking_simulator = self.tvb_spikeNet_interface.spiking_network.Run

        # If there are Spiking nodes and are represented exclusively in Spiking Network...
        if self.tvb_spikeNet_interface.exclusive_nodes and len(self.tvb_spikeNet_interface.spiking_nodes_ids) > 0:
            # ...zero coupling interface_weights among Spiking nodes:
            self.connectivity.weights[self.tvb_spikeNet_interface.spiking_nodes_ids] \
                [:, self.tvb_spikeNet_interface.spiking_nodes_ids] = 0.0

        # Setup history
        # TODO: Reflect upon the idea to allow SpikeNet initialization and history setting via TVB
        self._configure_history(self.initial_conditions)

        # TODO: Shall we implement a parallel implentation for multiple modes for SpikeNet as well?!
        if self.current_state.shape[2] > 1:
            raise_value_error("Multiple modes' simulation not supported for TVB multiscale simulations!\n"
                              "Current modes number is %d." % self.initial_conditions.shape[3])

        # Estimate of memory usage.
        self._census_memory_requirement()

        return self

    # TODO: update all those functions below to compute the fine scale requirements as well, ...if you can! :)

    # used by simulator adaptor
    def memory_requirement(self):
        # TODO: calculate SpikeNet memory requirement as well!
        """
        Return an estimated of the memory requirements (Bytes) for this
        simulator's current configuration.
        """
        return super(Simulator, self).memory_requirement()

    # appears to be unused
    def runtime(self, simulation_length):
        # TODO: calculate SpikeNet runtime as well!
        """
        Return an estimated run time (seconds) for the simulator's current
        configuration and a specified simulation length.

        """
        return super(Simulator, self).runtime(simulation_length)

    # used by simulator adaptor
    def storage_requirement(self, simulation_length):
        # TODO: calculate SpikeNet storage requirement as well!
        """
        Return an estimated storage requirement (Bytes) for the simulator's
        current configuration and a specified simulation length.

        """
        return super(Simulator, self).storage_requirement(simulation_length)

    def update_state(self, state, node_coupling, local_coupling=0.0):
        # If there are non-state variables, they need to be updated for the initial condition:
        try:
            self.model.update_non_state_variables(state, node_coupling, local_coupling)
            self.bound_and_clamp(state)
        except:
            # If not, the kwarg will fail and nothing will happen
            pass

    def __call__(self, simulation_length=None, random_state=None):
        """
        Return an iterator which steps through simulation time, generating monitor outputs.

        See the run method for a convenient way to collect all output in one call.

        :param simulation_length: Length of the simulation to perform in ms.
        :param random_state:  State of NumPy RNG to use for stochastic integration.
        :return: Iterator over monitor outputs.
        """

        self.calls += 1
        if simulation_length is not None:
            self.simulation_length = simulation_length

        # Intialization
        self._guesstimate_runtime()
        self._calculate_storage_requirement()
        self._handle_random_state(random_state)
        n_reg = self.connectivity.number_of_regions
        local_coupling = self._prepare_local_coupling()
        stimulus = self._prepare_stimulus()
        state = self.current_state
        # Do for initial condition:
        step = self.current_step + 1  # the first step in the loop
        node_coupling = self._loop_compute_node_coupling(step)
        self._loop_update_stimulus(step, stimulus)
        # This is not necessary in most cases
        # if update_non_state_variables=True in the model dfun by default
        self.update_state(state, node_coupling, local_coupling)

        # spikeNet simulation preparation:
        self.configure_spiking_simulator()

        # A flag to skip unnecessary steps when Spiking Simulator does NOT update TVB state
        updateTVBstateFromSpikeNet = len(self.tvb_spikeNet_interface.spikeNet_to_tvb_sv_interfaces_ids) > 0

        # integration loop
        n_steps = int(math.ceil(self.simulation_length / self.integrator.dt))
        tic = time.time()
        tic_ratio = 0.1
        tic_point = tic_ratio * n_steps
        for step in range(self.current_step + 1,  self.current_step + n_steps + 1):
            # TVB state -> SpikeNet (state or parameter)
            # Communicate TVB state to some SpikeNet device (TVB proxy) or TVB coupling to SpikeNet nodes,
            # including any necessary conversions from TVB state to SpikeNet variables,
            # in a model specific manner
            # TODO: find what is the general treatment of local coupling, if any!
            #  Is this addition correct in all cases for all builders?
            self.tvb_spikeNet_interface.tvb_state_to_spikeNet(state, node_coupling+local_coupling, stimulus, self.model)
            # SpikeNet state -> TVB model parameter
            # Couple the SpikeNet state to some TVB model parameter,
            # including any necessary conversions in a model specific manner
            self.model = self.tvb_spikeNet_interface.spikeNet_state_to_tvb_parameter(self.model)
            # Integrate TVB to get the new TVB state
            state = self.integrator.scheme(state, self.model.dfun, node_coupling, local_coupling, stimulus)
            if numpy.any(numpy.isnan(state)) or numpy.any(numpy.isinf(state)):
                raise ValueError("NaN or Inf values detected in simulator state!:\n%s" % str(state))
            # Integrate Spiking Network to get the new Spiking Network state
            self.run_spiking_simulator(self.integrator.dt)
            if updateTVBstateFromSpikeNet:
                # SpikeNet state -> TVB state
                # Update the new TVB state variable with the new SpikeNet state,
                # including any necessary conversions from SpikeNet variables to TVB state,
                # in a model specific manner
                state = self.tvb_spikeNet_interface.spikeNet_state_to_tvb_state(state)
                self.bound_and_clamp(state)
            # Prepare coupling and stimulus for next time step
            # and, therefore, for the new TVB state:
            node_coupling = self._loop_compute_node_coupling(step)
            self._loop_update_stimulus(step, stimulus)
            # Update any non-state variables and apply any boundaries again to the new state:
            self.update_state(state, node_coupling, local_coupling)
            # Now direct the new state to history buffer and monitors
            self._loop_update_history(step, n_reg, state)
            output = self._loop_monitor_output(step, state)
            if output is not None:
                yield output
            if step-self.current_step >= tic_point:
                toc = time.time() - tic
                if toc > 600:
                    if toc > 7200:
                        time_string = "%0.1f hours" % (toc / 3600)
                    else:
                        time_string = "%0.1f min" % (toc / 60)
                else:
                    time_string = "%0.1f sec" % toc
                print_this = "\r...%0.1f%% done in %s" % \
                            (100.0 * (step - self.current_step) / n_steps, time_string)
                sys.stdout.write(print_this)
                sys.stdout.flush()
                tic_point += tic_ratio * n_steps

        self.current_state = state
        self.current_step = self.current_step + n_steps - 1  # -1 : don't repeat last point
