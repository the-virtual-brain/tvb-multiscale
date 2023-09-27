# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
This is the module responsible for co-simulation of TVB with spiking simulators.
It inherits the Simulator class.

.. moduleauthor:: Dionysios Perdikis <dionysios.perdikis@charite.de>


"""

import time
import warnings
from decimal import Decimal

import numpy

from tvb.basic.neotraits.api import Attr, NArray, Int
from tvb.simulator.models.base import Model
from tvb.simulator.simulator import Simulator, math
from tvb.simulator.monitors import TemporalAverage
from tvb.simulator.integrators import HeunDeterministic
from tvb.simulator.backend import ReferenceBackend

from tvb.contrib.cosimulation.cosimulator import CoSimulator as CoSimulatorBase

from tvb.contrib.cosimulation.cosim_monitors import RawCosim, CosimMonitorFromCoupling

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBOutputInterfaces
from tvb_multiscale.core.interfaces.tvb.interfaces import TVBInputInterfaces


class CoSimulator(CoSimulatorBase, HasTraits):

    model: Model = Attr(
        field_type=Model,
        label="Local dynamic model",
        default=WilsonCowan(),
        required=True,
        doc="""A tvb.simulator.Model object which describe the local dynamic
            equations, their parameters, and, to some extent, where connectivity
            (local and long-range) enters and which state-variables the Monitors
            monitor. By default the 'Generic2dOscillator' model is used. Read the
            Scientific documentation to learn more about this model.""")

    output_interfaces = Attr(
        field_type=TVBOutputInterfaces,
        label="TVB to cosimulation outlet interfaces",
        default=None,
        required=False,
        doc="""BaseInterfaces to couple from TVB to a 
               cosimulation outlet (i.e., translator level or another (co-)simulator""")

    input_interfaces = Attr(
        field_type=TVBInputInterfaces,
        label="Cosimulation to TVB interfaces",
        default=None,
        required=False,
        doc="""BaseInterfaces for updating from a cosimulation outlet 
               (i.e., translator level or another (co-)simulator to TVB.""")

    out_proxy_inds = NArray(
        dtype=int,
        label="Indices of TVB output proxy nodes",
        default=numpy.asarray([], dtype=int),
        required=True)

    min_idelay_sync_n_step_ratio = Int(
        label="min_idelay_synch_n_step_ratio",
        choices=(1, 2),
        default=1,
        required=True,
        doc="""min_idelay to synchronization_n_step ratio, 
               i.e., an integer value defining how many times smaller should the synchronization time be 
               compared to the minimum delay time in integration time steps.
               For the moment we limit it to 1 (synchronization_time = min_delay) 
               or 2 (synchronization_time = min_delay/2)""")

    relative_output_interfaces_time_steps = Int(
        label="relative_output_interfaces_time_steps",
        default=0,
        required=True,
        doc="""Relative time steps for cosimulation monitors to sample 
               in the future (for coupling monitors) or in the past (for non-coupling monitors), 
               for the output interfaces.""")

    PRINT_PROGRESSION_MESSAGE = True

    n_output_interfaces = 0
    n_input_interfaces = 0

    _number_of_dt_decimals = None

    def __init__(self, **kwargs):

        # Attributes of TVB-multiscale CoSimulator:
        self.model = WilsonCowan()
        self.output_interfaces = None
        self.input_interfaces = None
        self.out_proxy_inds = numpy.asarray([], dtype=int)
        self.min_idelay_sync_n_step_ratio = 1
        self.relative_output_interfaces_time_steps = 0
        self.PRINT_PROGRESSION_MESSAGE = True
        self.n_output_interfaces = 0
        self.n_input_interfaces = 0
        self._number_of_dt_decimals = None

        # Attributes of tvb_contrib.cosimulation CoSimulator:
        self.exclusive = False
        self.voi = numpy.asarray([], dtype=numpy.int_)
        self.proxy_inds = numpy.asarray([], dtype=numpy.int_)
        self.cosim_monitors = []
        self.synchronization_time = 0.0
        self.synchronization_n_step = 0
        self.good_cosim_update_values_shape = (0, 0, 0, 0)
        self.cosim_history = None  # type: CosimHistory
        self._cosimulation_flag = False
        self._compute_requirements = True
        self.number_of_cosim_monitors = 0
        self._cosim_monitors_noncoupling_indices = []
        self._cosim_monitors_coupling_indices = []
        self._existing_connections = []
        self._min_delay = 0.0
        self._min_idelay = 1
        self.relative_output_time_steps = 0

        # Attributes of tvb_library.tvb.simulator Simulator:
        # Required attributes cannot be set to None!
        # self.connectivity = None
        # self.coupling = None
        self.integrator = HeunDeterministic()
        self.monitors = (TemporalAverage(),)
        self.surface = None
        self.stimulus = None
        self.conduction_speed = 3.0
        self.initial_conditions = None
        self.simulation_length = 1000.0
        self.backend = ReferenceBackend()
        self.history = None  # type: SparseHistory

        super(CoSimulator, self).__init__(**kwargs)

    @property
    def in_proxy_inds(self):
        return self.proxy_inds

    @property
    def all_proxy_inds(self):
        return numpy.unique(self.proxy_inds.tolist() + self.out_proxy_inds.tolist())

    def compute_default_synchronization_time_and_n_step(self):
        default_synchronization_n_step = \
                int(numpy.floor(self._min_idelay / self.min_idelay_sync_n_step_ratio))
        default_synchronization_time = numpy.around(default_synchronization_n_step * self.integrator.dt,
                                                    decimals=self._number_of_dt_decimals)
        return default_synchronization_time, default_synchronization_n_step

    @property
    def default_synchronization_time(self):
        return self.compute_default_synchronization_time_and_n_step()[0]

    @property
    def default_synchronization_n_step(self):
        return self.compute_default_synchronization_time_and_n_step()[1]

    def _configure_synchronization_time(self):
        """This method will set the synchronization time and number of steps,
           longer or equal to the integration time step.
           Moreover, the synchronization time must be equal or shorter
           than the minimum delay of all existing connections.
           Existing connections are considered those with nonzero weights.
        """
        # The synchronization time should be at least equal to integrator.dt:
        self.synchronization_time = numpy.around(numpy.maximum(self.synchronization_time, self.integrator.dt),
                                                 decimals=self._number_of_dt_decimals)
        if self.synchronization_time < self.integrator.dt:
            self.log.warning("synchronization time =  %g is smaller than the time step integrator.dt = %g\n"
                             "Setting synchronization time equal to integrator.dt!"
                             % (self.synchronization_time, self.integrator.dt))
            self.synchronization_time = self.integrator.dt
        # Compute the number of synchronization time steps:
        self.synchronization_n_step = int(numpy.round(self.synchronization_time / self.integrator.dt).item())
        synchronization_time = numpy.around(self.integrator.dt * self.synchronization_n_step,
                                            decimals=self._number_of_dt_decimals)
        if synchronization_time != self.synchronization_time:
            self.log.warning("Slightly adjusting synchronization time from %g to %g\n"
                             "to be an integer multiple of integrator.dt = %g!" %
                             (self.synchronization_time, synchronization_time, self.integrator.dt))
        self.synchronization_time = synchronization_time
        # Check if the synchronization time is smaller than the minimum delay of the connectivity:
        if self.synchronization_time > self._min_delay:
            raise ValueError('The synchronization time %g is longer than '
                             'the minimum delay time %g '
                             'of all existing connections (i.e., of nonzero weight)!'
                             % (self.synchronization_time, self._min_delay))
        if self.n_output_interfaces:
            self.output_interfaces.synchronization_time = self.synchronization_time
            self.output_interfaces.synchronization_n_step = self.synchronization_n_step
        if self.n_input_interfaces:
            self.input_interfaces.synchronization_time = self.synchronization_time
            self.input_interfaces.synchronization_n_step = self.synchronization_n_step

    def _preconfigure_synchronization_time(self):
        """This method will default synchronization time
           to be equal to the minimum delay time of connectivity,
           in case the user hasn't set it up until this point."""
        self.compute_default_synchronization_time_and_n_step()
        default_synchronization_time = self.default_synchronization_time
        if self.synchronization_time == 0.0:
            self.synchronization_time = default_synchronization_time
        try:
            self._configure_synchronization_time()
        except Exception as e:
            if self.synchronization_time > default_synchronization_time:
                self.log.warning(e)
                self.log.warning('Resetting it equal to minimum delay time!')
                self.synchronization_time = default_synchronization_time
            else:
                raise e

    def _preconfigure_interfaces_vois_proxy_inds(self):
        """This method will
            - set the voi and proxy_inds of the CoSimulator, based on the predefined input and output interfaces,
            - configure all interfaces.
        """
        voi = []
        proxy_inds = []
        out_proxy_inds = []
        if self.output_interfaces:
            # Configure all TVB to Cosim interfaces:
            self.output_interfaces.configure()
            if self.n_output_interfaces:
                voi += self.output_interfaces.voi_unique.tolist()
                out_proxy_inds += self.output_interfaces.proxy_inds_unique.tolist()
        if self.input_interfaces:
            # Configure all Cosim to TVB interfaces:
            self.input_interfaces.configure()
            if self.n_input_interfaces:
                voi += self.input_interfaces.voi_unique.tolist()
                proxy_inds = self.input_interfaces.proxy_inds_unique.tolist()
        self.voi = numpy.unique(voi).astype(numpy.int32)
        self.proxy_inds = numpy.unique(proxy_inds).astype(numpy.int32)
        self.out_proxy_inds = numpy.unique(out_proxy_inds).astype(numpy.int32)

    def _assert_cosim_monitors_vois_period(self):
        """This method will assert that
            - the period of all CosimMonitor instances set is equal to the integrator's dt.
            - there is at least one CosimMonitor instance set for any voi the output interfaces need,
         """
        periods = [cosim_monitor.period for cosim_monitor in self.cosim_monitors]
        assert numpy.allclose(periods, self.integrator.dt, 1e-6)
        if self.n_output_interfaces:
            n_cosim_monitors = len(self.cosim_monitors)
            assert n_cosim_monitors > 0
            for interface in self.output_interfaces.interfaces:
                assert n_cosim_monitors > interface.monitor_ind
                monitor = self.cosim_monitors[interface.monitor_ind]
                if interface.coupling_mode.upper() == "TVB":
                    assert isinstance(monitor, CosimMonitorFromCoupling)
                    cvar = self.model.cvar.tolist()
                    assert [cvar.index(voi) in monitor.voi for voi in interface.voi]
                else:
                    assert not isinstance(monitor, CosimMonitorFromCoupling)
                    assert [voi in monitor.voi for voi in interface.voi]

    def _configure_local_vois_and_proxy_inds_per_interface(self):
        """This method will set the local -per cosimulation and interface- voi and proxy_inds indices,
           based on the voi of each linked cosimulation monitor, for TVB to Cosimulator interfaces,
           and on the expected shape of ths cosimulation updates data for Cosimulator to TVB interfaces.
        """
        if self.n_output_interfaces:
            # Set the correct voi indices with reference to the linked TVB CosimMonitor, for each cosimulation:
            self.output_interfaces.set_local_indices(self.cosim_monitors)
        if self.n_input_interfaces:
            # Method to get the correct indices of voi and proxy_inds, for each cosimulation,
            # adjusted to the contents, shape etc of the cosim_updates,
            # based on TVB CoSimulators' vois and proxy_inds, i.e., good_cosim_update_values_shape
            self.input_interfaces.set_local_indices(self.voi, self.proxy_inds)

    def configure(self, full_configure=True):
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
        Simulator.configure(self, full_configure=full_configure)
        self.compute_min_delay()
        self._number_of_dt_decimals = numpy.abs(Decimal('%g' % self.integrator.dt).as_tuple().exponent)
        self._compute_requirements = True
        self.n_tvb_steps_ran_since_last_synch = 0
        if self.output_interfaces:
            self.output_interfaces.dt = self.integrator.dt
            self.n_output_interfaces = self.output_interfaces.number_of_interfaces
        self.n_input_interfaces = self.input_interfaces.number_of_interfaces if self.input_interfaces else 0
        self._preconfigure_interfaces_vois_proxy_inds()
        self._preconfigure_synchronization_time()
        all_proxy_inds = self.all_proxy_inds
        if self.voi.shape[0] * all_proxy_inds.shape[0] != 0:
            self._cosimulation_flag = True
            self._configure_cosimulation()
            self._assert_cosim_monitors_vois_period()
            self._configure_local_vois_and_proxy_inds_per_interface()
        elif self.voi.shape[0] + all_proxy_inds.shape[0] > 0:
            raise ValueError("One of CoSimulator.voi (size=%i) or simulator.all_proxy_inds (size=%i) are empty!"
                             % (self.voi.shape[0], all_proxy_inds.shape[0]))
        else:
            self._cosimulation_flag = False
            self.synchronization_n_step = 0
            self.synchronization_time = 0.0
        if self.current_step:
            msg = "Current step is not 0 upon configuration!\n" + \
                  "Setting it to 0. Initial condition might be affected!"
            self.log.warning(msg)
            print(msg)
            self.current_step = 0

    def _prepare_cosimulation_call(self, simulation_length=None, n_steps=None, cosim_updates=None):
        # Check if the cosimulation update inputs (if any) are correct and update cosimulation history:

        if simulation_length is not None:
            self.simulation_length = float(simulation_length)

        if self._cosimulation_flag:
            if n_steps is not None:
                raise ValueError("n_steps is not used in cosimulation!")
            if cosim_updates is None:
                n_steps = self.synchronization_n_step
            elif len(cosim_updates) != 2:
                raise ValueError("Incorrect cosimulation updates input length %i, expected 2 (i.e., time steps, values)"
                                 % len(cosim_updates))
            elif len(cosim_updates[1].shape) != 4 \
                     or self.good_cosim_update_values_shape[0] < cosim_updates[1].shape[0] \
                     or numpy.any(self.good_cosim_update_values_shape[1:] != cosim_updates[1].shape[1:]):
                raise ValueError("Incorrect cosimulation updates values shape %s, \nexpected %s "
                                 "(i.e., (<=synchronization_n_step, n_voi, n_proxy_nodes, number_of_modes))" %
                                 (str(cosim_updates[1].shape), str(self.good_cosim_update_values_shape)))
            else:
                n_steps = cosim_updates[0].shape[0]
                # Now update cosimulation history with the cosimulation inputs:
                # TODO: Resolve difference in time update with master
                self._update_cosim_history(cosim_updates[0], cosim_updates[1])

            self.simulation_length = n_steps * self.integrator.dt

        else:
            # Normal TVB simulation - no cosimulation:
            if cosim_updates is not None:
                raise ValueError("cosim_update is not used in normal simulation")

            if n_steps is None:
                n_steps = int(math.ceil(self.simulation_length / self.integrator.dt))
            else:
                if not numpy.issubdtype(type(n_steps), numpy.integer):
                    raise TypeError("Incorrect type for n_steps: %s, expected integer" % type(n_steps))
                self.simulation_length = n_steps * self.integrator.dt
        return n_steps

    def _prepare_stimulus(self):
        if self.simulation_length != self.synchronization_time:
            simulation_length = float(self.simulation_length)
            self.simulation_length = float(self.synchronization_time)
            stimulus = super(CoSimulator, self)._prepare_stimulus()
            self.simulation_length = simulation_length
        else:
            stimulus = super(CoSimulator, self)._prepare_stimulus()
        return stimulus

    def __call__(self, simulation_length=None, random_state=None, n_steps=None,
                 cosim_updates=None, recompute_requirements=False, skip_prepare_cosim=False):
        """
        Return an iterator which steps through simulation time, generating monitor outputs.

        See the run method for a convenient way to collect all output in one call.

        :param simulation_length: Length of the simulation to perform in ms.
        :param random_state:  State of NumPy RNG to use for stochastic integration.
        :param n_steps: Length of the simulation to perform in integration steps. Overrides simulation_length.
        :param cosim_updates: data from the other co-simulator to update TVB state and history
        :param recompute_requirements: check if the requirement of the simulation
        :params check_inputs: check if the cosimulation update inputs (if any) are correct and update cosimulation history
        :return: Iterator over monitor outputs.
        """

        self.calls += 1

        if not skip_prepare_cosim:
            n_steps = self._prepare_cosimulation_call(simulation_length, n_steps, cosim_updates)

        # Initialization
        if self._compute_requirements or recompute_requirements:
            # Compute requirements for CoSimulation.simulation_length, not for synchronization time
            self._guesstimate_runtime()
            self._calculate_storage_requirement()
            self._compute_requirements = False
        self.integrator.set_random_state(random_state)

        local_coupling = self._prepare_local_coupling()
        stimulus = self._prepare_stimulus()
        state = self.current_state
        start_step = self.current_step + 1
        node_coupling = self._loop_compute_node_coupling(start_step)

        # integration loop
        for step in range(start_step, start_step + n_steps):
            self._loop_update_stimulus(step, stimulus)
            state = self.integrate_next_step(state, self.model, node_coupling, local_coupling,
                                             numpy.where(stimulus is None, 0.0, stimulus))
            state_output = self._loop_update_cosim_history(step, state)
            node_coupling = self._loop_compute_node_coupling(step + 1)
            output = self._loop_monitor_output(step-self.synchronization_n_step, state_output, node_coupling)
            if output is not None:
                yield output

        self.current_state = state
        self.current_step = self.current_step + n_steps

    def get_cosim_updates(self, cosimulation=True):
        cosim_updates = None
        if cosimulation and self.input_interfaces:
            # Get the update data from the other cosimulator
            cosim_updates = self.input_interfaces(self.good_cosim_update_values_shape)
            isnans = numpy.isnan(cosim_updates[-1])
            if numpy.all(isnans):
                cosim_updates = None
                self.log.warning("No or all NaN valued cosimulator updates at time step %d!" % self.current_step)
            elif numpy.any(isnans):
                msg = "NaN values detected in cosimulator updates at time step %d!" % self.current_step
                self.log.error(msg)
                raise Exception(msg)
        return cosim_updates

    def send_cosim_coupling(self, cosimulation=True):
        outputs = []
        if cosimulation and self.output_interfaces and self.n_tvb_steps_ran_since_last_synch > 0:
            if self.output_interfaces.number_of_interfaces:
                # Send the data to the other cosimulator
                outputs = \
                    self.output_interfaces(self.loop_cosim_monitor_output(self.n_tvb_steps_ran_since_last_synch,
                                                                          self.relative_output_interfaces_time_steps))
        return outputs

    def run_for_synchronization_time(self, ts, xs, wall_time_start, cosim_updates=None, cosimulation=True, **kwds):
        # Loop of integration for synchronization_time
        current_step = int(self.current_step)
        for data in self(cosim_updates=cosim_updates, **kwds):
            for tl, xl, t_x in zip(ts, xs, data):
                if t_x is not None:
                    t, x = t_x
                    tl.append(t)
                    xl.append(x)
        return self.current_step - current_step

    def _log_print_progress_message(self, simulated_steps, simulation_length):
        log_msg = "...%.3f%% completed in %g sec!" % \
                  ((100 * (simulated_steps * self.integrator.dt) / simulation_length), time.time() - self._tic)
        self.log.info(log_msg)
        if self.PRINT_PROGRESSION_MESSAGE:
            print("\r" + log_msg, end="")
    
    def info(self, recursive=0):
        info = HasTraits.info(self, recursive=recursive)
        return info

    def info_details(self, recursive=0, **kwargs):
        info = HasTraits.info_details(self, recursive=recursive, **kwargs)
        return info

    def summary_info(self, recursive=0):
        return HasTraits.summary_info(self, recursive=recursive)

    def summary_info_details(self, recursive=0, **kwargs):
        return HasTraits.summary_info_details(self, recursive=recursive, **kwargs)
