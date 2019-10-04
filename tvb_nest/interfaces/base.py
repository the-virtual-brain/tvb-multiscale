# -*- coding: utf-8 -*-

from abc import ABCMeta
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import is_integer
from six import add_metaclass

LOG = initialize_logger(__name__)

@add_metaclass(ABCMeta)
class TVBNESTInterface(object):

    nest_instance = None
    tvb_model = None
    dt = 0.1

    tvb_nodes_ids = []
    nest_nodes_ids = []

    nest_network = None

    tvb_to_nest_interfaces = None
    nest_to_tvb_interfaces = None
    nest_to_tvb_params_interfaces_ids = None
    nest_to_tvb_sv_interfaces_ids = None
    nest_to_tvb_params = None

    w_nest_spikes_to_tvb_rate = 1.0  # (assuming spikes/ms in TVB)
    w_nest_spikes_to_tvb_sv = 1.0
    w_tvb_sv_to_nest_spike_rate = 1000.0  # (spike rate in NEST is in spikes/sec, whereas dt is in ms)
    w_tvb_sv_to_nest_current = 1000.0  # (1000.0 (nA -> pA), because I_e, and dc_generator amplitude in NEST are in pA)
    w_tvb_sv_to_nest_potential = 1.0  # assuming mV in both NEST and TVB

    def __init__(self, config=CONFIGURED):
        self.config = config
        LOG.info("%s created!" % self.__class__)

    @property
    def nest_nodes(self):
        return self.nest_network.region_nodes

    @property
    def number_of_nest_nodes(self):
        return len(self.nest_nodes_ids)

    @property
    def number_of_tvb_nodes(self):
        return len(self.tvb_nodes_ids)

    @property
    def number_of_nodes(self):
        return self.number_of_nest_nodes + self.number_of_tvb_nodes

    @property
    def number_of_tvb_state_variables(self):
        return len(self.tvb_model.state_variables)

    # TODO: find a possible way to differentiate scalings between
    #  receiver (as in _tvb_state_to_nest_current),
    #  and sender (as in all other cases below), node indexing

    def _configure_parameters_shapes(self):
        dummy = np.ones((self.number_of_nodes, 1))
        # Confirm good shape for TVB-NEST interface model parameters
        # TODO: find a possible way to differentiate scalings between
        #  receiver (as in _tvb_state_to_nest_current),
        #  and sender (as in all other cases below), node indexing
        #  Also, the size doesn't have to be in all cases equal to number_of_nodes,
        #  but sometimes equal to number_of_nest_nodes or to number_of_tvb_nodes
        for prop in ["w_tvb_sv_to_nest_current",
                     "w_tvb_sv_to_nest_potential",
                     "w_tvb_sv_to_nest_spike_rate",
                     "w_nest_spikes_to_tvb_rate",
                     "w_nest_spikes_to_tvb_sv"]:
            setattr(self, prop, (dummy * np.array(getattr(self, prop))).squeeze())

    def configure(self, tvb_model):
        """  """
        self.w_nest_spikes_to_tvb_sv /= self.dt  # Used to convert number of spikes to a spike rate
        self.w_nest_spikes_to_tvb_rate /= self.dt  # Used to convert number of spikes to a spike rate
        self.nest_to_tvb_params = []
        self.nest_to_tvb_params_interfaces_ids = []
        self.nest_to_tvb_sv_interfaces_ids = []
        for interface_id, interface in enumerate(self.nest_to_tvb_interfaces.values()):
            if is_integer(interface.tvb_sv_id) and interface.tvb_sv_id >= 0:
                self.nest_to_tvb_sv_interfaces_ids.append(interface_id)
            else:
                self.nest_to_tvb_params_interfaces_ids.append(interface_id)
                self.nest_to_tvb_params.append(interface.name)
        self.tvb_model = tvb_model
        self._configure_parameters_shapes()

    def tvb_state_to_nest(self, state, coupling, stimulus):
        for interface in self.tvb_to_nest_interfaces.values():
            if interface.model == "current":
                # We assume that current is a mean field quantity
                # applied equally and in parallel
                # to all target neurons of the NEST spiking populations
                # Instantaneous transmission. TVB history is used to buffer delayed communication.
                interface.set_values(self.w_tvb_sv_to_nest_current[self.nest_nodes_ids] *
                                     coupling[interface.tvb_coupling_id, self.nest_nodes_ids].squeeze())
            elif interface.model == "potential":
                # We assume that potential is a mean field quantity
                # applied equally and in parallel
                # to all target neurons of the NEST spiking populations
                # Instantaneous transmission. TVB history is used to buffer delayed communication.
                interface.set_values(self.w_tvb_sv_to_nest_potential[self.nest_nodes_ids] *
                                     coupling[interface.tvb_coupling_id, self.nest_nodes_ids].squeeze())
            elif interface.model == "dc_generator":
                # We assume that current is a mean field quantity
                # applied equally and in parallel
                # to all target neurons of the NEST spiking populations
                # This is why no scaling has been applied
                # for the synaptic weight from the dc_generator device, representing a TVB node,
                # to the target NEST node
                # For this output TVB state variable:
                # ...transmit it to the corresponding NEST devices,
                # ...which represent each TVB node
                interface.SetStatus({"amplitude": self.w_tvb_sv_to_nest_current[self.tvb_nodes_ids] * \
                                                  state[interface.tvb_sv_id, self.tvb_nodes_ids].squeeze(),
                                     "origin": self.nest_instance.GetKernelStatus("time"),
                                     "start": self.nest_instance.GetKernelStatus("min_delay"),
                                     "stop": self.dt})
            elif interface.model in ["poisson_generator", "spike_generator"]:
                # Rate is already a meanfield quantity.
                # All neurons of the target NEST spiking populations
                # will receive the same spike rate.
                # No further scaling is required with the population size (number of neurons)
                # For this output TVB state variable:
                # ...convert to NEST rate for every TVB node..
                # For this output TVB state variable:
                # ...convert to NEST rate for every TVB node...
                nest_rate = np.maximum(0,
                                       self.w_tvb_sv_to_nest_spike_rate[self.tvb_nodes_ids] *
                                       state[interface.tvb_sv_id, self.tvb_nodes_ids].squeeze())
                if interface.model in ["poisson_generator"]:
                    # ...and transmit it to the corresponding NEST device,
                    # ...which represents that TVB node
                    interface.SetStatus({"rate": nest_rate,
                                         "origin": self.nest_instance.GetKernelStatus("time"),
                                         "start": self.nest_instance.GetKernelStatus("min_delay"),
                                         "stop": self.dt})
                else:  # "spike_generator"
                    # ...and transmit it to the corresponding NEST device,
                    # ...which represents that TVB node
                    # TODO: solve the problem of the spike times distribution to the whole tvb_dt interval...
                    interface.SetStatus({'spike_times': np.ones(len(nest_rate)) *
                                                        self.nest_instance.GetKernelStatus("min_delay"),
                                         'origin': self.nest_instance.GetKernelStatus("time"),
                                         'spike_weights': nest_rate})
            else:
                raise ValueError("Interface model %s is not supported yet!" % interface.model)

    def nest_state_to_tvb_parameter(self, model):
        # This method runs at time t, before simulating time step t to t+dt
        for interface_id in self.nest_to_tvb_params_interfaces_ids:
            # ...update them:
            interface = self.nest_to_tvb_interfaces[interface_id]
            # Update TVB parameter
            param_values = getattr(model, interface.name)
            if interface.model == "spike_detector":
                param_values[self.nest_nodes_ids] = \
                    self.w_nest_spikes_to_tvb_sv[self.nest_nodes_ids] / \
                    np.array(interface.number_of_connections) * \
                    np.array(interface.number_of_spikes)
                interface.reset
            # TODO: add any other possible NEST output devices to TVB parameters interfaces here!
            else:
                raise ValueError("Interface model %s is not supported yet!" % interface.model)
            setattr(model, "__" + interface.name, param_values)
        return model

    def nest_state_to_tvb_state(self, state):
        # This method runs at time t+dt, after simulating time step t to t+dt
        for interface_id in self.nest_to_tvb_sv_interfaces_ids:
            interface = self.nest_to_tvb_interfaces[interface_id]
            # Update TVB state
            if interface.model == "spike_detector":
                # The number of spikes has to be converted to a spike rate via division:
                #  by the total number of neurons to convert it to a mean field quantity,
                #  and by the time step dt, which is already included in the w_nest_spikes_to_tvb_sv scaling.
                # Instantaneous transmission. TVB history is used to buffer delayed communication.
                state[interface.tvb_sv_id, self.nest_nodes_ids, 0] = \
                    self.w_nest_spikes_to_tvb_sv[self.nest_nodes_ids] / \
                    np.array(interface.number_of_connections) * \
                    np.array(interface.number_of_spikes)
                interface.reset
            # TODO: add any other possible NEST output devices to TVB parameters interfaces here!
            else:
                raise ValueError("Interface model %s is not supported yet!" % interface.model)
        return state
