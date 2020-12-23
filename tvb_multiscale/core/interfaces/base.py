# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
from tvb_multiscale.core.config import CONFIGURED, initialize_logger, LINE
from tvb_multiscale.core.spiking_models.devices import \
    InputDeviceDict, OutputDeviceDict, OutputSpikeDeviceDict, OutputContinuousTimeDeviceDict

from tvb.contrib.scripts.utils.data_structures_utils \
    import is_integer, concatenate_heterogeneous_DataArrays


LOG = initialize_logger(__name__)


PARAMETERS = ["current", "potential"]


class TVBSpikeNetInterface(object):

    # This is the actual interface class between TVB and a SpikingNetwork

    _available_input_devices = InputDeviceDict.keys()
    _current_input_devices = []
    _spike_rate_input_devices = []
    _available_output_devices = OutputDeviceDict.keys()
    _spike_rate_output_devices = OutputSpikeDeviceDict.keys()
    _multimeter_output_devices = OutputContinuousTimeDeviceDict.keys()
    _voltmeter_output_devices = ["voltmeter"]

    tvb_model = None
    dt = 0.1

    tvb_nodes_ids = np.array([])
    spiking_nodes_ids = np.array([])

    spiking_network = None

    tvb_to_spikeNet_interfaces = []
    spikeNet_to_tvb_interfaces = []
    # spikeNet_to_tvb_params_interfaces_ids = []  # deprecate this due to use of non-state dynamical variables
    spikeNet_to_tvb_sv_interfaces_ids = []
    spikeNet_to_tvb_params = OrderedDict()

    def __init__(self, config=CONFIGURED):
        self.config = config
        LOG.info("%s created!" % self.__class__)

    def __repr__(self):
        return self.__class__.__name__ + " TVB-NEST Interface"

    def __str__(self):
        return self.print_str()

    def print_str(self, detailed_output=False, connectivity=False):
        output = 3*LINE + "%s\n\n" % self.__repr__()
        tvb_to_spikeNet_interfaces = ""
        if len(self.tvb_to_spikeNet_interfaces) > 0:
            for interface_index, interface in self.tvb_to_spikeNet_interfaces.iteritems():
                tvb_to_spikeNet_interfaces += \
                    LINE + "\nIndex: %s\n%s" % \
                    (interface_index, interface.print_str(detailed_output=detailed_output, connectivity=connectivity))
        if len(tvb_to_spikeNet_interfaces) > 0:
            output += LINE + "\n\nTVB to spikeNet interfaces:\n%s" % tvb_to_spikeNet_interfaces
        spikeNet_to_tvb_interfaces = ""
        if len(self.spikeNet_to_tvb_interfaces) > 0:
            for interface_index, interface in self.spikeNet_to_tvb_interfaces.iteritems():
                spikeNet_to_tvb_interfaces += \
                    LINE + "\nIndex: %s\n%s" % \
                    (interface_index, interface.print_str(detailed_output=detailed_output, connectivity=connectivity))
        if len(spikeNet_to_tvb_interfaces) > 0:
            output += 2*LINE + "\n\nTVB to spikeNet interfaces:\n%s" % spikeNet_to_tvb_interfaces
        return output

    @property
    def spiking_nodes(self):
        return self.spiking_network.brain_regions

    @property
    def spikeNet_min_delay(self):
        return self.spiking_network.min_delay

    @property
    def number_of_spiking_nodes(self):
        return len(self.spiking_nodes_ids)

    @property
    def number_of_tvb_nodes(self):
        return len(self.tvb_nodes_ids)

    @property
    def number_of_nodes(self):
        return len(np.union1d(self.tvb_nodes_ids , self.spiking_nodes_ids))

    @property
    def number_of_tvb_state_variables(self):
        return len(self.tvb_model.state_variables)

    def configure(self, tvb_model):
        self.tvb_model = tvb_model
        # Organize the different kinds of interfaces and set the TVB region model of the TVB Simulator
        self.spikeNet_to_tvb_params = OrderedDict()
        # self.spikeNet_to_tvb_params_interfaces_ids = []   # deprecated
        self.spikeNet_to_tvb_sv_interfaces_ids = []
        for interface_id, interface in enumerate(self.spikeNet_to_tvb_interfaces):
            if is_integer(interface.tvb_sv_id) and interface.tvb_sv_id >= 0:
                self.spikeNet_to_tvb_sv_interfaces_ids.append(interface_id)
            else:
                # self.spikeNet_to_tvb_params_interfaces_ids.append(interface_id)  # deprecated
                raise ValueError("tvb_sv_id=%s doesn't correspond "
                                 "to the index of a TVB state variable for interface %s!\n"
                                 % (str(interface.tvb_sv_id), str(interface)))
            # Even if the target in TVB is a state variable,
            # we are going to create a TVB parameter with the same name
            if interface.name not in list(self.spikeNet_to_tvb_params.keys()):
                self.spikeNet_to_tvb_params[interface.name] = interface.nodes_ids
            else:
                self.spikeNet_to_tvb_params[interface.name] += interface.nodes_ids
            self.spikeNet_to_tvb_params[interface.name] = \
                np.unique(self.spikeNet_to_tvb_params[interface.name]).tolist()

    def tvb_state_to_spikeNet(self, state, coupling, stimulus):
        # Apply TVB -> Spiking Network input at time t before integrating time step t -> t+dt
        for interface in self.tvb_to_spikeNet_interfaces:
            if interface.model in self._available_input_devices:
                values = state[interface.tvb_sv_id].squeeze()  # if we need the state variable
                if interface.model in self._current_input_devices:
                    transform_fun = self.transforms["tvb_to_current"]
                    # We assume that current is a mean field quantity
                    # applied equally and in parallel
                    # to all target neurons of the spiking populations
                    # This is why no scaling has been applied
                    # for the synaptic weight from the dc_generator device, representing a TVB node,
                    # to the target spiking node
                    # For this output TVB state variable:
                    # ...transmit it to the corresponding devices of the spiking network,
                    # ...which represent each TVB node
                elif interface.model in self._spike_rate_input_devices:
                    # Rate is already a meanfield quantity.
                    # All neurons of the target spiking populations
                    # will receive the same spike rate.
                    # No further scaling is required with the population size (number of neurons)
                    # For this output TVB state variable:
                    # ...convert to spiking rate for every TVB node..
                    # For this output TVB state variable:
                    # ...convert to spiking rate for every TVB node...
                    transform_fun = self.transforms["tvb_to_spike_rate"]
            elif interface.model in PARAMETERS:
                values = coupling[interface.tvb_coupling_id].squeeze()
                if interface.model == "current":
                    transform_fun = self.transforms["tvb_to_current"]
                    # We assume that current is a mean field quantity
                    # applied equally and in parallel
                    # to all target neurons of the spiking populations
                    # Instantaneous transmission. TVB history is used to buffer delayed communication.
                elif interface.model == "potential":
                    transform_fun = self.transforms["tvb_to_potential"]
                    # We assume that potential is a mean field quantity
                    # applied equally and in parallel
                    # to all target neurons of the spiking populations
                    # Instantaneous transmission. TVB history is used to buffer delayed communication.
            else:
                raise ValueError("Interface model %s is not supported yet!" % interface.model)
            # General form: interface_scale_weight * transformation_of(TVB_state_values)
            values = interface.scale * \
                     transform_fun(values, interface.nodes_ids)
            interface.set(values)

    # Deprecated
    # def spikeNet_state_to_tvb_parameter(self, model):
    #     # Apply Spiking Network -> TVB parameter input at time t before integrating time step t -> t+dt
    #     for interface_id in self.spikeNet_to_tvb_params_interfaces_ids:
    #         # ...update them:
    #         interface = self.spikeNet_to_tvb_interfaces[interface_id]
    #         # Update TVB parameter
    #         param_values = getattr(model, interface.name)
    #         if interface.model in self._spike_rate_output_devices:
    #             transform_fun = self.transforms["spikes_to_tvb"]
    #             values = interface.population_mean_spikes_number
    #             interface.reset  # We need to erase the spikes we have already read and communicated to TVB
    #         elif interface.model == self._multimeter_output_devices:
    #             transform_fun = self.transforms["spikes_sv_to_tvb"]
    #             values = interface.current_population_mean_values
    #         elif interface.model == self._voltmeter_output_devices:
    #             transform_fun = self.transforms["potential_to_tvb"]
    #             values = interface.current_population_mean_values
    #         # TODO: add any other possible Spiking Network output devices to TVB parameters interfaces here!
    #         else:
    #             raise ValueError("Interface model %s is not supported yet!" % interface.model)
    #         # General form: interface_scale_weight * transformation_of(SpikeNet_state_values)
    #         param_values[interface.nodes_ids] = \
    #             interface.scale * transform_fun(values, interface.nodes_ids)
    #         setattr(model, "__" + interface.name, param_values)
    #     return model

    def spikeNet_state_to_tvb_state(self, state):
        # Apply Spiking Network -> TVB state input at time t+dt after integrating time step t -> t+dt
        for interface_id in self.spikeNet_to_tvb_sv_interfaces_ids:
            interface = self.spikeNet_to_tvb_interfaces[interface_id]
            # Update TVB state
            if interface.model in self._spike_rate_output_devices:
                # The number of spikes has to be converted to a spike rate via division:
                #  by the total number of neurons to convert it to a mean field quantity,
                #  and by the time step dt, which is already included in the spikes_to_tvb scaling.
                # Instantaneous transmission. TVB history is used to buffer delayed communication.
                transform_fun = self.transforms["spikes_to_tvb"]
                values = interface.population_mean_spikes_number
            elif interface.model == self._multimeter_output_devices:
                # Instantaneous transmission. TVB history is used to buffer delayed communication.
                transform_fun = self.transforms["spikes_sv_to_tvb"]
                values = interface.current_population_mean_values
            elif interface.model == self._voltmeter_output_devices:
                # Instantaneous transmission. TVB history is used to buffer delayed communication.
                transform_fun = self.transforms["potential_to_tvb"]
                values = interface.current_population_mean_values
            # TODO: add any other possible Spiking Network output devices to TVB parameters interfaces here!
            else:
                raise ValueError("Interface model %s is not supported yet!" % interface.model)
            # General form: interface_scale_weight * transformation_of(SpikeNet_state_values)
            state[interface.tvb_sv_id, interface.nodes_ids, 0] = \
                interface.scale * transform_fun(values,  interface.nodes_ids)
        return state
