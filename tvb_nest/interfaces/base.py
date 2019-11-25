# -*- coding: utf-8 -*-

from abc import ABCMeta
from six import add_metaclass
from xarray import DataArray
import pandas as pd
import numpy as np
from tvb_nest.config import CONFIGURED
from tvb_nest.simulator_nest.models.devices import NESTInputDeviceDict
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_scripts.utils.data_structures_utils import is_integer
from tvb_scripts.time_series.model import TimeSeries, TimeSeriesRegion


LOG = initialize_logger(__name__)


DEVICES = NESTInputDeviceDict.keys()
PARAMETERS = ["current", "potential"]


@add_metaclass(ABCMeta)
class TVBNESTInterface(object):

    nest_instance = None
    tvb_model = None
    dt = 0.1

    tvb_nodes_ids = []
    nest_nodes_ids = []

    nest_network = None

    tvb_to_nest_interfaces = None
    tvb_to_nest_params = []
    nest_to_tvb_interfaces = None
    nest_to_tvb_params_interfaces_ids = None
    nest_to_tvb_sv_interfaces_ids = None
    nest_to_tvb_params = []

    # TVB <-> NEST transformations' weights/funs
    # If set as weights, they will become a transformation function of
    # lambda state, regions_indices: w[regions_indices] * state[regions_indices]
    # If set as a function of lambda state: fun(state), it will become a vector function of:
    # lambda state, regions_indices: np.array([fun(state[index]) for index in regions_indices)])
    w_spikes_to_tvb_rate = 1.0  # (assuming spikes/ms in TVB)
    w_spikes_to_tvb_sv = 1.0
    w_spike_sv_to_tvb_sv = 1.0
    w_V_m_to_tvb_sv = 1.0
    w_tvb_sv_to_spike_rate = 1000.0  # (spike rate in NEST is in spikes/sec, whereas dt is in ms)
    w_tvb_sv_to_current = 1000.0  # (1000.0 (nA -> pA), because I_e, and dc_generator amplitude in NEST are in pA)
    w_tvb_sv_to_V_m = 1.0  # assuming mV in both NEST and TVB

    def __init__(self, config=CONFIGURED.nest):
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

    def _configure_parameters(self):
        self.w_spikes_to_tvb_sv /= self.dt  # Used to convert number of spikes to a spike rate
        self.w_spikes_to_tvb_rate /= self.dt  # Used to convert number of spikes to a spike rate
        dummy = np.ones((self.number_of_nodes, ))
        # Confirm good shape for TVB-NEST interface model parameters
        # TODO: find a possible way to differentiate scalings between
        #  receiver (as in _tvb_state_to_nest_current),
        #  and sender (as in all other cases below), node indexing
        #  Also, the size doesn't have to be in all cases equal to number_of_nodes,
        #  but sometimes equal to number_of_nest_nodes or to number_of_tvb_nodes
        for prop in ["w_tvb_sv_to_current",
                     "w_tvb_sv_to_V_m",
                     "w_tvb_sv_to_spike_rate"]:
            property = getattr(self, prop)
            if hasattr(property, "__call__"):
                # If the property is already set as a function:
                fun = lambda state_variable, region_nodes_indices=None: property(state_variable[region_nodes_indices])
            else:
                property = dummy * property
                # If the property is set as a function just as a weight:
                fun = \
                    lambda state_variable, region_nodes_indices, weights=property: \
                        state_variable[region_nodes_indices] * weights[region_nodes_indices]
            setattr(self, prop, fun)
        for prop in ["w_spikes_to_tvb_rate",
                     "w_spikes_to_tvb_sv",
                     "w_spike_sv_to_tvb_sv",
                     "w_V_m_to_tvb_sv"]:
            property = getattr(self, prop)
            if hasattr(property, "__call__"):
                # If the property is already set as a function:
                fun = lambda nest_variable, region_nodes_indices=None: property(nest_variable)
            else:
                # If the property is set as a function just as a weight:
                property = dummy * property
                fun = \
                    lambda nest_variable, region_nodes_indices, weights=property: \
                        nest_variable * weights[region_nodes_indices]
            setattr(self, prop, fun)

    def configure(self, tvb_model):
        """  """
        self.tvb_to_nest_params = []
        for interface_id, interface in enumerate(self.tvb_to_nest_interfaces):
            try:
                tvb_sv_or_coupling_id = getattr(interface, "tvb_sv_id")
            except:
                tvb_sv_or_coupling_id = getattr(interface, "tvb_coupling_id")
            if not is_integer(tvb_sv_or_coupling_id) or not tvb_sv_or_coupling_id >= 0:
                self.tvb_to_nest_params.append(interface.name)
        self.nest_to_tvb_params = []
        self.nest_to_tvb_params_interfaces_ids = []
        self.nest_to_tvb_sv_interfaces_ids = []
        for interface_id, interface in enumerate(self.nest_to_tvb_interfaces):
            if is_integer(interface.tvb_sv_id) and interface.tvb_sv_id >= 0:
                self.nest_to_tvb_sv_interfaces_ids.append(interface_id)
            else:
                self.nest_to_tvb_params_interfaces_ids.append(interface_id)
                self.nest_to_tvb_params.append(interface.name)
        self.tvb_model = tvb_model
        self._configure_parameters()

    def tvb_state_to_nest(self, state, coupling, stimulus, model):
        for interface in self.tvb_to_nest_interfaces:
            if interface.model in DEVICES:
                if interface.tvb_sv_id is not None:
                    values = state[interface.tvb_sv_id].squeeze()  # if we need the state variable
                else:
                    values = getattr(model, interface.name)  # if we need a parameter
                if interface.model == "dc_generator":
                    transform_fun = self.w_tvb_sv_to_current
                    # We assume that current is a mean field quantity
                    # applied equally and in parallel
                    # to all target neurons of the NEST spiking populations
                    # This is why no scaling has been applied
                    # for the synaptic weight from the dc_generator device, representing a TVB node,
                    # to the target NEST node
                    # For this output TVB state variable:
                    # ...transmit it to the corresponding NEST devices,
                    # ...which represent each TVB node
                elif interface.model in ["poisson_generator", "spike_generator"]:
                    # Rate is already a meanfield quantity.
                    # All neurons of the target NEST spiking populations
                    # will receive the same spike rate.
                    # No further scaling is required with the population size (number of neurons)
                    # For this output TVB state variable:
                    # ...convert to NEST rate for every TVB node..
                    # For this output TVB state variable:
                    # ...convert to NEST rate for every TVB node...
                    transform_fun = self.w_tvb_sv_to_spike_rate
            elif interface.model in PARAMETERS:
                values = coupling[interface.tvb_coupling_id].squeeze()
                if interface.model == "current":
                    transform_fun = self.w_tvb_sv_to_current
                    # We assume that current is a mean field quantity
                    # applied equally and in parallel
                    # to all target neurons of the NEST spiking populations
                    # Instantaneous transmission. TVB history is used to buffer delayed communication.
                elif interface.model == "potential":
                    transform_fun = self.w_tvb_sv_to_V_m
                    # We assume that potential is a mean field quantity
                    # applied equally and in parallel
                    # to all target neurons of the NEST spiking populations
                    # Instantaneous transmission. TVB history is used to buffer delayed communication.
            else:
                raise ValueError("Interface model %s is not supported yet!" % interface.model)
            values = interface.scale * \
                     transform_fun(values, interface.nodes_ids)
            interface.set(values)

    def nest_state_to_tvb_parameter(self, model):
        # This method runs at time t, before simulating time step t to t+dt
        for interface_id in self.nest_to_tvb_params_interfaces_ids:
            # ...update them:
            interface = self.nest_to_tvb_interfaces[interface_id]
            # Update TVB parameter
            param_values = getattr(model, interface.name)
            if interface.model in ["spike_detector", "spike_multimeter"]:
                transform_fun = self.w_spikes_to_tvb_rate
                values = interface.population_spikes_number
                interface.reset
            elif interface.model == "multimeter":
                transform_fun = self.w_spike_sv_to_tvb_sv
                values = interface.current_population_mean_values
            elif interface.model == "voltmeter":
                transform_fun = self.w_V_m_to_tvb_sv
                values = interface.current_population_mean_values
            # TODO: add any other possible NEST output devices to TVB parameters interfaces here!
            else:
                raise ValueError("Interface model %s is not supported yet!" % interface.model)
            param_values[interface.nodes_ids] = \
                interface.scale * transform_fun(values, interface.nodes_ids)
            setattr(model, "__" + interface.name, param_values)
        return model

    def nest_state_to_tvb_state(self, state):
        # This method runs at time t+dt, after simulating time step t to t+dt
        for interface_id in self.nest_to_tvb_sv_interfaces_ids:
            interface = self.nest_to_tvb_interfaces[interface_id]
            # Update TVB state
            if interface.model in ["spike_detector", "spike_multimeter"]:
                # The number of spikes has to be converted to a spike rate via division:
                #  by the total number of neurons to convert it to a mean field quantity,
                #  and by the time step dt, which is already included in the w_nest_spikes_to_tvb_sv scaling.
                # Instantaneous transmission. TVB history is used to buffer delayed communication.
                state[interface.tvb_sv_id, interface.nodes_ids, 0] = \
                    interface.scale * \
                    self.w_spikes_to_tvb_sv(interface.population_spikes_number,
                                            interface.nodes_ids)
                interface.reset
            elif interface.model == "multimeter":
                # Instantaneous transmission. TVB history is used to buffer delayed communication.
                state[interface.tvb_sv_id, interface.nodes_ids, 0] = \
                    interface.scale * \
                    self.w_spike_sv_to_tvb_sv(interface.current_population_mean_values,
                                              interface.nodes_ids)

            elif interface.model == "voltmeter":
                # Instantaneous transmission. TVB history is used to buffer delayed communication.
                state[interface.tvb_sv_id, interface.nodes_ids, 0] = \
                    interface.scale * \
                    self.w_V_m_to_tvb_sv(interface.current_population_mean_values,
                                         interface.nodes_ids)
            # TODO: add any other possible NEST output devices to TVB parameters interfaces here!
            else:
                raise ValueError("Interface model %s is not supported yet!" % interface.model)
        return state

    def get_mean_data_from_NEST_multimeter_to_TVBTimeSeries(self, **kwargs):
        # the keys of which correspond to population level labels,
        # and the values to lists of data returned for each node region NEST network.
        # In the case of multimeter mean data, they also take the form of
        # dictionaries of variables measured by multimeters
        mean_data = self.nest_network.get_mean_data_from_multimeter(**kwargs)
        if mean_data is None:
            return None
        connectivity = kwargs.pop("connectivity", None)
        if connectivity is None:
            time_series = TimeSeries()
        else:
            time_series = TimeSeriesRegion(connectivity=connectivity)
        if isinstance(mean_data, DataArray):
            return time_series.from_xarray_DataArray(mean_data)
        else:  # Assuming a pd.Series or a dict of xarrays.DataArray
            output_xarrays = []
            pop_names = []
            for d_name, d in mean_data.iteritems():
                output_xarrays.append(time_series.from_xarray_DataArray(d))
                pop_names.append(d_name)
            if isinstance(mean_data, pd.Series):
                name = mean_data.name
            else:
                name = kwargs.get("name", "Mean data from NEST multimeter")
            return pd.Series(output_xarrays, index=pd.Index(pop_names, name="Population"), name=name)

    def get_mean_spikes_rates_from_NEST_to_TVBTimeSeries(self, **kwargs):
        # rate is a DataFrame
        # the keys of which correspond to population level labels,
        # and the values to lists of data returned for each node region NEST network.
        rates, spike_detectors = self.nest_network.compute_mean_spikes_rates(**kwargs)
        if rates is None:
            return None, None
        connectivity = kwargs.pop("connectivity", None)
        if connectivity is None:
            time_series = TimeSeries()
        else:
            time_series = TimeSeriesRegion(connectivity=connectivity)
        if isinstance(rates, DataArray):
            return time_series.from_xarray_DataArray(rates), spike_detectors
        else:
            # Assuming a pd.Series or a dict of xarrays.DataArray
            output_xarrays = []
            pop_names = []
            for d_name, d in rates.iteritems():
                output_xarrays.append(time_series.from_xarray_DataArray(d))
                pop_names.append(d_name)
            if isinstance(rates, pd.Series):
                name = rates.name
            else:
                name = kwargs.get("name", "Mean rates from NEST spike detectors")
            return pd.Series(output_xarrays, index=pd.Index(pop_names, name="Population"), name=name), \
                   spike_detectors
