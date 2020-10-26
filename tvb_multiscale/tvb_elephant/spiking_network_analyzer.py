# -*- coding: utf-8 -*-

import numpy as np
from pandas import Series

from tvb_multiscale.core.config import initialize_logger
from tvb_multiscale.core.spiking_models.network import SpikingNetwork
from tvb_multiscale.core.spiking_models.devices import OutputSpikeDeviceDict

from tvb.basic.neotraits.api import HasTraits, Attr, Float
from tvb.datatypes import connectivity

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


LOG = initialize_logger(__name__)


class SpikingNetworkAnalyzer(HasTraits):

    """SpikingNetworkAnalyzer
       - gets data from a SpikingNetwork output devices,
       - or reads them from files,
       - performs computations of mean field quantities, including spikes' rates,
       - and returns the results in the desired output type
         (numpy.array, xarray.DataArray, TVB TimeSeries, pandas.Series of xarray.DataArray)
       """

    spikeNet = Attr(field_type=SpikingNetwork,
                    label="Spiking Network",
                    default=None,
                    required=False,
                    doc="""An instance of a SpikingNetwork class""")

    period = Float(
        label="Sampling period",
        default=1.0,
        required=True,
        doc="""Sampling period (ms)""")

    start_time = Float(
        label="Start time",
        default=None,
        required=False,
        doc="""Starting time of data (ms)""")

    end_time = Float(
        label="End time",
        default=None,
        required=False,
        doc="""Ending time of data (ms)""")

    transient = Float(
        label="Transient time",
        default=0.0,
        required=False,
        doc="""Transient time to be removed from data for any further analysis (ms)""")

    # Equivalent to elephants' kernel = "auto":
    spikes_kernel_width = Float(
        label="Kernel/window length",
        default=None,
        required=False,
        doc="""Kernel or sliding window time length (ms)""")

    spikes_kernel = Attr(field_type=str, default="auto", required=False)

    # "DataArray" for xarray.DataArray
    # "TVB" for TVB "TimeSeries" instances,
    # or "numpy" for numpy array,
    # or "Series" for pandas.Series
    # For multiple regions/populations, if "DataArray" is chosen we merge heterogeneous arrays
    output_type = Attr(field_type=str, default="DataArray", required=False)

    connectivity = Attr(
        field_type=connectivity.Connectivity,
        label="Long-range connectivity",
        default=None,
        required=False,
        doc="""A tvb.datatypes.Connectivity object which contains the
         structural long-range connectivity data (i.e., white-matter tracts). In
         combination with the ``Long-range coupling function`` it defines the inter-regional
         connections. These couplings undergo a time delay via signal propagation
         with a propagation speed of ``Conduction Speed``.
         \nIt is necessary if the output_type="TVB" for TimeSeriesRegion class instance.""")

    flatten_neurons_inds = Attr(field_type=bool, default=True, required=False,
                        doc="True, if we substitue the global neurons' indices by their local rank, False otherwise.")

    def get_spikeNet_spike_devices(self, mode="events", regions=None, populations_devices=None):
        """This method will loop though all Spiking Network's devices to return all devices of a given model.
           Arguments:
            - mode: if "activity",
                     the method looks for "multimeter" devices that record spikes' weights continuously in time.
                    Default = "events", looking for devices that record spikes' events.
            - regions: an optional sequence (list, tuple, array) of regions' nodes' indices to be selected.
                       Default = None, corresponding to returning the devices of all regions' nodes.
            - populations_devices: an optional sequence (list, tuple, array) of
                                   populations' devices' labels to be selected.
                                   Default = None, corresponding to returning the devices of all populations.
           Returns:
            - a Series of selected DeviceSet instances
        """
        spike_devices = Series()
        if not isinstance(self.spikeNet, SpikingNetwork):
            LOG.warning("There is no SpikingNetwork linked! Returning no spike recording devies at all!")
            return spike_devices
        if mode.find("activity") > -1:
            spike_devices = self.spiking_network.get_devices_by_model("spike_multimeter", regions=regions)
        else:
            for device_name in OutputSpikeDeviceDict.keys():
                spike_devices = self.get_devices_by_model(device_name, regions=regions)
                if len(spike_devices) > 0:
                    break  # If this is not an empty dict of devices
        if len(spike_devices) == 0:
            LOG.warning("No spike measuring device in this Spiking Network network!")
            return spike_devices
        if populations_devices is not None:
            populations_devices = np.intersect1d(list(spike_devices.index),
                                                 ensure_list(populations_devices)).tolist()
            if len(populations_devices) == 0:
                LOG.warning("No spike measuring device left after user selection!")
                return spike_devices
            spike_devices = spike_devices[populations_devices]
        return spike_devices


    def compute_spikeNet_mean_firing_rates(self):
        pass


def compute_spikes_rate_across_time(self, time, spikes_kernel_width, spikes_kernel_width_in_points=None,
                                    spikes_kernel=None, mode="per_neuron", flatten_neurons_inds=True,
                                    name=None, **kwargs):
    """This method computes spike rate across time.
       Arguments:
        time: the time vector
        spikes_kernel_width: float: the width of the sliding window kernel in ms
        spikes_kernel_width_in_points: int: the width of the sliding window kernel in time points,
                                       Default=None, and computed using the time vector
        spikes_kernel: an array of a sliding window. Default=None, in which case a rectangular kernel is formed
        mode: if "per_neuron" the output is returned for each neuron separetely.
              Otherwise, it is computed across all neurons. Default = "per_neuron"
        name: name of the data to be computed. Default = None,
              which defaults to device_model_name + " - Total spike rate across time"
       Returns:
        xarray.DataArray with spike rates' time series
    """
    if spikes_kernel is None:
        if spikes_kernel_width_in_points is None:
            spikes_kernel_width_in_points = np.maximum(1, int(np.ceil(spikes_kernel_width / np.diff(time))))
        # Default spikes' kernel is just a rectangular one, normalized with its width.
        spikes_kernel = np.ones((spikes_kernel_width_in_points, )) / spikes_kernel_width

    if name is None:
        name = self.model + " - Total spike rate across time"

    if mode == "per_neuron":
        senders_neurons = []
        rates = []
        # Computing separately per neuron
        for neuron, spikes_times in self.get_spikes_times_by_neurons(full_senders=True, **kwargs).items():
            senders_neurons.append(neuron)
            if len(spikes_times) > 0:
                # Getting spikes counts per time interval
                spikes_counts = compute_spikes_counts(spikes_times, time)
                # Computing rate as a convolution with spikes_kernel
                rates.append(spikes_rate_convolution(spikes_counts, spikes_kernel))
            else:
                rates.append(np.zeros(time.shape))
        if flatten_neurons_inds:
            senders_neurons = np.arange(len(senders_neurons))
        return xr.DataArray(rates, dims=["Neuron", "Time"], coords={"Neuron": senders_neurons,
                                                                    "Time": time})
    else:
        # Computing for the whole population(s) alltogether
        spikes_times = self.get_spikes_times(**kwargs)
        if len(spikes_times) > 0:
            # Getting spikes counts per time interval
            spikes_counts = compute_spikes_counts(spikes_times, time)
            # Computing rate as some kind of convolution with spikes_kernel
            rates = spikes_rate_convolution(spikes_counts, spikes_kernel)
        else:
            rates = np.zeros(time.shape)
        return xr.DataArray(rates, dims=["Time"], coords={"Time": time}, name=name)


def compute_mean_spikes_rate_across_time(self, time, spikes_kernel_width, spikes_kernel_width_in_points,
                                         spikes_kernel=None, name=None, **kwargs):
    """This method computes mean spike rate across time
       by dividing the total spike rate with the total number of neurons.
       Arguments:
        time: the time vector
        spikes_kernel_width: float: the width of the sliding window kernel in ms
        spikes_kernel_width_in_points: int: the width of the sliding window kernel in time points,
                                       Default=None, and computed using the time vector
        spikes_kernel: an array of a sliding window. Default=None, in which case a rectangular kernel is formed
        mode: if "per_neuron" the output is returned for each neuron separetely.
              Otherwise, it is computed across all neurons. Default = "per_neuron"
        name: name of the data to be computed. Default = None,
              which defaults to device_model_name + " - Mean spike rate accross time"
       Returns:
        xarray.DataArray with spike rates' time series
    """
    if name is None:
        name = self.model + " - Mean spike rate accross time"
    n_neurons = self.get_number_of_neurons()
    if n_neurons > 0:
        return self.compute_spikes_rate_across_time(time, spikes_kernel_width, spikes_kernel_width_in_points,
                                                    spikes_kernel=spikes_kernel, mode="total",
                                                    name=name, **kwargs) / n_neurons
    else:
        return xr.DataArray(0.0 * time, dims=["Time"], coords={"Time": time}, name=name)