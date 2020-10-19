# -*- coding: utf-8 -*-


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