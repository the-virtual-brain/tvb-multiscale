# -*- coding: utf-8 -*-
import os
import time
from collections import OrderedDict
import numpy as np
from scipy.stats import spearmanr
from xarray import DataArray
from pandas import MultiIndex

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.config import Config, CONFIGURED
from tvb_multiscale.io.h5_writer import H5Writer
from tvb_multiscale.plot.plotter import Plotter
from tvb_utils.utils import safe_makedirs

from tvb.simulator.simulator import Simulator
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.coupling import Linear
from tvb.simulator.integrators import IntegratorStochastic, HeunStochastic
from tvb.simulator.monitors import Raw, TemporalAverage
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion
from tvb.contrib.scripts.datatypes.time_series_xarray import TimeSeriesRegion as TimeSeriesRegionX
from tvb.contrib.scripts.service.time_series_service import TimeSeriesService
from tvb.contrib.scripts.service.head_service import HeadService
from tvb.contrib.scripts.utils.data_structures_utils import is_integer, ensure_list


def mean_field_per_population(source_ts, populations, pop_sizes):
    ts_service = TimeSeriesService()
    pop_cumsum = np.cumsum([0] + pop_sizes).tolist()
    mean_field = []
    for i_pop, (pop_name, pop_inds, pop_size) in enumerate(zip(populations, pop_cumsum[1:], pop_sizes)):
        mean_field.append(
            ts_service.sum_across_dimension(
                source_ts.get_modes_by_index(
                    np.arange(pop_cumsum[i_pop], pop_inds).astype("i")),
                3)
        )
        mean_field[-1].data /= pop_size
        mean_field[-1].title = "Region mean field time series of %s population" % pop_name
    mean_field = ts_service.concatenate_modes(mean_field)
    labels_dimensions = mean_field.labels_dimensions
    try:
        del labels_dimensions[mean_field.labels_ordering[3]]
    except:
        pass
    mean_field.update_dimension_names("Population", 3)
    labels_dimensions["Population"] = np.array(populations)
    mean_field.labels_dimensions = labels_dimensions
    mean_field.title = "Region mean field time series"
    return mean_field


def spikes_per_population(source_spikes, populations, pop_sizes):
    pop_cumsum = np.cumsum([0] + pop_sizes).tolist()
    spikes = []
    for i_pop, (pop_name, pop_inds) in enumerate(zip(populations, pop_cumsum[1:])):
        spikes.append(
            source_spikes.get_modes_by_index(
                np.arange(pop_cumsum[i_pop], pop_inds).astype("i")))
        spikes[-1].title = "Region spikes' time series of %s population" % pop_name
        labels_dimensions = spikes[-1].labels_dimensions
        try:
            del labels_dimensions[spikes[-1].labels_ordering[1]]
        except:
            pass
        spikes[-1].update_dimension_names("Population", 1)
        labels_dimensions["Population"] = np.array([pop_name])
        spikes[-1].labels_dimensions = labels_dimensions
    return spikes


def spike_rates_from_TVB_spike_ts(spikes, integrator_dt, pop_sizes):
    # spikes_ts are assumed to have an amplitude of tvb_integrator_dt / tvb_monitor_dt
    ts_service = TimeSeriesService()
    rate = []
    for i_pop, (spike_ts, pop_size) in enumerate(zip(ensure_list(spikes), pop_sizes)):
        this_rate = ts_service.sum_across_dimension(spike_ts, 3)
        this_rate.data = this_rate.data / integrator_dt * 1000 / pop_size
        rate.append(this_rate)
        try:
            del rate[-1].labels_dimensions[rate[-1].labels_ordering[3]]
        except:
            pass
        rate[-1].labels_ordering[3] = "Mode"
        rate[-1].title = "Spike rate"
    rate = ts_service.concatenate_variables(rate)
    rate.title = "Mean field population spike rates"
    return rate


def spike_rates_from_mean_field_rates(mean_field):
    rate = mean_field.get_state_variables("rate")
    try:
        del rate.labels_dimensions[rate.labels_ordering[1]]
        del rate.labels_dimensions[rate.labels_ordering[3]]
    except:
        pass
    rate = rate.swapaxes(1, 3)  # Swap "State Variable" with "Population"
    rate.labels_ordering[3] = "Mode"
    rate.title = "Mean field population spike rates"
    return rate


def pearson(x):
    return np.corrcoef(x.T)


def spearman(x):
    return spearmanr(x)[0]


def TimeSeries_correlation(ts, corrfun=pearson, force_dims=4):
    data = ts._data # Get the DataArray of TimeSeries
    if data.shape[-1] == 1:  # Get rid of the 4th dimension if it is only 1
        data = data.squeeze(axis=-1)
    dims = list(data.dims[1:])
    # Combine variables State Variable x Region x ...
    # and prepare the two dimensions (i, j) of the correlation matrix
    stacked_dims = "-".join(dims)
    names = []
    new_dims = []
    for d in ["i", "j"]:
        names.append([dim+"_"+d for dim in dims])
        new_dims.append(stacked_dims+"_"+d)
    # Stack data across combined variables
    data = data.stack(**{stacked_dims: tuple(data.dims[1:])})
    # Prepare output DataArray
    n = len(data.coords[stacked_dims])
    corrs = DataArray(np.empty((n, n), dtype="f"),
                      dims=new_dims,
                      coords={new_dims[0]: MultiIndex.from_tuples(data.coords[stacked_dims].values, names=names[0]),
                              new_dims[1]: MultiIndex.from_tuples(data.coords[stacked_dims].values, names=names[1])})
    try:
        # TODO: a better hack for when spearman returns nan
        corrs.values = corrfun(data.values)  # Compute all combinations of correlations across Time
    except:
        corrs.values = corrfun(data.values) * np.ones(corrs.values.shape)
    corrs = corrs.unstack(new_dims)  # Unstack the combinations of State Variable x Region x ...
    new_dims = list(corrs.dims)
    corrs = corrs.transpose(*tuple(new_dims[0::2] + new_dims[1::2]))  # Put variables in front of regions
    if force_dims is not None:  # Compute the mean over possible 4th dimension ("Mode", or "Neuron")
        while len(corrs.dims) > force_dims:
            corrs = corrs.mean(axis=-1)
    return corrs


class Workflow(object):
    config = Config(separate_by_run=False)

    name = "Workflow"

    pse_params = {}

    writer = True
    write_time_series = False
    plotter = True
    path = ""
    h5_file = None

    tvb_sim_numba = False

    connectivity_path = CONFIGURED.DEFAULT_CONNECTIVITY_ZIP
    decouple = False
    symmetric_connectome = False
    time_delays = True
    force_dims = None

    populations = ["Excitatory", "Inhibitory"]
    populations_sizes = [1, 1]

    simulator = None
    tvb_model = ReducedWongWangExcIO  # ReducedWongWangExcIOInhI
    tvb_spike_stimulus = None
    tvb_spiking_model = False
    model_params = {"TVB": {}, "NEST": {"E": {}, "I": {}}}
    mf_nodes_ids = []
    tvb_spikes_var = "spikes"
    tvb_rate_vars = ["R"]  # ["R_e", "R_i"]
    tvb_spike_rate_var = "rate"
    tvb_init_cond = 0.0
    tvb_monitor = Raw
    tvb_monitor_period = None

    dt = 0.1
    integrator = HeunStochastic
    tvb_noise_strength = 0.01
    transient = None
    simulation_length = 100.0
    print_progression_message = True

    spiking_regions_ids = []

    rates = None

    def __init__(self, **pse_params):
        self.pse_params = pse_params

    @property
    def number_of_regions(self):
        return self.connectivity.number_of_regions

    @property
    def figsize(self):
        return self.config.figures.DEFAULT_SIZE

    def _folder_name(self):
        folder = []
        for param, val in self.pse_params.items():
            folder.append("%s%g" % (param, np.mean(val).item()))
        return "_".join(folder)

    def reset(self, pse_params):
        self.pse_params = pse_params
        self.model_params = {"TVB": {}, "NEST": {"E": {}, "I": {}}}
        self.mf_nodes_ids = []
        self.tvb_spike_stimulus = None
        self.simulator = None
        self.rates = None

    def configure(self):
        if self.transient is None:
            self.transient = 0.1 * self.simulation_length
            self.simulation_length += self.transient
        self.duration = (self.simulation_length - self.transient) / 1000  # make it seconds
        if self.tvb_monitor_period is None:
            self.tvb_monitor_period = self.dt
        if self.tvb_monitor_period > self.dt:
            self.tvb_monitor = TemporalAverage
        else:
            self.tvb_monitor = Raw
        if self.writer or self.plotter:
            self.res_folder = os.path.join(self.config.out._folder_res.replace("res", self.name), self._folder_name())
            self.config.figures._out_base = self.res_folder
            safe_makedirs(self.res_folder)
            if self.writer:
                if self.writer is True:
                    self.writer = H5Writer()
                self.path = os.path.join(self.res_folder, "params.h5")
                self.create_file()
                if len(self.pse_params) > 0:
                    self.write_pse_params(close_file=True)
            if self.plotter:
                if self.plotter is True:
                    self.plotter = Plotter(self.config)

    def create_file(self, close_file=True):
        self.writer.write_mode = "w"
        self.h5_file, self.path = self.writer._open_file("multiple groups", self.path)
        self.close_file(close_file)
        self.writer.write_mode = "a"  # From now on we append file

    def open_file(self):
        if self.h5_file is None:
            self.h5_file, self.path = self.writer._open_file("multiple groups", self.path)
        return self.h5_file

    def close_file(self, close_file=True):
        if close_file:
            self.h5_file.close()
            self.h5_file = None

    def write_group(self, data, name, method, close_file=True):
        if self.h5_file is None:
            self.h5_file = self.open_file()
        try:
            self.h5_file[name]
        except:
            self.h5_file.create_group(name)
        getattr(self.writer, "write_%s" % method)(data, h5_file=self.h5_file[name], close_file=False)
        self.close_file(close_file)

    def write_ts(self, ts, name, recursive=True):
        self.writer.write_tvb_to_h5(TimeSeriesRegion().from_xarray_DataArray(ts._data,
                                                                             connectivity=ts.connectivity),
                                    self.path.replace("params.h5", "%s.h5" % name), recursive)

    def write_object(self, ts, name):
        self.writer.write_mode = "w"
        self.writer.write_object(ts, path=self.path.replace("params.h5", "%s.h5" % name))
        self.writer.write_mode = "a"

    @property
    def general_parameters(self):
        return {"populations": self.populations, "spiking_regions_ids": self.spiking_regions_ids,
                "tvb_connectivity_path": self.connectivity_path,
                "decouple": self.decouple, "time_delays": self.time_delays, "force_dims": str(self.force_dims),
                "transient": self.transient, "simulation_length": self.simulation_length,
                "path": self.config.out._out_base}

    def write_pse_params(self, close_file=True):
        self.write_group(self.pse_params, "pse_params", "dictionary", close_file=close_file)

    def write_general_params(self, close_file=True):
        self.write_group(self.general_parameters, "general_params", "dictionary", close_file=close_file)

    def write_model_params(self, close_file=True):
        self.write_group(self.model_params, "model_params", "dictionary", close_file=close_file)

    def force_dimensionality(self):
        if is_integer(self.force_dims):
            inds = np.arange(self.force_dims).astype("i")
        else:
            inds = np.array(ensure_list(self.force_dims))
        self.connectivity = HeadService().slice_connectivity(self.connectivity, inds)

    @property
    def tvb_model_dict(self):
        tvb_model_dict = dict(vars(self.simulator.model))
        tvb_model_dict["gid"] = str(tvb_model_dict["gid"])
        del tvb_model_dict["log"]
        del tvb_model_dict["observe"]
        return tvb_model_dict

    @property
    def integrator_dict(self):
        integrator_dict = {"integrator": self.simulator.integrator.__class__.__name__,
                           "dt": self.simulator.integrator.dt}
        if isinstance(self.simulator.integrator, IntegratorStochastic):
            integrator_dict.update({"noise": self.simulator.integrator.noise.__class__.__name__,
                                    "noise_strength": self.simulator.integrator.noise.nsig})
        else:
            integrator_dict.update({"noise": "",
                                    "noise_strength": np.array([0.0, ])})
        return integrator_dict

    @property
    def spike_stimulus_dict(self):
        spike_stimulus_dict = OrderedDict()
        for spike_target in self.tvb_spike_stimulus.keys():
            spike_stimulus_dict[spike_target] = self.simulator._spike_stimulus_fun(spike_target, slice(None))
        return spike_stimulus_dict

    @property
    def coupling_dict(self):
        coupling_dict = OrderedDict()
        coupling_dict["coupling"] = self.simulator.coupling.__class__.__name__
        coupling_dict["a"] = self.simulator.coupling.a
        coupling_dict["b"] = self.simulator.coupling.b
        return coupling_dict

    def write_tvb_simulator(self):
        self.writer.write_tvb_to_h5(self.simulator.connectivity,
                                    os.path.join(self.res_folder, "Connectivity.h5"))
        # self.write_group(self.simulator.connectivity, "connectivity", "connectivity", close_file=False)
        self.write_group(self.tvb_model_dict, "tvb_model", "dictionary", close_file=False)
        if self.tvb_spike_stimulus is not None:
            self.write_group(self.spike_stimulus_dict, "spike_stimulus", "dictionary", close_file=False)
        if self.number_of_regions > 1:
            self.write_group(self.coupling_dict, "coupling", "dictionary", close_file=False)
        self.write_group(self.integrator_dict, "integrator", "dictionary", close_file=False)
        self.write_group({"monitor": self.tvb_monitor.__name__,
                          "period": self.tvb_monitor_period}, "monitor", "dictionary", close_file=True)

    def prepare_connectivity(self):
        if os.path.isfile(self.connectivity_path):
            self.connectivity = Connectivity.from_file(self.connectivity_path)
        if self.force_dims is not None:
            self.force_dimensionality()
        if self.decouple:
            self.connectivity.weights *= 0.0
        elif self.connectivity.weights.max() > 0.0:
            self.connectivity.weights = self.connectivity.scaled_weights(mode="region")
            if self.symmetric_connectome:
                self.connectivity.weights = np.sqrt(self.connectivity.weights * self.connectivity.weights.T)
                self.connectivity.tract_lengths = np.sqrt(self.connectivity.tract_lengths * self.connectivity.tract_lengths.T)
            self.connectivity.weights /= np.percentile(self.connectivity.weights, 95)
            self.connectivity.weights[self.connectivity.weights > 1.0] = 1.0
        if not self.time_delays:
            self.connectivity.tract_lengths *= 0.0
        self.connectivity.configure()

    def prepare_initial_conditions(self):
        initial_conditions = \
            self.tvb_init_cond * np.ones((self.simulator.horizon,
                                          self.simulator.model.nvar,
                                          self.simulator.connectivity.number_of_regions,
                                          self.simulator.model.number_of_modes))
        self.simulator._configure_history(initial_conditions=initial_conditions)

    def prepare_simulator(self):
        self.prepare_connectivity()
        self.simulator = Simulator()
        self.simulator.connectivity = self.connectivity
        self.simulator.model = self.tvb_model(**(self.model_params["TVB"]))
        self.tvb_spiking_model = \
            isinstance(self.simulator.model, (SpikingWongWangExcIOInhI, ))  # MultiscaleWongWangExcIOInhI
        if self.tvb_spiking_model:
            self.simulator.model.N_E = np.array([self.populations_sizes[0], ])
            self.simulator.model.N_I = np.array([self.populations_sizes[1], ])
            # if isinstance(self.simulator.model, MultiscaleWongWangExcIOInhI):
            #     self.simulator.model._spiking_regions_inds = self.spiking_regions_ids
            # else:
            # All of them are spiking regions
            self.spiking_regions_ids = list(range(self.simulator.connectivity.number_of_regions))
        # if self.number_of_regions > 1:
        #     self.simulator.coupling = Linear(a=np.array([1.0, ]), b=np.array([0.0, ]))
        self.simulator.model.configure()
        self.simulator.integrator = self.integrator()
        self.simulator.integrator.dt = self.dt
        if isinstance(self.simulator.integrator, IntegratorStochastic):
            # Some code only for SpikingWongWangExcIOInhI & MultiscaleWongWangExcIOInhI
            self.simulator.integrator.noise.nsig = np.array(self.simulator.model.nvar * [self.tvb_noise_strength])
            if self.tvb_spiking_model:
                self.simulator.integrator.noise.nsig[6:] = 0.0  # No noise for t_ref and derived variables
            elif isinstance(self.simulator.model, ReducedWongWangExcIOInhI):
                self.simulator.integrator.noise.nsig[2:] = 0.0  # No noise for R_e, R_i, Rin_e, Rin_i, I_e, I_i
            else:
                self.simulator.integrator.noise.nsig[1:] = 0.0  # No noise for R, Rin, I variables
        self.simulator.integrator.configure()
        mon_raw = self.tvb_monitor(period=self.tvb_monitor_period)
        self.simulator.monitors = (mon_raw,)  # mon_bold, mon_eeg
        self.simulator.spike_stimulus = self.tvb_spike_stimulus
        # Configure the simulator
        self.simulator.use_numba = self.tvb_sim_numba
        self.simulator.configure()
        self.prepare_initial_conditions()
        if self.plotter and self.simulator.connectivity.number_of_regions > 1:
            self.plotter.plot_tvb_connectivity(self.simulator.connectivity)
        if self.writer:
            self.write_tvb_simulator()

    def simulate(self):
        results = self.simulator.run(simulation_length=self.simulation_length,
                                     print_progression_message=self.print_progression_message)
        try:
            tvb_ts = TimeSeriesRegionX(results[0][1], time=results[0][0],
                                       connectivity=self.simulator.connectivity,
                                       labels_ordering=["Time", "State Variable", "Region", "Neurons"],
                                       labels_dimensions={
                                            "State Variable": ensure_list(self.simulator.model.state_variables),
                                            "Region": self.simulator.connectivity.region_labels.tolist()},
                                       sample_period=self.simulator.integrator.dt)
        except:
            print("No TVB time series have been loaded!")
            tvb_ts = None
        del results
        if tvb_ts is not None and self.transient:
            tvb_ts = tvb_ts[self.transient:]
        return tvb_ts

    def get_mean_field(self, tvb_ts):
        mf_ts = None
        if self.tvb_spiking_model:
            mf_ts = mean_field_per_population(tvb_ts, self.populations, self.populations_sizes)
        return mf_ts

    def get_tvb_spikes(self, tvb_ts):
        tvb_spikes = None
        if self.tvb_spiking_model:
            tvb_spikes = spikes_per_population(
                    tvb_ts.get_state_variables(
                        self.tvb_spikes_var).get_subspace_by_index(self.spiking_regions_ids),
                 self.populations, self.populations_sizes)
        return tvb_spikes

    def get_tvb_rates(self, tvb_ts=None, mf_ts=None, tvb_spikes=None):
        if self.tvb_spiking_model:
            if self.tvb_spike_rate_var not in mf_ts.labels_dimensions["State Variable"]:
                tvb_rates = \
                    spike_rates_from_TVB_spike_ts(tvb_spikes, self.simulator.integrator.dt, self.populations_sizes)
                tvb_rates.title = "Region mean field spike rate time series"
            else:
                mf_ts[:, self.tvb_spike_rate_var, :, :].data /= \
                    (self.simulator.integrator.dt * 0.001)  # rate in Hz
                tvb_rates = spike_rates_from_mean_field_rates(mf_ts)
                tvb_rates.title = "Region mean field rate time series"
        else:
            tvb_rates = tvb_ts[:, self.tvb_rate_vars]
            tvb_rates.name = "Region mean field rate time series"
        return DataArray(tvb_rates.mean(axis=0).squeeze(axis=-1),
                         dims=tvb_rates.dims[1:3],
                         coords={tvb_rates.dims[1]: tvb_rates.coords[tvb_rates.dims[1]],
                                 tvb_rates.dims[2]: tvb_rates.coords[tvb_rates.dims[2]]}), tvb_rates

    def get_tvb_corrs(self, tvb_rates):
        return {"Pearson": TimeSeries_correlation(tvb_rates, corrfun=pearson, force_dims=4),
                "Spearman": TimeSeries_correlation(tvb_rates, corrfun=spearman, force_dims=4)}

    def plot_tvb_ts(self, tvb_ts, mf_ts=None):
        if mf_ts is None:
            mf_ts = tvb_ts
        # For timeseries plot:
        mf_ts.plot_timeseries(plotter_config=self.plotter.config, per_variable=True,
                              figsize=self.figsize, add_legend=False)
        # For raster plot:
        if self.number_of_regions > 9:
            mf_ts.plot_raster(plotter_config=self.plotter.config, per_variable=True,
                              figsize=self.figsize, add_legend=False)
        n_spiking_nodes_ids = len(self.spiking_regions_ids)
        if n_spiking_nodes_ids > 0:
            if n_spiking_nodes_ids < self.simulator.connectivity.number_of_regions:
                mf_ts[:, :, self.spiking_regions_ids].plot_timeseries(plotter_config=self.plotter.config,
                                                                      per_variable=True, figsize=self.figsize,
                                                                      figname="Spiking nodes mean-field "
                                                                              "TVB Time Series")
                if n_spiking_nodes_ids > 3:
                    mf_ts[:, :, self.spiking_regions_ids].plot_raster(plotter_config=self.plotter.config,
                                                                      per_variable=True, figsize=self.figsize,
                                                                      figname="Spiking nodes TVB Time Series Raster")
            tvb_ts[:, :, self.spiking_regions_ids].plot_map(y=tvb_ts._data.dims[3],
                                                            row=tvb_ts._data.dims[2],
                                                            per_variable=True,
                                                            figname="Spiking nodes TVB Time Series",
                                                            figsize=self.figsize,
                                                            plotter_config=self.plotter.config)

    def run(self, model_params={}):
        self.model_params.update(model_params)
        if self.writer:
           self.write_general_params(close_file=False)
           if len(self.model_params) > 0:
               self.write_model_params(close_file=True)

        # ----------------------1. Define a TVB simulator (model, integrator, monitors...)------------------------------
        print("Preparing TVB simulator...")
        self.prepare_simulator()

        self.mf_nodes_ids = list(range(self.simulator.connectivity.number_of_regions))

        # ------2. Build the NEST network model (fine-scale regions' nodes, stimulation devices, spike_detectors etc)---

        if len(self.spiking_regions_ids) > 0:
            for ind in self.spiking_regions_ids:
                self.mf_nodes_ids.remove(ind)

        # -----------------------------------3. Simulate and gather results---------------------------------------------
        # ...and simulate!
        print("Simulating TVB...")
        t_start = time.time()
        tvb_ts = self.simulate()
        print("\nSimulated in %g secs!" % (time.time() - t_start))

        # -----------------------------------4. Compute rate per region and population----------------------------------
        self.rates = {}
        self.corrs = {}
        if tvb_ts is not None:
            if self.writer and self.write_time_series:
                self.write_ts(tvb_ts, "TVB_TimeSeries", recursive=True)
            mf_ts = self.get_mean_field(tvb_ts)
            if self.plotter:
                self.plot_tvb_ts(tvb_ts, mf_ts)
            tvb_spikes = self.get_tvb_spikes(tvb_ts)
            if tvb_spikes is not None and self.plotter:
                for i_pop, spike in enumerate(tvb_spikes):
                    spike.plot(y=spike._data.dims[3], row=spike._data.dims[2],
                               cmap="jet", figsize=(20, 10), plotter_config=self.plotter.config)
            self.rates["TVB"], tvb_rates = self.get_tvb_rates(tvb_ts, mf_ts, tvb_spikes)
            del tvb_ts  # Free memory...
            del mf_ts  # Free memory...
            del tvb_spikes  # Free memory...
            self.corrs["TVB"] = self.get_tvb_corrs(tvb_rates)
            if self.plotter:
                tvb_rates.plot_timeseries(plotter_config=self.plotter.config, figsize=self.figsize)
                del tvb_rates  # Free memory
            if self.writer:
                self.write_object(self.rates["TVB"].to_dict(), "TVB_rates")
                self.write_object(self.corrs["TVB"], "TVB_corrs")

        return self.rates, self.corrs
