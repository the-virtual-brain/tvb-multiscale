# -*- coding: utf-8 -*-
import os
import time
from copy import deepcopy
from collections import OrderedDict
import h5py

import numpy as np
from matplotlib import pyplot as pl
from xarray import DataArray

from tvb_multiscale.config import Config, CONFIGURED
from tvb_multiscale.examples.paperwork.workflow import Workflow
from tvb_multiscale.io.h5_writer import H5Writer
from tvb_utils.utils import print_toc_message, read_dicts_from_h5file_recursively
from tvb_elephant.spike_stimulus_builder import SpikeStimulusBuilder

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb.simulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI


def plot_result(PSE_params, result, name, path):
    arr = DataArray(data=result, dims=list(PSE_params.keys()), coords=dict(PSE_params), name=name)
    fig, axes = pl.subplots(nrows=2, ncols=2, gridspec_kw={"width_ratios": [1, 0.1]}, figsize=(10, 10))
    axes[1, 1].remove()
    im = arr.plot(x=arr.dims[0], y=arr.dims[1], ax=axes[0, 0], add_colorbar=False)
    axes[0, 0].set_xlabel("")
    fig.colorbar(im, cax=axes[0, 1], orientation="vertical", label=arr.name)
    arr.plot.line(x=arr.dims[0], hue=arr.dims[1], ax=axes[1, 0])
    pl.savefig(path)


def symmetric_connectivity(SC, Nreg):
    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
    connectivity.weights = np.zeros((Nreg, Nreg))
    SC = np.array(SC)
    SCsize = np.sqrt(np.sum(SC ** 2))
    inds = np.tril_indices(Nreg, -1)
    connectivity.weights[inds[0], inds[1]] = SC
    triu_inds = np.triu_indices(Nreg, 1)
    connectivity.weights[triu_inds[0], triu_inds[1]] = SC
    return connectivity, SC, SCsize, triu_inds


class PSEWorkflowBase(object):
    name = "PSEWorkflow"
    config = Config(separate_by_run=False)
    PSE = {"params": OrderedDict(), "n_params": OrderedDict(), "results": OrderedDict()}
    workflow = Workflow()
    writer = H5Writer()
    _plot_results = ["rate", "Pearson", "Spearman"]

    def __init__(self):
        self.folder_res = self.config.out.FOLDER_RES.replace("res", self.name)
        if not os.path.isdir(self.folder_res):
            os.makedirs(self.folder_res)
        self.res_path = os.path.join(self.folder_res, self.name + ".h5")
        self.folder_figs = os.path.join(self.folder_res, "figs")
        if not os.path.isdir(self.folder_figs):
            os.makedirs(self.folder_figs)

    def print_PSE(self, pse_params):
        msg = ""
        for p, val in pse_params.items():
            msg += p + "=%g " % val
        print(msg)

    def configure_PSE(self):
        self.pse_shape = []
        for p in list(self.PSE["params"].keys()):
            np = "n_"+p
            self.PSE["n_params"][np] = len(self.PSE["params"][p])
            self.pse_shape.append(self.PSE["n_params"][np])
        self.pse_shape = tuple(self.pse_shape)

    def pse_to_model_params(self):
        pass

    def results_to_PSE(self, i1, i2, rates, corrs=None):
        pass

    def write_PSE(self):
        self.writer.write_dictionary(self.PSE, path=self.res_path)

    def plot_PSE(self):
        params = self.PSE["params"]
        PSE = self.PSE["results"]
        for res in self._plot_results:
            for pop in list(PSE[res].keys()):
                if res in ["Pearson", "Spearman"]:
                    name = "%s %s Corr" % (res, pop)
                else:
                    name = "%s %s" % (res, pop)
                if res == "rate":
                    name += " spikes/sec"
                filename = name.replace(" spikes/sec", "").replace(" ", "_") + ".png"
                try:
                    plot_result(params, PSE[res][pop], name,
                                os.path.join(self.folder_figs, filename))
                except:
                    continue

    def load_PSE(self):
        h5_file = h5py.File(self.res_path, 'r', libver='latest')
        try:
            for p in h5_file["params"].keys():
                self.PSE["params"][p] = h5_file["params"][p][()]
                self.PSE["n_params"]["n_"+p] = len(self.PSE["params"][p])
        except:
            pass
        try:
            self.PSE["results"] = read_dicts_from_h5file_recursively(h5_file["results"])
        except:
            pass

    def run(self):
        params = list(self.PSE["params"].keys())
        pse_params = OrderedDict()
        for i_s, s in enumerate(self.PSE["params"][params[0]]):
            pse_params[params[0]] = s
            for i_w, w in enumerate(self.PSE["params"][params[1]]):
                pse_params[params[1]] = w
                tic = time.time()
                self.print_PSE(pse_params)
                self.workflow.reset_workflow(**pse_params)
                self.workflow.configure()
                self.workflow.model_params = self.pse_to_model_params(pse_params)
                rates, corrs = self.workflow.run()
                self.results_to_PSE(i_s, i_w, rates, corrs)
                print_toc_message(tic)
        self.write_PSE()
        self.plot_PSE()


class PSE_1_tvb_mf_node_St_w(PSEWorkflowBase):
    name = "PSE_1_tvb_mf_node_St_w"

    def __init__(self, init_cond=0):
        self.PSE["params"]["Stimulus"] = np.arange(0.9, 1.31, 0.01)
        self.PSE["params"]["w+"] = np.arange(0.5, 1.6, 0.4)
        self.configure_PSE()
        self.PSE["results"]["rate"] = {"E": np.zeros(self.pse_shape)}
        self._plot_results = "rate"
        self.workflow = Workflow()
        self.workflow.tvb_model = ReducedWongWangExcIO
        if init_cond:
            self.workflow.tvb_init_cond = np.zeros((1, 4, 1, 1))
            self.workflow.tvb_init_cond[0, 0, 0, 0] = 1.0
            self.name = "high" + self.name
        else:
            self.name = "low" + self.name
        self.workflow.name = self.name
        self.workflow.force_dims = 10
        self.workflow.simulation_length = 2000.0
        self.workflow.transient = 1000.0
        self.workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
        self.workflow.tvb_sim_numba = True
        self.workflow.plotter = True
        self.workflow.writer = True
        super(PSE_1_tvb_mf_node_St_w, self).__init__()

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["TVB"] = {"G": np.array([0.0, ]),
                               "w": np.array([pse_params["w+"], ]),
                               "I_o": np.array([pse_params["Stimulus"] * 0.3, ])}
        return model_params

    def results_to_PSE(self, i1, i2, rates, corrs=None):
        self.PSE["results"]["rate"]["E"][i1, i2] = rates["TVB"][0].values.mean()

    # def plot_PSE(self):
    #     plot_result(self.PSE["params"], self.PSE["results"]["rate"]["E"], "Rate spikes/sec",
    #                 os.path.join(self.folder_figs, self.name + ".png"))


class PSE_2_tvb_mf_nodes_G_w(PSEWorkflowBase):
    name = "PSE_2_tvb_mf_nodes_G_w"

    def __init__(self, init_cond=0):
        self.PSE["params"]["G"] = np.arange(0.0, 405.0, 10.0)  #  100.0, 320.0, 20.0
        self.PSE["params"]["w+"] = np.arange(0.5, 1.6, 0.4)
        self.configure_PSE()
        Nreg_shape = (2, ) + self.pse_shape
        self.PSE["results"]["rate per node"] = {"E": np.zeros(Nreg_shape)}
        self.PSE["results"]["rate"] = {"E": np.zeros(self.pse_shape)}
        self.PSE["results"]["rate % diff"] = {"E": np.zeros(self.pse_shape)}
        for corr in ["Pearson", "Spearman"]:
            self.PSE["results"][corr] = OrderedDict()
            self.PSE["results"][corr]["EE"] = np.zeros(self.pse_shape)
        self._plot_results = ["rate", "rate % diff", "Pearson", "Spearman"]
        self.workflow = Workflow()
        self.workflow.tvb_model = ReducedWongWangExcIO
        if init_cond:
            self.workflow.tvb_init_cond = np.zeros((1, 4, 1, 1))
            self.workflow.tvb_init_cond[0, 0, 0, 0] = 1.0
            self.name = "high" + self.name
        else:
            self.name = "low" + self.name
        self.workflow.name = self.name
        self.workflow.force_dims = 2
        self.workflow.symmetric_connectome = True
        self.workflow.time_delays = False
        self.workflow.simulation_length = 2000.0
        self.workflow.transient = 1000.0
        self.workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
        self.workflow.tvb_sim_numba = True
        self.workflow.plotter = True
        self.workflow.writer = True
        super(PSE_2_tvb_mf_nodes_G_w, self).__init__()

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["TVB"] = {"G": np.array([pse_params["G"], ]),
                               "w": np.array([pse_params["w+"], ])}
        return model_params

    def results_to_PSE(self, i_g, i_w, rates, corrs):
        PSE = self.PSE["results"]
        PSE["rate per node"]["E"][:, i_g, i_w] = rates["TVB"][0].values.squeeze()
        PSE["rate"]["E"][i_g, i_w] = PSE["rate per node"]["E"][:, i_g, i_w].mean()
        PSE["rate % diff"]["E"][i_g, i_w] = \
            100 * np.abs(np.diff(PSE["rate per node"]["E"][:, i_g, i_w]) /
                         PSE["rate"]["E"][i_g, i_w])
        for corr in ["Pearson", "Spearman"]:
            PSE[corr]["EE"][i_g, i_w] = corrs["TVB"][corr][0, 0, 0, 1].values.item()


class PSE_3_tvb_mf_nodes_G_w(PSE_2_tvb_mf_nodes_G_w):
    name = "PSE_3_tvb_mf_nodes_G_w"

    def __init__(self, init_cond=0):
        self.PSE["params"]["G"] = np.arange(0.0, 405.0, 10.0)  #  100.0, 320.0, 20.0
        self.PSE["params"]["w+"] = np.arange(0.5, 1.6, 0.4)
        self.configure_PSE()
        Nreg = 3
        Nreg_shape = (3, ) + self.pse_shape
        self.PSE["results"]["rate per node"] = {"E": np.zeros(Nreg_shape)}
        self.PSE["results"]["rate"] = {"E": np.zeros(self.pse_shape)}
        self.PSE["results"]["rate % zscore"] = {"E": np.zeros(self.pse_shape)}
        for corr in ["Pearson", "Spearman"]:
            self.PSE["results"][corr] = OrderedDict()
            self.PSE["results"][corr]["EE"] = np.zeros(Nreg_shape)
            self.PSE["results"][corr]["FC-SC"] = np.zeros(self.pse_shape)
        self._plot_results = ["rate", "rate % zscore", "Pearson", "Spearman"]
        self.workflow = Workflow()
        self.workflow.tvb_model = ReducedWongWangExcIO
        if init_cond:
            self.workflow.tvb_init_cond = np.zeros((1, 4, 1, 1))
            self.workflow.tvb_init_cond[0, 0, 0, 0] = 1.0
            self.name = "high" + self.name
        else:
            self.name = "low" + self.name
        self.workflow.name = self.name
        self._SC = [0.1, 0.5, 0.9]
        connectivity, self._SC, self._SCsize, self._triu_inds = symmetric_connectivity(self._SC, 3)
        self.workflow.force_dims = Nreg
        self.workflow.connectivity_path = "None"
        self.workflow.connectivity = connectivity
        self.workflow.symmetric_connectome = True
        self.workflow.time_delays = False
        self.workflow.simulation_length = 2000.0
        self.workflow.transient = 1000.0
        self.workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
        self.workflow.tvb_sim_numba = True
        self.workflow.plotter = True
        self.workflow.writer = True
        PSEWorkflowBase.__init__(self)

    def results_to_PSE(self, i_g, i_w, rates, corrs):
        PSE = self.PSE["results"]
        PSE["rate per node"]["E"][:, i_g, i_w] = rates["TVB"][0].values.squeeze()
        PSE["rate"]["E"][i_g, i_w] = PSE["rate per node"]["E"][:, i_g, i_w].mean()
        PSE["rate % zscore"]["E"][i_g, i_w] = \
            100 * np.abs(np.std(PSE["rate per node"]["E"][:, i_g, i_w]) /
                         PSE["rate"]["E"][i_g, i_w])
        for corr in ["Pearson", "Spearman"]:
            PSE[corr]["EE"][:, i_g, i_w] = corrs["TVB"][corr][0, 0].values[self._triu_inds[0], self._triu_inds[1]]
            PSE[corr]["FC-SC"][i_g, i_w] = \
                (np.dot(PSE[corr]["EE"][:, i_g, i_w], self._SC)) / \
                (np.sqrt(np.sum(PSE[corr]["EE"][:, i_g, i_w] ** 2)) * self._SCsize)


class PSE_1_tvb_spiking_node_St_w(PSEWorkflowBase):
    name = "PSE_1_tvb_spiking_node_St_w"

    def __init__(self, init_cond=0):
        self.PSE["params"]["Stimulus"] = np.arange(0.9, 2.1, 0.5)
        self.PSE["params"]["w+"] = np.arange(0.5, 1.6, 0.4)
        self.configure_PSE()
        self.PSE["results"]["rate"] = {"E": np.zeros(self.pse_shape),
                                       "I": np.zeros(self.pse_shape)}
        self._plot_results = ["rate"]
        self.workflow = Workflow()
        self.workflow.tvb_model = SpikingWongWangExcIOInhI
        self.workflow.tvb_init_cond = np.zeros((1, self.workflow.tvb_model._nvar, 1, 1))
        if init_cond:
            self.workflow.tvb_init_cond[:, 5, :, :] = -70.0  # Setting V_m to V_rest
            self.name = "high" + self.name
        else:
            self.workflow.tvb_init_cond[:, 5, :, :] = -50.1  # Setting V_m to V_rest
            self.name = "low" + self.name
        self.workflow.name = self.name
        self.workflow.populations_sizes = [100, 100]
        self.workflow.force_dims = 1
        self.workflow.dt = 0.025
        self.workflow.simulation_length = 2000.0
        self.stim_time_length = int(np.ceil(self.workflow.simulation_length / self.workflow.dt))
        self.workflow.transient = 1000.0
        self.workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
        self.workflow.tvb_sim_numba = False
        self.workflow.plotter = False
        self.workflow.writer = False
        super(PSE_1_tvb_spiking_node_St_w, self).__init__()

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["TVB"] = {"G": np.array([0.0, ]),
                               "w_EE": np.array([pse_params["w+"], ])}
        return model_params

    def results_to_PSE(self, i1, i2, rates, corrs=None):
        self.PSE["results"]["rate"]["E"][i1, i2] = rates["TVB"][0].values.item()
        self.PSE["results"]["rate"]["I"][i1, i2] = rates["TVB"][1].values.item()

    def run(self):
        params = list(self.PSE["params"].keys())
        pse_params = OrderedDict()
        for i_s, s in enumerate(self.PSE["params"][params[0]]):
            pse_params[params[0]] = s
            stb = SpikeStimulusBuilder(targets=["spikes_ext"], rate=s*2180.0, A=None,
                                       # A=None for uncorrelated spikes
                                       number_of_regions=1, number_of_neurons=200, target_regions=None,
                                       t_start=0.0, dt=self.workflow.dt, time_length=self.stim_time_length,
                                       sparse=False)
            stimulus = stb.build()
            for i_w, w in enumerate(self.PSE["params"][params[1]]):
                pse_params[params[1]] = w
                tic = time.time()
                self.print_PSE(pse_params)
                self.workflow.reset_workflow(**pse_params)
                self.workflow.tvb_spike_stimulus = stimulus
                self.workflow.configure()
                self.workflow.model_params = self.pse_to_model_params(pse_params)
                rates, corrs = self.workflow.run()
                self.results_to_PSE(i_s, i_w, rates, corrs)
                print_toc_message(tic)
        self.write_PSE()
        self.plot_PSE()


class PSE_2_tvb_spiking_nodes_G_w(PSEWorkflowBase):
    name = "PSE_2_tvb_spiking_nodes_G_w"

    def __init__(self, init_cond=0):
        self.PSE["params"]["G"] = np.arange(0.0, 405.0, 10.0)
        self.PSE["params"]["w+"] = np.arange(0.5, 1.6, 0.4)
        self.configure_PSE()
        Nreg_shape = (2,) + self.pse_shape
        self.PSE["results"]["rate per node"] = {"E": np.zeros(Nreg_shape),
                                                "I": np.zeros(Nreg_shape)}
        self.PSE["results"]["rate"] = {"E": np.zeros(self.pse_shape),
                                       "I": np.zeros(self.pse_shape)}
        self.PSE["results"]["rate % diff"] = {"E": np.zeros(self.pse_shape),
                                              "I": np.zeros(self.pse_shape)}
        for corr in ["Pearson", "Spearman"]:
            self.PSE["results"][corr] = OrderedDict()
            self.PSE["results"][corr]["EE"] = np.zeros(self.pse_shape)
        self._plot_results = ["rate", "rate % diff", "Pearson", "Spearman"]
        self.workflow = Workflow()
        self.workflow.tvb_model = SpikingWongWangExcIOInhI
        self.workflow.tvb_init_cond = np.zeros((1, self.workflow.tvb_model._nvar, 1, 1))
        if init_cond:
            self.workflow.tvb_init_cond[:, 5, :, :] = -70.0  # Setting V_m to V_rest
            self.name = "high" + self.name
        else:
            self.workflow.tvb_init_cond[:, 5, :, :] = -50.1  # Setting V_m to V_rest
            self.name = "low" + self.name
        self.workflow.name = self.name
        self.workflow.populations_sizes = [100, 100]
        self.workflow.force_dims = 1
        self.workflow.dt = 0.025
        self.workflow.simulation_length = 2000.0
        self.stim_time_length = int(np.ceil(self.workflow.simulation_length / self.workflow.dt))
        stb = SpikeStimulusBuilder(targets=["spikes_ext"], rate=2180.0, A=None,
                                   # A=None for uncorrelated spikes
                                   number_of_regions=1, number_of_neurons=200, target_regions=None,
                                   t_start=0.0, dt=self.workflow.dt, time_length=self.stim_time_length,
                                   sparse=False)
        self.stimulus = stb.build()
        self.workflow.transient = 1000.0
        self.workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
        self.workflow.tvb_sim_numba = False
        self.workflow.plotter = True
        self.workflow.writer = True
        super(PSE_2_tvb_spiking_nodes_G_w, self).__init__()

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["TVB"] = {"G": np.array([pse_params["G"], ]),
                               "w_EE": np.array([pse_params["w+"], ])}
        return model_params

    def run(self):
        params = list(self.PSE["params"].keys())
        pse_params = OrderedDict()
        for i_s, s in enumerate(self.PSE["params"][params[0]]):
            pse_params[params[0]] = s
            for i_w, w in enumerate(self.PSE["params"][params[1]]):
                pse_params[params[1]] = w
                tic = time.time()
                self.print_PSE(pse_params)
                self.workflow.reset_workflow(**pse_params)
                self.workflow.tvb_spike_stimulus = deepcopy(self.stimulus)
                self.workflow.configure()
                self.workflow.model_params = self.pse_to_model_params(pse_params)
                rates, corrs = self.workflow.run()
                self.results_to_PSE(i_s, i_w, rates, corrs)
                print_toc_message(tic)
        self.write_PSE()
        self.plot_PSE()

    def results_to_PSE(self, i_g, i_w, rates, corrs):
        PSE = self.PSE["results"]
        PSE["rate per node"]["E"][:, i_g, i_w] = rates["TVB"][0].values.squeeze()
        PSE["rate per node"]["I"][:, i_g, i_w] = rates["TVB"][1].values.squeeze()
        for pop in ["E", "I"]:
            PSE["rate"][pop][i_g, i_w] = PSE["rate per node"][pop][:, i_g, i_w].mean()
            PSE["rate % diff"][i_g, i_w] = \
                100 * np.abs(np.diff(PSE["per"][pop][:, i_g, i_w]) / PSE["rate"][pop][i_g, i_w])
        for corr in ["Pearson", "Spearman"]:
            PSE[corr]["EE"][i_g, i_w] = corrs["TVB"][corr][0, 0, 0, 1].values.item()


class PSE_3_tvb_spiking_nodes_G_w(PSE_2_tvb_spiking_nodes_G_w):
    name = "PSE_3_tvb_spiking_nodes_G_w"

    def __init__(self, init_cond=0):
        self.PSE["params"]["G"] = np.arange(0.0, 405.0, 10.0)
        self.PSE["params"]["w+"] = np.arange(0.5, 1.6, 0.4)
        self.configure_PSE()
        Nreg = 3
        Nreg_shape = (Nreg,) + self.pse_shape
        self.PSE["results"]["rate per node"] = {"E": np.zeros(Nreg_shape),
                                                "I": np.zeros(Nreg_shape)}
        self.PSE["results"]["rate"] = {"E": np.zeros(self.pse_shape),
                                       "I": np.zeros(self.pse_shape)}
        self.PSE["results"]["rate % zscore"] = {"E": np.zeros(self.pse_shape),
                                                "I": np.zeros(self.pse_shape)}
        for corr in ["Pearson", "Spearman"]:
            self.PSE["results"][corr] = OrderedDict()
            self.PSE["results"][corr]["EE"] = np.zeros(Nreg_shape)
            self.PSE["results"][corr]["FC-SC"] = np.zeros(self.pse_shape)
        self._plot_results = ["rate", "rate % zscore", "Pearson", "Spearman"]
        self.workflow = Workflow()
        self.workflow.tvb_model = SpikingWongWangExcIOInhI
        self.workflow.tvb_init_cond = np.zeros((1, self.workflow.tvb_model._nvar, 1, 1))
        if init_cond:
            self.workflow.tvb_init_cond[:, 5, :, :] = -70.0  # Setting V_m to V_rest
            self.name = "high" + self.name
        else:
            self.workflow.tvb_init_cond[:, 5, :, :] = -50.1  # Setting V_m to V_rest
            self.name = "low" + self.name
        self.workflow.name = self.name
        self.workflow.populations_sizes = [100, 100]
        self._SC = [0.1, 0.5, 0.9]
        connectivity, self._SC, self._SCsize, self._triu_inds = symmetric_connectivity(self._SC, Nreg)
        self.workflow.force_dims = Nreg
        self.workflow.connectivity_path = "None"
        self.workflow.connectivity = connectivity
        self.workflow.dt = 0.025
        self.workflow.simulation_length = 2000.0
        self.stim_time_length = int(np.ceil(self.workflow.simulation_length / self.workflow.dt))
        stb = SpikeStimulusBuilder(targets=["spikes_ext"], rate=2180.0, A=None,
                                   # A=None for uncorrelated spikes
                                   number_of_regions=1, number_of_neurons=200, target_regions=None,
                                   t_start=0.0, dt=self.workflow.dt, time_length=self.stim_time_length,
                                   sparse=False)
        self.stimulus = stb.build()
        self.workflow.transient = 1000.0
        self.workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
        self.workflow.tvb_sim_numba = False
        self.workflow.plotter = True
        self.workflow.writer = True
        super(PSE_2_tvb_spiking_nodes_G_w, self).__init__()

    def results_to_PSE(self, i_g, i_w, rates, corrs):
        PSE = self.PSE["results"]
        PSE["rate per node"]["E"][:, i_g, i_w] = rates["TVB"][0].values.squeeze()
        PSE["rate per node"]["I"][:, i_g, i_w] = rates["TVB"][1].values.squeeze()
        for pop in ["E", "I"]:
            PSE["rate"][pop][i_g, i_w] = PSE["rate per node"][pop][:, i_g, i_w].mean()
            PSE["rate % zscore"][i_g, i_w] = \
                100 * np.abs(np.std(PSE["per"][pop][:, i_g, i_w]) / PSE["rate"][pop][i_g, i_w])
        for corr in ["Pearson", "Spearman"]:
            PSE[corr]["EE"][:, i_g, i_w] = corrs["TVB"][corr][0, 0].values[self._triu_inds[0], self._triu_inds[1]]
            PSE[corr]["FC-SC"][i_g, i_w] = \
                (np.dot(PSE[corr]["EE"][:, i_g, i_w], self._SC)) / \
                (np.sqrt(np.sum(PSE[corr]["EE"][:, i_g, i_w] ** 2)) * self._SCsize)
