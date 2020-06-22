# -*- coding: utf-8 -*-
import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from tvb_multiscale.examples.paperwork.pse_workflow_base import symmetric_connectivity, PSEWorkflowBase

from tvb_multiscale.examples.paperwork.workflow import Workflow
from tvb_utils.utils import print_toc_message

from tvb.simulator.models.spiking_wong_wang_exc_io_inh_i import SpikingWongWangExcIOInhI
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


def prepare_spike_stimulus(rate, dt, time_length, number_of_regions, number_of_neurons):
    from tvb_elephant.spike_stimulus_builder import SpikeStimulusBuilder
    stb = SpikeStimulusBuilder(targets=["spikes_ext"], rate=rate, A=None,
                              # A=None for uncorrelated spikes
                              number_of_regions=number_of_regions, number_of_neurons=number_of_neurons,
                              target_regions=None, t_start=0.0, dt=dt, time_length=time_length,
                              sparse=False)

    stimulus = stb.build()
    return stimulus


class PSEWorkflowSpiking(PSEWorkflowBase):
    name = "PSEWorkflowSpiking"
    _plot_results = ["rate", "Pearson", "Spearman", "spike train"]
    _corr_results = ["Pearson", "Spearman", "spike train"]

    def __init__(self, w=None, branch="low", fast=False):
        self.branch = branch
        if self.branch == "low":
            self.noise = 0.01
        else:
            self.noise = 0.0
        self.name = branch + self.name
        self.PSE["params"]["w+"] = np.arange(0.5, 1.6, 0.4)
        self.workflow = Workflow()
        self.workflow.tvb_model = SpikingWongWangExcIOInhI
        self.workflow.name = self.name
        self.workflow.populations_sizes = [100, 100]
        self.n_neurons = np.sum(self.workflow.populations_sizes).item()
        self.workflow.symmetric_connectome = True
        self.workflow.time_delays = False
        self.workflow.tvb_init_cond = np.zeros((1, self.workflow.tvb_model._nvar, 1, 1))
        self.workflow.tvb_init_cond[:, 5, :, :] = -70.0  # Setting V_m to V_rest
        self.workflow.dt = 0.025
        self.workflow.simulation_length = 2000.0
        self.workflow.transient = 1000.0
        if fast:
            self.workflow.simulation_length /= 10
            self.workflow.transient /= 10
        self.stim_time_length = int(np.ceil(self.workflow.simulation_length / self.workflow.dt))
        self.workflow.tvb_noise_strength = self.noise  # 0.0001 / 2
        self.workflow.tvb_sim_numba = False
        self.workflow.plotter = True
        self.workflow.writer = True
        self.workflow.write_time_series = False
        self.workflow.print_progression_message = self.print_progression_message
        kwargs = {}
        if w is not None:
            w = ensure_list(w)
            if len(w) == 1:
                kwargs = {"w+": w[0]}
        self.configure_paths(**kwargs)

    def configure_PSE(self, w=None):
        if w is None:
            w = np.sort(np.arange(1.0, 1.7, 0.1).tolist() + [1.55])
        else:
            w = np.sort(ensure_list(w))
        self.PSE["params"]["w+"] = w
        super(PSEWorkflowSpiking, self).configure_PSE()


class PSE_1_TVBspikingNodeStW(PSEWorkflowSpiking):
    name = "PSE_1_TVBspikingNodeStW"

    def __init__(self, w=None, branch="low", fast=False):
        super(PSE_1_TVBspikingNodeStW, self).__init__(w, branch, fast)
        if fast:
            step = 1.0
        else:
            step = 0.1
        self.PSE["params"]["Stimulus"] = np.arange(0.9, 5.1, step)
        self.configure_PSE(w)
        self.PSE["results"]["rate"] = {"E": np.zeros(self.pse_shape),
                                       "I": np.zeros(self.pse_shape)}
        self._plot_results = ["rate"]
        self.workflow.force_dims = 1

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["TVB"] = {"G": np.array([0.0, ]),
                               "w_EE": np.array([pse_params["w+"], ])}
        return model_params

    def results_to_PSE(self, i1, i2, rates, corrs=None):
        self.PSE["results"]["rate"]["E"][i1, i2] = rates["TVB"][0].values.item()
        self.PSE["results"]["rate"]["I"][i1, i2] = rates["TVB"][1].values.item()

    def prepare_stimulus(self, s):
        return prepare_spike_stimulus(s * 2180.0, self.workflow.dt, self.stim_time_length, 1, self.n_neurons)

    def run(self):
        params = list(self.PSE["params"].keys())
        pse_params = OrderedDict()
        for i_s, s in enumerate(self.PSE["params"][params[0]]):
            pse_params[params[0]] = s
            stimulus = self.prepare_stimulus(s)
            for i_w, w in enumerate(self.PSE["params"][params[1]]):
                pse_params[params[1]] = w
                tic = time.time()
                self.print_PSE(pse_params)
                self.workflow.reset(pse_params)
                self.workflow.tvb_spike_stimulus = stimulus
                self.workflow.configure()
                self.workflow.model_params = self.pse_to_model_params(pse_params)
                rates, corrs = self.workflow.run()
                self.results_to_PSE(i_s, i_w, rates, corrs)
                print_toc_message(tic)
        self.write_PSE()
        self.plot_PSE()


class PSE_2_TVBspikingNodesGW(PSEWorkflowSpiking):
    name = "PSE_2_TVBspikingNodesGW"

    def __init__(self, w=None, branch="low", fast=False):
        super(PSE_2_TVBspikingNodesGW, self).__init__(w, branch, fast)
        if fast:
            step = 100.0
        else:
            step = 10.0
        self.PSE["params"]["G"] = np.arange(0.0, 305.0, step)
        self.configure_PSE(w)
        Nreg = 2
        Nreg_shape = (Nreg,) + self.pse_shape
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
        self.workflow.force_dims = Nreg
        self.stimulus = \
            prepare_spike_stimulus(2180.0, self.workflow.dt, self.stim_time_length, Nreg, self.n_neurons)

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
                self.workflow.reset(pse_params)
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
            PSE["rate % diff"][pop][i_g, i_w] = \
                100 * np.abs(np.diff(PSE["rate per node"][pop][:, i_g, i_w]) / PSE["rate"][pop][i_g, i_w])
        for corr in self._corr_results:
            PSE[corr]["EE"][i_g, i_w] = corrs["TVB"][corr][0, 0, 0, 1].values.item()


class PSE_3_TVBspikingNodesGW(PSE_2_TVBspikingNodesGW):
    name = "PSE_3_TVBspikingNodesGW"

    def __init__(self, w=None, branch="low", fast=False):
        super(PSE_2_TVBspikingNodesGW, self).__init__(w, branch, fast)
        if fast:
            step = 100.0
        else:
            step = 10.0
        self.PSE["params"]["G"] = np.arange(0.0, 205.0, step)
        self.configure_PSE(w)
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
        self._SC = [0.1, 0.5, 0.9]
        connectivity, self._SC, self._SCsize, self._triu_inds = symmetric_connectivity(self._SC, Nreg)
        self.workflow.force_dims = Nreg
        self.workflow.connectivity_path = "None"
        self.workflow.connectivity = connectivity
        self.stimulus = \
            prepare_spike_stimulus(2180.0, self.workflow.dt, self.stim_time_length, Nreg, self.n_neurons)

    def results_to_PSE(self, i_g, i_w, rates, corrs):
        PSE = self.PSE["results"]
        PSE["rate per node"]["E"][:, i_g, i_w] = rates["TVB"][0].values.squeeze()
        PSE["rate per node"]["I"][:, i_g, i_w] = rates["TVB"][1].values.squeeze()
        for pop in ["E", "I"]:
            PSE["rate"][pop][i_g, i_w] = PSE["rate per node"][pop][:, i_g, i_w].mean()
            PSE["rate % zscore"][pop][i_g, i_w] = \
                100 * np.abs(np.std(PSE["rate per node"][pop][:, i_g, i_w]) / PSE["rate"][pop][i_g, i_w])
        for corr in self._corr_results:
            PSE[corr]["EE"][:, i_g, i_w] = corrs["TVB"][corr][0, 0].values[self._triu_inds[0], self._triu_inds[1]]
            PSE[corr]["FC-SC"][i_g, i_w] = \
                (np.dot(PSE[corr]["EE"][:, i_g, i_w], self._SC)) / \
                (np.sqrt(np.sum(PSE[corr]["EE"][:, i_g, i_w] ** 2)) * self._SCsize)
