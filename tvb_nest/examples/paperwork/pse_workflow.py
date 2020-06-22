# -*- coding: utf-8 -*-
import time
from collections import OrderedDict

import numpy as np

from tvb_nest.config import Config
from tvb_nest.examples.paperwork.workflow import Workflow
from tvb_multiscale.examples.paperwork.pse_workflow_base import symmetric_connectivity, PSEWorkflowBase
from tvb_utils.utils import print_toc_message

from tvb.contrib.scripts.utils.data_structures_utils import ensure_list


class PSENESTWorkflowBase(PSEWorkflowBase):
    name = "PSENESTWorkflow"

    def __init__(self, w=None, branch="low", fast=False):
        super(PSENESTWorkflowBase, self).__init__()
        self.branch = branch
        self.name = self.branch + self.name
        self.config = Config(separate_by_run=False)
        self._plot_results = ["rate", "Pearson", "Spearman", "spike train"]
        self._corr_results = ["Pearson", "Spearman", "spike train"]
        self.workflow = Workflow()
        self.workflow.name = self.name
        self.workflow.tvb_to_nest_interface = None
        self.workflow.symmetric_connectome = True
        self.workflow.time_delays = False
        self.workflow.dt = 0.1
        self.workflow.simulation_length = 2000.0
        self.workflow.transient = 1000.0
        if fast:
            self.workflow.simulation_length /= 10
            self.workflow.transient /= 10
        self.workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
        self.workflow.nest_stimulus_rate = 2018.0
        self.workflow.nest_stimulus_times = [0.1]
        if self.branch == "high":
            self.workflow.nest_stimulus_rate *= np.array([2.0, 1.0])
            # self.workflow.simulation_length += self.workflow.transient
            # self.workflow.transient *= 2
            self.workflow.nest_stimulus_times += [self.workflow.transient/4]
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
        super(PSENESTWorkflowBase, self).configure_PSE()


class PSE_1_NESTnodeStW(PSENESTWorkflowBase):
    name = "PSE_1_NESTnodeStW"

    def __init__(self, w=None, branch="low", fast=False):
        super(PSE_1_NESTnodeStW, self).__init__(w, branch, fast)
        if fast:
            step = 2.0
        else:
            step = 0.1
        self.PSE["params"]["Stimulus"] = np.arange(0.9, 5.1, step)
        self.configure_PSE(w)
        self.PSE["results"]["rate"] = {"E": np.zeros(self.pse_shape),
                                       "I": np.zeros(self.pse_shape)}
        self._plot_results = ["rate"]
        self.workflow.force_dims = [34]
        self.workflow.nest_nodes_ids = [0]  # the indices of fine scale regions modeled with NEST

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["NEST"]["E"] = {"w_E": pse_params["w+"].item()}
        self.workflow.nest_stimulus_rate = 2018.0 * pse_params["Stimulus"].item()
        if self.branch == "high":
            self.workflow.nest_stimulus_rate *= np.array([2.0, 1.0])
        return model_params

    def results_to_PSE(self, i1, i2, rates, corrs=None):
        self.PSE["results"]["rate"]["E"][i1, i2] = rates["NEST"][0].values.item()
        self.PSE["results"]["rate"]["I"][i1, i2] = rates["NEST"][1].values.item()


class PSE_2_NESTnodesGW(PSENESTWorkflowBase):
    name = "PSE_2_NESTnodesGW"

    def __init__(self, w=None, branch="low", fast=False):
        super(PSE_2_NESTnodesGW, self).__init__(w, branch, fast)
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
        for corr in self._corr_results:
            self.PSE["results"][corr] = OrderedDict()
            self.PSE["results"][corr]["EE"] = np.zeros(self.pse_shape)
        self._plot_results = ["rate", "rate % diff", "Pearson", "Spearman", "spike train"]
        self.workflow.force_dims = [34, 35]
        self.workflow.nest_nodes_ids = [0, 1]  # the indices of fine scale regions modeled with NEST

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["TVB"] = {"G": np.array([pse_params["G"], ])}
        model_params["NEST"]["E"] = {"w_E": pse_params["w+"]}
        return model_params

    def results_to_PSE(self, i_g, i_w, rates, corrs):
        PSE = self.PSE["results"]
        PSE["rate per node"]["E"][:, i_g, i_w] = rates["NEST"][0].values.squeeze()
        PSE["rate per node"]["I"][:, i_g, i_w] = rates["NEST"][1].values.squeeze()
        for pop in ["E", "I"]:
            PSE["rate"][pop][i_g, i_w] = PSE["rate per node"][pop][:, i_g, i_w].mean()
            PSE["rate % diff"][i_g, i_w] = \
                100 * np.abs(np.diff(PSE["rate per node"][pop][:, i_g, i_w]) / PSE["rate"][pop][i_g, i_w])
        for corr in self._corr_results:
            corr_name = corr.replace(" ", "_")
            PSE[corr]["EE"][i_g, i_w] = corrs["NEST"][corr_name][0, 0, 0, 1].values.item()


class PSE_3_NESTnodesGW(PSE_2_NESTnodesGW):
    name = "PSE_3_NESTnodesGW"

    def __init__(self, w=None, branch="low", fast=False):
        super(PSE_2_NESTnodesGW, self).__init__(w, branch, fast)
        if fast:
            step = 100.0
        else:
            step = 10.0
        self.PSE["params"]["G"] = np.arange(0.0, 305.0, step)
        self.configure_PSE(w)
        Nreg = 3
        Nreg_shape = (Nreg,) + self.pse_shape
        self.PSE["results"]["rate per node"] = {"E": np.zeros(Nreg_shape),
                                                "I": np.zeros(Nreg_shape)}
        self.PSE["results"]["rate"] = {"E": np.zeros(self.pse_shape),
                                       "I": np.zeros(self.pse_shape)}
        self.PSE["results"]["rate % zscore"] = {"E": np.zeros(self.pse_shape),
                                                "I": np.zeros(self.pse_shape)}
        for corr in self._corr_results:
            self.PSE["results"][corr] = OrderedDict()
            self.PSE["results"][corr]["EE"] = np.zeros(Nreg_shape)
            self.PSE["results"][corr]["FC-SC"] = np.zeros(self.pse_shape)
        self._plot_results = ["rate", "rate % zscore", "Pearson", "Spearman", "spike train"]
        self._SC = [0.1, 0.5, 0.9]
        connectivity, self._SC, self._SCsize, self._triu_inds = symmetric_connectivity(self._SC, 3)
        self.workflow.force_dims = Nreg
        self.workflow.connectivity_path = "None"
        self.workflow.connectivity = connectivity
        self.workflow.force_dims = Nreg
        self.workflow.nest_nodes_ids = [0, 1, 2]  # the indices of fine scale regions modeled with NEST

    def results_to_PSE(self, i_g, i_w, rates, corrs):
        PSE = self.PSE["results"]
        PSE["rate per node"]["E"][:, i_g, i_w] = rates["NEST"][0].values.squeeze()
        PSE["rate per node"]["I"][:, i_g, i_w] = rates["NEST"][1].values.squeeze()
        for pop in ["E", "I"]:
            PSE["rate"][pop][i_g, i_w] = PSE["rate per node"][pop][:, i_g, i_w].mean()
            PSE["rate % zscore"][i_g, i_w] = \
                100 * np.abs(np.std(PSE["rate per node"][pop][:, i_g, i_w]) / PSE["rate"][pop][i_g, i_w])
        for corr in self._corr_results:
            corr_name = corr.replace(" ", "_")
            PSE[corr]["EE"][:, i_g, i_w] = corrs["NEST"][corr_name][0, 0].values[self._triu_inds[0], self._triu_inds[1]]
            PSE[corr]["FC-SC"][i_g, i_w] = \
                (np.dot(PSE[corr]["EE"][:, i_g, i_w], self._SC)) / \
                (np.sqrt(np.sum(PSE[corr]["EE"][:, i_g, i_w] ** 2)) * self._SCsize)
