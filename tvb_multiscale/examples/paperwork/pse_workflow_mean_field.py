from collections import OrderedDict

import numpy as np
from tvb_multiscale.config import Config
from tvb_multiscale.examples.paperwork.pse_workflow_base import PSEWorkflowBase, symmetric_connectivity
from tvb_multiscale.examples.paperwork.workflow import Workflow
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list

from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO


class PSEWorkflowMF(PSEWorkflowBase):
    name = "PSEWorkflowMF"
    _plot_results = ["rate", "Pearson", "Spearman"]
    _corr_results = ["Pearson", "Spearman"]

    def __init__(self, w=None, branch="low", fast=False, output_base=None):
        self.config = Config(separate_by_run=False, output_base=output_base)
        self.branch = branch
        self.workflow = Workflow()
        self.workflow.config = self.config
        self.workflow.tvb_model = ReducedWongWangExcIO
        if self.branch == "high":
            self.workflow.tvb_init_cond = np.zeros((1, 4, 1, 1))
            self.workflow.tvb_init_cond[0, 0, 0, 0] = 1.0
        self.name = self.branch + self.name
        self.workflow.name = self.name
        self.workflow.symmetric_connectome = True
        self.workflow.time_delays = False
        self.workflow.simulation_length = 2000.0
        self.workflow.transient = 1000.0
        if fast:
            self.workflow.simulation_length /= 10
            self.workflow.transient /= 10
        self.workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
        self.workflow.tvb_sim_numba = True
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
            w = np.arange(0.8, 1.6, 0.1)
        else:
            w = np.sort(ensure_list(w))
        self.PSE["params"]["w+"] = w
        super(PSEWorkflowMF, self).configure_PSE()


class PSE_1_TVBmfNodeStW(PSEWorkflowMF):
    name = "PSE_1_tvb_mf_node_St_w"

    def __init__(self, w=None, branch="low", fast=False, output_base=None):
        super(PSE_1_TVBmfNodeStW, self).__init__(w, branch, fast, output_base)
        if fast:
            step = 0.1
        else:
            step = 0.01
        self.PSE["params"]["Stimulus"] = np.arange(0.9, 1.31, step)
        self.configure_PSE(w)
        self.PSE["results"]["rate"] = {"E": np.zeros(self.pse_shape)}
        self._plot_results = ["rate"]
        self.workflow.force_dims = 10

    def pse_to_model_params(self, pse_params):
        model_params = self.workflow.model_params
        model_params["TVB"] = {"G": np.array([0.0, ]),
                               "w": np.array([pse_params["w+"], ]),
                               "I_o": np.array([pse_params["Stimulus"] * 0.3, ])}
        return model_params

    def results_to_PSE(self, i1, i2, rates, corrs=None):
        self.PSE["results"]["rate"]["E"][i1, i2] = rates["TVB"][0].values.mean()


class PSE_2_TVBmfNodesGW(PSEWorkflowMF):
    name = "PSE_2_tvb_mf_nodes_G_w"

    def __init__(self, w=None, branch="low", fast=False, output_base=None):
        super(PSE_2_TVBmfNodesGW, self).__init__(w, branch, fast, output_base)
        if fast:
            step = 100.0
        else:
            step = 10.0
        self.PSE["params"]["G"] = np.arange(0.0, 305.0, step)
        self.configure_PSE(w)
        Nreg = 2
        Nreg_shape = (Nreg, ) + self.pse_shape
        self.PSE["results"]["rate per node"] = {"E": np.zeros(Nreg_shape)}
        self.PSE["results"]["rate"] = {"E": np.zeros(self.pse_shape)}
        self.PSE["results"]["rate % diff"] = {"E": np.zeros(self.pse_shape)}
        for corr in ["Pearson", "Spearman"]:
            self.PSE["results"][corr] = OrderedDict()
            self.PSE["results"][corr]["EE"] = np.zeros(self.pse_shape)
        self._plot_results = ["rate", "rate % diff", "Pearson", "Spearman"]
        self.workflow.force_dims = Nreg

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
        for corr in self._corr_results:
            PSE[corr]["EE"][i_g, i_w] = corrs["TVB"][corr][0, 0, 0, 1].values.item()


class PSE_3_TVBmfNodesGW(PSE_2_TVBmfNodesGW):
    name = "PSE_3_tvb_mf_nodes_G_w"

    def __init__(self, w=None, branch="low", fast=False, output_base=None):
        super(PSE_2_TVBmfNodesGW, self).__init__(w, branch, fast, output_base)
        if fast:
            step = 100.0
        else:
            step = 10.0
        self.PSE["params"]["G"] = np.arange(0.0, 305.0, step)  #  100.0, 320.0, 20.0
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
        self._SC = [0.1, 0.5, 0.9]
        connectivity, self._SC, self._SCsize, self._triu_inds = symmetric_connectivity(self._SC, 3)
        self.workflow.force_dims = Nreg
        self.workflow.connectivity_path = "None"
        self.workflow.connectivity = connectivity

    def results_to_PSE(self, i_g, i_w, rates, corrs):
        PSE = self.PSE["results"]
        PSE["rate per node"]["E"][:, i_g, i_w] = rates["TVB"][0].values.squeeze()
        PSE["rate"]["E"][i_g, i_w] = PSE["rate per node"]["E"][:, i_g, i_w].mean()
        PSE["rate % zscore"]["E"][i_g, i_w] = \
            100 * np.abs(np.std(PSE["rate per node"]["E"][:, i_g, i_w]) /
                         PSE["rate"]["E"][i_g, i_w])
        for corr in self._corr_results:
            PSE[corr]["EE"][:, i_g, i_w] = corrs["TVB"][corr][0, 0].values[self._triu_inds[0], self._triu_inds[1]]
            PSE[corr]["FC-SC"][i_g, i_w] = \
                (np.dot(PSE[corr]["EE"][:, i_g, i_w], self._SC)) / \
                (np.sqrt(np.sum(PSE[corr]["EE"][:, i_g, i_w] ** 2)) * self._SCsize)
