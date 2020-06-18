from collections import OrderedDict

import numpy as np
from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb_multiscale.examples.paperwork.pse_workflow_base import PSEWorkflowBase, symmetric_connectivity
from tvb_multiscale.examples.paperwork.workflow import Workflow


class PSEWorkflowMF(PSEWorkflowBase):
    name = "PSEWorkflowMF"
    _plot_results = ["rate", "Pearson", "Spearman"]
    _corr_results = ["Pearson", "Spearman"]

    def __init__(self, init_cond=0):
        self.workflow = Workflow()
        self.workflow.tvb_model = ReducedWongWangExcIO
        if init_cond:
            self.workflow.tvb_init_cond = np.zeros((1, 4, 1, 1))
            self.workflow.tvb_init_cond[0, 0, 0, 0] = 1.0
            self.name = "high" + self.name
        else:
            self.name = "low" + self.name
        self.workflow.name = self.name
        self.workflow.symmetric_connectome = True
        self.workflow.time_delays = False
        self.workflow.simulation_length = 2000.0
        self.workflow.transient = 1000.0
        self.workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
        self.workflow.tvb_sim_numba = True
        self.workflow.plotter = True
        self.workflow.writer = True
        self.workflow.write_time_series = False
        self.workflow.print_progression_message = self.print_progression_message
        self.configure_paths()

    def configure_PSE(self):
        self.PSE["params"]["w+"] = np.arange(0.5, 1.6, 0.4)
        super(PSEWorkflowMF, self).configure_PSE()


class PSE_1_TVBmfNodeStW(PSEWorkflowMF):
    name = "PSE_1_tvb_mf_node_St_w"

    def __init__(self, init_cond=0):
        super(PSE_1_TVBmfNodeStW, self).__init__(init_cond)
        self.PSE["params"]["Stimulus"] = np.arange(0.9, 1.31, 0.01)
        self.configure_PSE()
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

    def __init__(self, init_cond=0):
        super(PSE_2_TVBmfNodesGW, self).__init__(init_cond)
        self.PSE["params"]["G"] = np.arange(0.0, 405.0, 10.0)  #  100.0, 320.0, 20.0
        self.configure_PSE()
        Nreg_shape = (2, ) + self.pse_shape
        self.PSE["results"]["rate per node"] = {"E": np.zeros(Nreg_shape)}
        self.PSE["results"]["rate"] = {"E": np.zeros(self.pse_shape)}
        self.PSE["results"]["rate % diff"] = {"E": np.zeros(self.pse_shape)}
        for corr in ["Pearson", "Spearman"]:
            self.PSE["results"][corr] = OrderedDict()
            self.PSE["results"][corr]["EE"] = np.zeros(self.pse_shape)
        self._plot_results = ["rate", "rate % diff", "Pearson", "Spearman"]
        self.workflow.force_dims = 2

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

    def __init__(self, init_cond=0):
        super(PSE_2_TVBmfNodesGW, self).__init__(init_cond)
        self.PSE["params"]["G"] = np.arange(0.0, 405.0, 10.0)  #  100.0, 320.0, 20.0
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
