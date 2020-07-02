# -*- coding: utf-8 -*-
import time

import numpy as np

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_nest.examples.paperwork.workflow import Workflow
from tvb.contrib.scripts.utils.log_error_utils import print_toc_message


def main_workflow():
    workflow = Workflow()
    workflow.force_dims = 2  # [33, 34, 35]
    workflow.symmetric_connectome = True
    # Select the regions for the fine scale modeling with NEST spiking networks
    workflow.nest_nodes_ids = [1]  # the indices of fine scale regions modeled with NEST
    # from tvb_multiscale.examples.paperwork.pse_workflow_base import symmetric_connectivity
    # workflow.connectivity = symmetric_connectivity([0.5, 0.5, 0.5], 3)[0]
    # workflow.connectivity_path = "None"
    workflow.time_delays = False
    # In this example, we model parahippocampal cortices (34.left and 35.right) with NESTnest_nodes_ids
    workflow.nest_to_tvb_interface = True
    workflow.plotter = True
    workflow.writer = True
    # workflow.tvb_sim_numba = True
    workflow.simulation_length = 2000.0
    workflow.transient = 1000.0
    workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
    workflow.tvb_init_cond = np.zeros((1, workflow.tvb_model._nvar, 1, 1))
    workflow.tvb_init_cond[0, 0, 0, 0] = 1.0
    workflow.nest_stimulus_rate *= np.array([2.0, 1.0])
    workflow.nest_stimulus_times = [0.1, 1000.0]
    workflow.configure()
    model_params = workflow.model_params
    model_params["NEST"]["E"] = {"w_E": 1.3, "w_I": 1.0}
    model_params["TVB"] = {'G': np.array([70.0, ]), 'w': np.array([0.9, ])}
    workflow.model_params = model_params
    rates, corrs = workflow.run()
    return rates, corrs


def nest_workflow():
    workflow = Workflow()
    # Select the regions for the fine scale modeling with NEST spiking networks
    workflow.force_dims = [34, 35]
    workflow.nest_nodes_ids = [0, 1]  # the indices of fine scale regions modeled with NEST
    workflow.tvb_to_nest_interface = None
    # In this example, we model parahippocampal cortices (34.left and 35.right) with NESTnest_nodes_ids
    workflow.plotter = True
    workflow.writer = True
    workflow.simulation_length = 100.0
    workflow.configure()
    rates, corrs = workflow.run()
    return rates, corrs


def single_nest_workflow():
    workflow = Workflow()  # **{"w_EE": 0.0, "w_EI": 0.0, "w_IE": 0.0} , "w_II": 0.0
    # Select the regions for the fine scale modeling with NEST spiking networks
    workflow.force_dims = [34]
    workflow.nest_nodes_ids = [0]  # the indices of fine scale regions modeled with NEST
    workflow.tvb_to_nest_interface = None
    # In this example, we model parahippocampal cortices (34.left and 35.right) with NESTnest_nodes_ids
    workflow.plotter = True
    workflow.writer = True
    workflow.simulation_length = 1100.0
    workflow.transient = 100.0
    workflow.configure()
    model_params = workflow.model_params
    model_params["NEST"]["E"] = {"w_E": 1.55, "w_I": 1.0}
    workflow.model_params = model_params
    # workflow.model_params["NEST"]["I"] = {"w_E": 0.0}  # , "w_I": 0.0
    rates, corrs = workflow.run()
    return rates, corrs


def tvb_workflow():
    workflow = Workflow()
    # Select the regions for the fine scale modeling with NEST spiking networks
    workflow.nest_nodes_ids = []  # the indices of fine scale regions modeled with NEST
    workflow.tvb_to_nest_interface = None
    # In this example, we model parahippocampal cortices (34.left and 35.right) with NESTnest_nodes_ids
    workflow.plotter = True
    workflow.writer = True
    workflow.simulation_length = 100.0
    workflow.configure()
    rates, corrs = workflow.run()
    return rates, corrs


def benchmark_workflow():
    workflow = Workflow()
    workflow.nest_populations_order = 200
    # In this example, we model parahippocampal cortices (34.left and 35.right)
    # Select the regions for the fine scale modeling with NEST spiking networks
    workflow.nest_nodes_ids = [34, 35]  #  the indices of fine scale regions modeled with NEST
    workflow.tvb_to_nest_interface = "rate"
    workflow.nest_to_tvb_interface = True
    workflow.plotter = False
    workflow.writer = False
    workflow.dt = 0.1
    workflow.simulation_length = 1000.0
    workflow.tvb_noise_strength = 0.0001 / 2
    workflow.tvb_sim_numba = True
    workflow.transient = 0.0
    workflow.configure()
    rates = workflow.run()
    return rates


def plot_benchmark():
    import json
    import os
    import numpy as np
    from matplotlib import pyplot
    dir = os.getcwd()
    with open(os.path.join(dir, "tvb_only_numba.json"), 'r') as fp:
        tvb_only = json.load(fp)
    results = {"total": {}, "NEST": {}, "TVBtoNEST": {}, "NESTtoTVB": {}}
    for im, mode in enumerate(["rate"]):  # , "dc", "Ie"
        dirname = "numba_%s" % mode
        for key, val in results.items():
            results[key][mode] = {}
        for ir, n_regions in enumerate([1, 2]):
            regs_lbl = "%dregion" % n_regions
            if n_regions == 1:
                s = ""
            else:
                s = "s"
            for key, val in results.items():
                results[key][mode][regs_lbl] = []
            for io, n_order in enumerate([10, 50, 100, 200]):
                filename = "%dregion%s_%dpoporder.json" % (n_regions, s, n_order)
                path = os.path.join(dir, dirname, filename)
                with open(path, 'r') as fp:
                    data = json.load(fp)
                for key, val in results.items():
                    results[key][mode][regs_lbl].append(data[key])
    print(results)

    xticks = 4*np.array([10, 50, 100, 200])
    pyplot.rcParams.update({'font.size': 12})
    pyplot.figure(figsize=(10, 5))
    pyplot.plot(0, tvb_only["total"], color="k", linestyle="",
                marker="*", markersize=10, markerfacecolor="k", markeredgecolor="k", label="Total - TVB only")
    for im, (mode, color) in enumerate(zip(["rate"], ["k"])):  # , "dc", "Ie"  "r", "b", "g"
        # pyplot.plot(xticks, results["total"][mode]["1region"], color=color, alpha=0.5, linestyle="-", linewidth=1,
        #             marker="o", markersize=10, markerfacecolor=color, markeredgecolor=color,
        #             label="Total - %s - 1 region" % mode)
        pyplot.plot(xticks, results["total"][mode]["2region"], color=color, alpha=0.5, linestyle="-", linewidth=1,
                    marker="o", markersize=10, markerfacecolor=color, markeredgecolor=color,
                    label="Total")
        # pyplot.plot(xticks, results["NEST"][mode]["1region"], color=color, alpha=0.5, linestyle="-", linewidth=1,
        #             marker="s", markersize=10, markerfacecolor=color, markeredgecolor=color,
        #             label="NEST - %s - 1 region" % mode)
        pyplot.plot(xticks, results["NEST"][mode]["2region"], color="r", alpha=0.5, linestyle="-", linewidth=1,
                    marker="o", markersize=10, markerfacecolor="r", markeredgecolor="r",
                    label="NEST integration")
        # pyplot.plot(xticks, results["TVBtoNEST"][mode]["1region"], color=color, alpha=0.5, linestyle="-", linewidth=1,
        #             marker="d", markersize=10, markerfacecolor=color, markeredgecolor=color,
        #             label="TVB to NEST - %s - 1 region" % mode)
        pyplot.plot(xticks, results["TVBtoNEST"][mode]["2region"], color="b", alpha=0.5, linestyle="-", linewidth=1,
                    marker="o", markersize=10, markerfacecolor="b", markeredgecolor="b",
                    label="TVB to NEST")
        # pyplot.plot(xticks, results["NESTtoTVB"][mode]["1region"], color=color, alpha=0.5, linestyle="-", linewidth=1,
        #             marker="x", markersize=10, markerfacecolor=color, markeredgecolor=color,
        #             label="NEST to TVB - %s - 1 region" % mode)
        pyplot.plot(xticks, results["NESTtoTVB"][mode]["2region"], color="g", alpha=0.5, linestyle="-", linewidth=1,
                    marker="o", markersize=10, markerfacecolor="g", markeredgecolor="g",
                    label="NEST to TVB")
        pyplot.xticks([0] + xticks.tolist())
        pyplot.xlabel("Total number of neurons in 2 region nodes modelled in NEST")
        pyplot.ylabel("Computation time (sec) of 1 sec of simulation")
        pyplot.legend()
    pyplot.savefig(os.path.join(dir, "benchmarking.png"))


if __name__ == "__main__":

    tic = time.time()

    print(main_workflow())
    # print(nest_workflow())
    # print(tvb_workflow())
    # print(single_nest_workflow())
    # single_nest_PSE()
    # print(benchmark_workflow())
    # plot_benchmark()
    # print(two_symmetric_nest_PSE())

    print_toc_message(tic)
