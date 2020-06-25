# -*- coding: utf-8 -*-
import os
import time
from copy import deepcopy

import numpy as np
from xarray import DataArray

from matplotlib import pyplot as pl
from matplotlib import cm

from tvb.basic.profile import TvbProfile

TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_nest.config import CONFIGURED
from tvb.contrib.scripts.utils.log_error_utils import print_toc_message
from tvb.contrib.scripts.utils.file_utils import safe_makedirs


def run_PSE(pse_class, todo="run", **kwargs):
    pse_workflow = pse_class(**kwargs)
    if todo == "run":
        pse_workflow.run()
    else:
        if todo.find("1D") > -1:
            pse_workflow.load_PSE_1D(**kwargs)
        elif todo.find("2D") > -1:
            pse_workflow.load_PSE_2D(**kwargs)
        else:
            pse_workflow.load_PSE()
        if todo.find("plot") > -1:
            pse_workflow.plot_PSE()
    print(pse_workflow.PSE)
    return pse_workflow.PSE


def single_nest_PSE(todo="run", w=None, branch="low", fast=False, output_base=None):
    from tvb_nest.examples.paperwork.pse_workflow import PSE_1_NESTnodeStW
    return run_PSE(PSE_1_NESTnodeStW, todo, w=w, branch=branch, fast=fast, output_base=output_base)


def two_nest_nodes_PSE(todo="run", w=None, branch="low", fast=False, output_base=None):
    from tvb_nest.examples.paperwork.pse_workflow import PSE_2_NESTnodesGW
    return run_PSE(PSE_2_NESTnodesGW, todo, w=w, branch=branch, fast=fast, output_base=output_base)


def three_nest_nodes_PSE(todo="run", w=None, branch="low", fast=False, output_base=None):
    from tvb_nest.examples.paperwork.pse_workflow import PSE_3_NESTnodesGW
    return run_PSE(PSE_3_NESTnodesGW, todo, w=w, branch=branch, fast=fast, output_base=output_base)


def plot_result(PSE_params, result, name, path):
    dims = ["branch"] + list(PSE_params.keys())
    coords = dict(PSE_params)
    coords["branch"] = ["low", "high"]
    arr = DataArray(data=np.array(result), dims=dims, coords=coords, name=name)
    fig = pl.figure(figsize=(10, 10))
    arr = arr.stack({"%sbranch" % dims[2]: [dims[2], "branch"]})
    lines = arr.plot.line(x=arr.dims[0], hue=arr.dims[1], add_legend=False)
    cmap = cm.get_cmap('jet')
    nw = np.maximum(1.0, np.ceil(arr.shape[1]/2) - 1.0)
    i_line = 0
    for i_color, w in enumerate(coords[dims[2]]):
        color = cmap(i_color/nw)
        for style, branch in zip(["--", "-"], coords["branch"]):
            lines[i_line].set_color(color)
            lines[i_line].set_linestyle(style)
            lines[i_line].set_label("%s, %s" % (w, branch))
            i_line += 1
    fig.canvas.draw()
    pl.legend()
    pl.savefig(path)


def plot_results(PSElow, PSEhigh, name, results, pops, names, folder_figs=None):
    if folder_figs is None:
        folder_figs = CONFIGURED.out.FOLDER_RES.replace("res", name)
    safe_makedirs(folder_figs)
    for res, nam in zip(results, names):
        for pop in pops:
            try:
                this_name = "%s %s" % (pop, nam)
                path = os.path.join(folder_figs, this_name+".png").replace(" (spikes/sec)", "").replace(" ", "_")
                plot_result(PSElow["params"],
                            [PSElow["results"][res][pop],
                             PSEhigh["results"][res][pop]],
                            this_name, path)
            except:
                pass


if __name__ == "__main__":

    tic = time.time()

    output_base ='/Users/dionperd/Software/TVB/tvb-multiscale/tvb_nest/examples/paperwork/outputs/PSE_2_NESTnodesW'
    # try:
    #     PSElow = deepcopy(single_nest_PSE(todo="plot2D", output_base=output_base))
    #     PSEhigh = deepcopy(single_nest_PSE(todo="plot2D", branch="high", output_base=output_base))
    #     name = "PSE_1_NESTnodeStW"
    #     results = ["rate"]
    #     pops = ["E", "I"]
    #     names = ["Rate (spikes/sec)"]
    #     plot_results(PSElow, PSEhigh, name, results, pops, names)
    # except:
    #     pass

    try:
        PSElow = deepcopy(two_nest_nodes_PSE(todo="plot1D", branch="low", output_base=output_base))
        PSEhigh = deepcopy(two_nest_nodes_PSE(todo="plot1D", branch="high", output_base=output_base))
        name = "PSE_2_NESTnodesW"
        results = ["rate", "rate % diff", "Pearson", "Spearman", "spike train"]
        pops = ["E", "I", "EE"]
        names = ["Rate (spikes/sec)", "rate % diff", "Pearson Corr", "Spearman Corr", "Spike train Corr"]
        plot_results(PSElow, PSEhigh, name, results, pops, names)

    except:
        pass
    #
    # try:
    #     PSElow = deepcopy(three_nest_nodes_PSE(todo="plot1D", branch="low", output_base=output_base))
    #     PSEhigh = deepcopy(three_nest_nodes_PSE(todo="plot1D", branch="high", output_base=output_base))
    #     name = "PSE_3_NESTnodesW"
    #     results = ["rate", "rate % zscore", "Pearson", "Spearman", "spike train"]
    #     pops = ["E", "I", "EE", "FC-SC"]
    #     names = ["Rate (spikes/sec)", "rate % zscore", "Pearson Corr", "Spearman Corr", "Spike train Corr"]
    #     plot_results(PSElow, PSEhigh, name, results, pops, names)
    # except:
    #     pass

    print_toc_message(tic)
