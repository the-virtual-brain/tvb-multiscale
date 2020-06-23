# -*- coding: utf-8 -*-
import sys
import time
import os
from collections import OrderedDict
import numpy as np
from tvb_nest.config import Config
from tvb_nest.examples.paperwork.workflow import Workflow
from tvb_multiscale.examples.paperwork.pse_workflow_base import print_PSE
from tvb_multiscale.io.h5_writer import H5Writer
from tvb.contrib.scripts.utils.log_error_utils import print_toc_message
from tvb.contrib.scripts.utils.file_utils import safe_makedirs


args = sys.argv

tic = time.time()

try:
    fast = args[5] == "fast"
except:
    fast = False

print("fast=%s" % str(fast))

# Run workflow for a single pair of G, w+ values.

config = Config(separate_by_run=False, output_base=args[4])
writer = H5Writer()
PSE = {"params": OrderedDict(), "results": OrderedDict()}
_plot_results = ["rate"]
# Configure PSE
w = float(args[1])
St = float(args[2])
branch = args[3]
name = branch + "PSE_1_NESTnodeStW"
print("name=%s" % str(name))
PSE["params"]["Stimulus"] = St
PSE["params"]["w+"] = w
pse_shape = (1, 1)
Nreg = 1
Nreg_shape = (Nreg,) + pse_shape
PSE["results"]["rate"] = {"E": np.zeros(pse_shape), "I": np.zeros(pse_shape)}
# Configure paths
folder_res = config.out._folder_res.replace("res", name)
safe_makedirs(folder_res)
res_path = os.path.join(folder_res, name)
for key, val in PSE["params"].items():
    addstring = "_%s%g" % (key, val)
    res_path = res_path + addstring
res_path = res_path + ".h5"
# Create and configure workflow
workflow = Workflow()
workflow.config = config
workflow.name = name
workflow.tvb_to_nest_interface = None
workflow.symmetric_connectome = True
workflow.time_delays = False
workflow.dt = 0.1
workflow.simulation_length = 2000.0
workflow.transient = 1000.0
if fast:
    workflow.simulation_length /= 10
    workflow.transient /= 10
workflow.tvb_noise_strength = 0.0  # 0.0001 / 2
workflow.nest_stimulus_rate = 2018.0
workflow.nest_stimulus_times = [0.1]
if branch == "high":
    workflow.nest_stimulus_rate *= np.array([2.0, 1.0])
    # workflow.simulation_length += workflow.transient
    # workflow.transient *= 2
    workflow.nest_stimulus_times += [workflow.transient / 4]
workflow.tvb_sim_numba = False
workflow.plotter = True
workflow.writer = True
workflow.write_time_series = False
workflow.print_progression_message = False
workflow.force_dims = [34]
workflow.nest_nodes_ids = [0]  # the indices of fine scale regions modeled with NEST
# Run workflow
print_PSE(PSE["params"])
workflow.reset(PSE["params"])
workflow.nest_stimulus_rate = 2018.0 * PSE["params"]["Stimulus"]
workflow.configure()
if branch == "high":
    workflow.nest_stimulus_rate *= np.array([2.0, 1.0])
workflow.model_params["NEST"]["E"] = {"w_E": PSE["params"]["w+"]}
rates, corrs = workflow.run()
del workflow
i_g = 0
i_w = 0
for pop in ["E", "I"]:
    PSE["results"]["rate"][pop][i_g, i_w] = rates["NEST"][0].values.item()
writer.write_dictionary(PSE, path=res_path)

print_toc_message(tic)
