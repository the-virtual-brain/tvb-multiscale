# -*- coding: utf-8 -*-

from copy import deepcopy
import dill

import numpy as np


def serialize_tvb_simulator(input_simulator):

    simulator = deepcopy(input_simulator)
    simulator.configure()

    d = \
        {"integrator.dt": float(simulator.integrator.dt),
         "connectivity.number_of_regions": int(simulator.connectivity.number_of_regions),
         "connectivity.region_labels": np.copy(simulator.connectivity.region_labels),
         "connectivity.weights": np.copy(simulator.connectivity.weights),
         "connectivity.delays": np.copy(simulator.connectivity.delays),
         "coupling.a": np.copy(simulator.coupling.a),
         "model": str(simulator.model.__class__.__name__),
         "model.nvar": int(simulator.model.nvar),
         "model.nintvar": int(simulator.model.nintvar),
         "model.state_variables": list(simulator.model.state_variables),
         "model.cvar": np.copy(simulator.model.cvar),
         "monitor.period": float(simulator.monitors[0].period)
    }

    excluded_params = ("state_variables", "state_variable_range", "state_variable_boundaries", "variables_of_interest",
                       "nvar", "nintvar", "cvar", "noise", "psi_table", "nerf_table", "gid")
    for param in type(simulator.model).declarative_attrs:
        if param in excluded_params:
            continue
        d["model.%s" % param] = np.copy(getattr(simulator.model, param))

    return d


def dump_serial_tvb_simulator(serial_tvb_sim, filepath):
    with open(filepath, "wb") as f:
        dill.dump(serial_tvb_sim, f)


def load_serial_tvb_simulator(filepath):
    with open(filepath, "wb") as f:
        serial_tvb_sim = dill.load(f)
    return serial_tvb_sim

