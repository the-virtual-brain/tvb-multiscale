# -*- coding: utf-8 -*-

import numpy as np


def serialize_tvb_cosimulator(cosimulator):

    d = \
        {"integrator.dt": float(cosimulator.integrator.dt),
         "connectivity.number_of_regions": int(cosimulator.connectivity.number_of_regions),
         "connectivity.region_labels": np.copy(cosimulator.connectivity.region_labels),
         "connectivity.weights": np.copy(cosimulator.connectivity.weights),
         "connectivity.delays": np.copy(cosimulator.connectivity.delays),
         "model": str(cosimulator.model.__class__.__name__),
         "model.nvar": int(cosimulator.model.nvar),
         "model.nintvar": int(cosimulator.model.nintvar) if cosimulator.model.nintvar is not None else 0,
         "model.state_variables": list(cosimulator.model.state_variables),
         "model.cvar": np.copy(cosimulator.model.cvar),
         "monitor.period": float(cosimulator.monitors[0].period),
    }
    d["synchronization_time"] = float(getattr(cosimulator, "synchronization_time", d["integrator.dt"]))
    d["synchronization_n_step"] = float(getattr(cosimulator, "synchronization_n_step", 1))
    if hasattr(cosimulator.integrator, "noise"):
        d["integrator.noise.nsig"] = cosimulator.integrator.noise.nsig

    excluded_params = ("state_variables", "state_variable_range", "state_variable_boundaries", "variables_of_interest",
                       "nvar", "nintvar", "cvar", "noise", "psi_table", "nerf_table", "gid")
    for param in type(cosimulator.model).declarative_attrs:
        if param in excluded_params:
            continue
        d["model.%s" % param] = np.copy(getattr(cosimulator.model, param))

    for param in cosimulator.coupling._own_declarative_attrs:
        d["coupling.%s" % param] = np.copy(getattr(cosimulator.coupling, param))

    return d
