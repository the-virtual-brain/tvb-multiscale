# -*- coding: utf-8 -*-

import numpy as np

from tvb.datatypes.connectivity import Connectivity


def serialize_tvb_cosimulator(cosimulator):

    d = \
        {"integrator.dt": float(cosimulator.integrator.dt),
         "connectivity.number_of_regions": int(cosimulator.connectivity.number_of_regions),
         "connectivity.number_of_connections": int(cosimulator.connectivity.number_of_connections),
         "connectivity.undirected": bool(cosimulator.connectivity.undirected),
         "model": str(cosimulator.model.__class__.__name__),
         "model.nvar": int(cosimulator.model.nvar),
         "model.nintvar": int(cosimulator.model.nintvar) if cosimulator.model.nintvar is not None else 0,
         "model.state_variables": list(cosimulator.model.state_variables),
         "model.cvar": np.copy(cosimulator.model.cvar),
         "monitor.period": float(cosimulator.monitors[0].period),
    }
    d["min_delay"] = float(getattr(cosimulator, "_min_delay", d["integrator.dt"]))
    d["min_idelay"] = int(getattr(cosimulator, "_min_idelay", 1))
    d["synchronization_time"] = float(getattr(cosimulator, "synchronization_time", d["integrator.dt"]))
    d["synchronization_n_step"] = int(getattr(cosimulator, "synchronization_n_step", 1))
    if hasattr(cosimulator.integrator, "noise"):
        d["integrator.noise.nsig"] = cosimulator.integrator.noise.nsig

    excluded_attrs = ("state_variables", "state_variable_range", "state_variable_boundaries", "variables_of_interest",
                       "nvar", "nintvar", "cvar", "noise", "psi_table", "nerf_table", "gid")
    for attr in type(cosimulator.model).declarative_attrs:
        if attr in excluded_attrs:
            continue
        d["model.%s" % attr] = np.copy(getattr(cosimulator.model, attr))

    excluded_attrs = ("undirected", "number_of_regions", "number_of_connections", 
                       "parent_connectivity", "saved_selection", "gid")
    for attr in type(cosimulator.connectivity).declarative_attrs:
        if attr in excluded_attrs:
            continue
        d["connectivity.%s" % attr] = np.copy(getattr(cosimulator.connectivity, attr))
        
    for attr in cosimulator.coupling._own_declarative_attrs:
        d["coupling.%s" % attr] = np.copy(getattr(cosimulator.coupling, attr))

    return d


def serial_tvb_simulator_to_connectivity(tvb_sim_dict):
    kwargs = {}
    excluded_attrs = ("undirected", "number_of_regions", "number_of_connections",
                       "parent_connectivity", "saved_selection", "gid")
    for attr in Connectivity.declarative_attrs:
        if attr in excluded_attrs:
            continue
        val = np.copy(tvb_sim_dict.get("connectivity.%s" % attr, None))
        if val is None:
            continue
        kwargs[attr] = val
    return Connectivity(**kwargs)
