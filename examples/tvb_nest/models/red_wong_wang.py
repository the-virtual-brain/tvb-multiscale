# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb_multiscale.core.tvb.cosimulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb_multiscale.core.interfaces.base.transformers.models.red_wong_wang import \
    ElephantSpikesRateRedWongWangExc, ElephantSpikesRateRedWongWangInh
from tvb_multiscale.tvb_nest.nest_models.builders.models.ww_deco import WWDeco2013Builder, WWDeco2014Builder

from examples.tvb_nest.example import main_example

from tvb.simulator.integrators import HeunStochastic
from tvb.simulator.noise import Additive


def red_wong_wang_excio_example(**kwargs):

    spiking_proxy_inds = kwargs.pop("spiking_proxy_inds", [0, 1])

    model_params = kwargs.pop("model_params", {})

    populations_order = kwargs.pop("populations_order", 100)

    model = kwargs.pop("model", "RATE").upper()
    if model == "CURRENT":
        voi = "S"
        transformer_params = {"scale_factor": np.array([model_params.get("J_N", 1000.0*np.array([0.2609, ]))])}
        proxy_params = {}
    else:
        voi = "R"
        transformer_params = {"scale_factor": populations_order*np.array([1.0])}
        proxy_params = {"number_of_neurons": 1}

    tvb_to_spikeNet_interfaces = [{"model": model,  # "RATE",  # "SPIKES", "CURRENT"
                                   "voi": voi,  # "S"
                                   "populations": "E",
                                   "transformer_params": transformer_params,
                                   "proxy_params": proxy_params
                                   }]

    tvb_to_spikeNet_transformer = kwargs.pop("tvb_to_spikeNet_transformer",
                                             kwargs.pop("tvb_to_spikeNet_transformer_model", None))
    if tvb_to_spikeNet_transformer:
        tvb_to_spikeNet_interfaces[0]["transformer_model"] = tvb_to_spikeNet_transformer
    tvb_to_spikeNet_interfaces[0]["transformer_params"].update(kwargs.pop("tvb_to_spikeNet_transformer_params", {}))
    tvb_to_spikeNet_proxy = kwargs.pop("tvb_to_spikeNet_proxy", kwargs.pop("tvb_to_spikeNet_proxy_model", None))
    if tvb_to_spikeNet_proxy:
        tvb_to_spikeNet_interfaces[0]["proxy_model"] = tvb_to_spikeNet_proxy
    tvb_to_spikeNet_interfaces[0]["proxy_params"].update(kwargs.pop("tvb_to_spikeNet_proxy_params", {}))

    spikeNet_to_tvb_transformer = kwargs.pop("spikeNet_to_tvb_transformer",
                                         kwargs.pop("spikeNet_to_tvb_transformer_model",
                                                    ElephantSpikesRateRedWongWangExc))
    spikeNet_to_tvb_interfaces = [{"voi": ("S", "R"), "populations": "E",
                               "transformer": spikeNet_to_tvb_transformer,
                               "transformer_params": {"scale_factor": np.array([1.0]) / populations_order,
                                                      "integrator":
                                                          HeunStochastic(dt=0.1,
                                                                         noise=Additive(
                                                                             nsig=np.array([[1e-3], [0.0]]))),
                                                      "state": np.zeros((2, len(spiking_proxy_inds))),
                                                      "tau_s": model_params.get("tau_s",
                                                                                np.array([100.0, ])),
                                                      "tau_r": np.array([10.0, ]),
                                                      "gamma": model_params.get("gamma",
                                                                                np.array([0.641 / 1000, ]))}
                               }]
    spikeNet_to_tvb_interfaces[0]["transformer_params"].update(kwargs.pop("spikeNet_to_tvb_transformer_params", {}))

    return main_example(ReducedWongWangExcIO, WWDeco2013Builder, spiking_proxy_inds,
                        populations_order=populations_order, model_params=model_params,
                        tvb_to_spikeNet_interfaces=tvb_to_spikeNet_interfaces,
                        spikeNet_to_tvb_interfaces=spikeNet_to_tvb_interfaces,
                        **kwargs)


def red_wong_wang_excio_inhi_example(**kwargs):

    spiking_proxy_inds = kwargs.pop("spiking_proxy_inds", [0, 1])

    model_params = kwargs.pop("model_params", {})

    populations_order = kwargs.pop("populations_order", 100)
    N_E = 1.6 * populations_order
    N_I = 0.4 * populations_order

    model = kwargs.pop("model", "RATE").upper()
    if model == "CURRENT":
        voi_e = "S_e"
        voi_i = "S_e"
        tvb_spikeNet_transformer_params_e = {"scale_factor": np.array([model_params.get("J_N",
                                                                                        1000.0*np.array([0.2609, ]))])}
        tvb_spikeNet_transformer_params_i = {"scale_factor": np.array([model_params.get("J_i",
                                                                                        1000.0 * np.array([1.0, ]))])}
        tvb_to_spikeNet_proxy_params = {}
    else:
        voi_e = "R_e"
        voi_i = "R_i"
        tvb_spikeNet_transformer_params_e = {"scale_factor": np.array([N_E])}
        tvb_spikeNet_transformer_params_i = {"scale_factor": np.array([N_I])}
        tvb_to_spikeNet_proxy_params = {"number_of_neurons": 1}
    tvb_to_spikeNet_transformer = kwargs.pop("tvb_to_spikeNet_transformer",
                                             kwargs.pop("tvb_to_spikeNet_transformer_model", None))
    tvb_to_spikeNet_transformer_params = kwargs.pop("tvb_to_spikeNet_transformer_params", {})
    tvb_spikeNet_transformer_params_e.update(tvb_to_spikeNet_transformer_params)
    tvb_spikeNet_transformer_params_i.update(tvb_to_spikeNet_transformer_params)

    tvb_to_spikeNet_proxy = kwargs.pop("tvb_to_spikeNet_proxy", kwargs.pop("tvb_to_spikeNet_proxy_model", None))
    tvb_to_spikeNet_proxy_params.update(kwargs.pop("tvb_to_spikeNet_proxy_params", {}))

    tvb_to_spikeNet_interfaces = []
    for ii, (_pop, voi, pop, transformer_params, proxy_params) in \
            enumerate(zip(["e", "i"],
                          [voi_e, voi_i],
                          ["E", "I"],
                          [tvb_spikeNet_transformer_params_e, tvb_spikeNet_transformer_params_i],
                          [tvb_to_spikeNet_proxy_params, tvb_to_spikeNet_proxy_params])):
        tvb_to_spikeNet_interfaces.append(
            {"model": model, "voi": voi, "populations": pop,
             "transformer_params": transformer_params, "proxy_params": proxy_params})
        if tvb_to_spikeNet_transformer:
            tvb_to_spikeNet_interfaces[ii]["transformer_model"] = tvb_to_spikeNet_transformer
        tvb_to_spikeNet_interfaces[ii]["transformer_params"].update(
            kwargs.pop("tvb_to_spikeNet_transformer_params_%s" % _pop, {}))
        if tvb_to_spikeNet_proxy:
            tvb_to_spikeNet_interfaces[ii]["proxy_model"] = tvb_to_spikeNet_proxy
        tvb_to_spikeNet_interfaces[ii]["proxy_params"].update(
            kwargs.pop("tvb_to_spikeNet_proxy_params_%s" % _pop, {}))

    spikeNet_to_tvb_transformer_e = kwargs.pop("spikeNet_to_tvb_transformer_e",
                                         kwargs.pop("spikeNet_to_tvb_transformer_model_e",
                                                    ElephantSpikesRateRedWongWangExc))
    spikeNet_to_tvb_transformer_i = kwargs.pop("spikeNet_to_tvb_transformer_i",
                                           kwargs.pop("spikeNet_to_tvb_transformer_model_i",
                                                      ElephantSpikesRateRedWongWangInh))
    spikeNet_to_tvb_transformer_params = kwargs.pop("spikeNet_to_tvb_transformer_params", {})
    spikeNet_to_tvb_interfaces = []
    for ii, (_pop, voi, pop, transformer, scale, tau_s, tau_r, gamma) in \
            enumerate(zip(["e", "i"], [("S_e", "R_e"), ("S_i", "R_i")], ["E", "I"],
                          [spikeNet_to_tvb_transformer_e, spikeNet_to_tvb_transformer_i],
                          [np.array([1.0]) / N_E, np.array([1.0]) / N_I],
                          [model_params.get("tau_e", np.array([100.0, ])),
                           model_params.get("tau_i", np.array([10.0, ]))],
                          [np.array([10.0, ]), np.array([10.0, ])],
                          [model_params.get("gamma_e", np.array([0.641 / 1000])),
                           model_params.get("gamma_i", np.array([1.0 / 1000]))])):

        spikeNet_to_tvb_interfaces.append({
            "voi": voi, "populations": pop,
            "transformer": transformer,
            "transformer_params":
                {"scale_factor": scale,
                 "integrator": HeunStochastic(dt=0.1, noise=Additive(nsig=np.array([[1e-3], [0.0]]))),
                 "state": np.zeros((2, len(spiking_proxy_inds))),
                 "tau_s": tau_s, "tau_r": tau_r, "gamma": gamma}
        })
        spikeNet_to_tvb_interfaces[ii]["transformer_params"].update(spikeNet_to_tvb_transformer_params)
        spikeNet_to_tvb_interfaces[ii]["transformer_params"].update(
            kwargs.pop("spikeNet_to_tvb_transformer_params_%s" % _pop, {}))

    return main_example(ReducedWongWangExcIOInhI, WWDeco2014Builder, spiking_proxy_inds,
                        model_params=model_params, populations_order=populations_order,
                        tvb_to_spikeNet_interfaces=tvb_to_spikeNet_interfaces,
                        spikeNet_to_tvb_interfaces=spikeNet_to_tvb_interfaces,
                        **kwargs)


if __name__ == "__main__":
    import sys
    if sys.argv[-1] == "1":
        red_wong_wang_excio_inhi_example(model="CURRENT")
    else:
        red_wong_wang_excio_example(model="CURRENT")
