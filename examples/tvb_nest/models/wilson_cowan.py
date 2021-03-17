# -*- coding: utf-8 -*-

import numpy as np

from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan
from tvb_multiscale.tvb_nest.spikeNet_models.builders.models.wilson_cowan import WilsonCowanBuilder

from examples.tvb_nest.example import main_example


def wilson_cowan_example(**kwargs):

    spiking_proxy_inds = kwargs.pop("spiking_proxy_inds", [0, 1])

    # -----------------------------------Wilson Cowan oscillatory regime------------------------------------------------

    model_params = {
        "r_e": np.array([0.0]),
        "r_i": np.array([0.0]),
        "k_e": np.array([1.0]),
        "k_i": np.array([1.0]),
        "tau_e": np.array([10.0]),
        "tau_i": np.array([10.0]),
        "c_ee": np.array([10.0]),
        "c_ei": np.array([6.0]),
        "c_ie": np.array([10.0]),
        "c_ii": np.array([1.0]),
        "alpha_e": np.array([1.2]),
        "alpha_i": np.array([2.0]),
        "a_e": np.array([1.0]),
        "a_i": np.array([1.0]),
        "b_e": np.array([0.0]),
        "b_i": np.array([0.0]),
        "c_e": np.array([1.0]),
        "c_i": np.array([1.0]),
        "theta_e": np.array([2.0]),
        "theta_i": np.array([3.5]),
        "P": np.array([0.5]),
        "Q": np.array([0.0])
    }

    model_params.update(kwargs.pop("model_params", {}))

    populations_order = kwargs.pop("populations_order", 100)

    model = kwargs.pop("model", "RATE")
    if model == "SPIKES":
        transformer_params = {"scale_factor": np.array([100.0]),
                              "number_of_neurons": np.array([populations_order]).astype("i"),
                              "interaction": "multiple",
                              "correlation_factor": np.array([0.1])
                              }
        proxy_params = {"number_of_neurons": populations_order}
    else:
        transformer_params = {"scale_factor": populations_order*np.array([100.0])}
        proxy_params = {"number_of_neurons": 1}

    tvb_to_spikeNet_interfaces = [{"model": model, "voi": "E", "populations": "E",
                                   "transformer_params": transformer_params,
                                   "proxy_params": proxy_params}]

    tvb_to_spikeNet_transformer = kwargs.pop("tvb_to_spikeNet_transformer",
                                             kwargs.pop("tvb_to_spikeNet_transformer_model", None))
    if tvb_to_spikeNet_transformer:
        tvb_to_spikeNet_interfaces[0]["transformer_model"] = tvb_to_spikeNet_transformer
    tvb_to_spikeNet_interfaces[0]["transformer_params"].update(kwargs.pop("tvb_to_spikeNet_transformer_params", {}))
    tvb_to_spikeNet_proxy = kwargs.pop("tvb_to_spikeNet_proxy", kwargs.pop("tvb_to_spikeNet_proxy_model", None))
    if tvb_to_spikeNet_proxy:
        tvb_to_spikeNet_interfaces[0]["proxy_model"] = tvb_to_spikeNet_proxy
    tvb_to_spikeNet_interfaces[0]["proxy_params"].update(kwargs.pop("tvb_to_spikeNet_proxy_params", {}))

    spikeNet_to_tvb_interfaces = [{"voi": "E", "populations": "E",
                                   "transformer_params":
                                       {"scale_factor":
                                            np.array([1e-4])/populations_order}},  # (dt(ms) * 1000)Hz*n_neurons)
                                  {"voi": "I", "populations": "I",
                                   "transformer_params":
                                       {"scale_factor":
                                            np.array([1e-4])/populations_order}}]  # (dt(ms) * 1000)Hz*n_neurons

    spikeNet_to_tvb_transformer = kwargs.pop("spikeNet_to_tvb_transformer",
                                             kwargs.pop("spikeNet_to_tvb_transformer_model", None))
    if spikeNet_to_tvb_transformer:
        spikeNet_to_tvb_interfaces[0]["transformer_model"] = spikeNet_to_tvb_transformer
    spikeNet_to_tvb_interfaces[0]["transformer_params"].update(kwargs.pop("spikeNet_to_tvb_transformer_params", {}))

    return main_example(WilsonCowan, WilsonCowanBuilder, spiking_proxy_inds, model_params=model_params,
                        populations_order=populations_order,
                        tvb_to_spikeNet_interfaces=tvb_to_spikeNet_interfaces,
                        spikeNet_to_tvb_interfaces=spikeNet_to_tvb_interfaces,
                        **kwargs)


if __name__ == "__main__":
    wilson_cowan_example(model="SPIKES")
