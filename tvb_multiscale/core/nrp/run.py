# -*- coding: utf-8 -*-

def run_tvb(tvb_app, cosim_updates):
    return tvb_app.run_for_synchronization_time(cosim_updates)


def run_spikeNet(spikeNet_app, cosim_updates):
    return spikeNet_app.run_for_synchronization_time(cosim_updates)


def run_transformer(transformer_app, cosim_updates):
    return transformer_app.run_for_synchronization_time(cosim_updates)
