# -*- coding: utf-8 -*-

def final_tvb(tvb_app, plot=True):
    if plot:
        tvb_app.plot()
    tvb_app.stop()
    return tvb_app


def final_spikeNet(spikeNet_app, plot=True):
    if plot:
        spikeNet_app.plot()
    spikeNet_app.stop()
    return spikeNet_app


def final_transformer(transformer_app):
    transformer_app.stop()
    return transformer_app
