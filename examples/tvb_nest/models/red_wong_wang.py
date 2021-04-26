# -*- coding: utf-8 -*-

from examples.tvb_nest.example import main_example
from examples.models.red_wong_wang import \
    red_wong_wang_excio_example as red_wong_wang_excio_example_base, \
    red_wong_wang_excio_inhi_example as red_wong_wang_excio_inhi_example_bae

from tvb_multiscale.tvb_nest.interfaces.models.builders.red_ww import \
     RedWWexcIOBuilder as RedWongWangExcIOTVBNESTInterfaceBuilder, \
     RedWWexcIOinhIBuilder as RedWongWangExcIOInhITVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.nest_models.models.ww_deco import WWDeco2013Builder, WWDeco2014Builder


def red_wong_wang_excio_example(**kwargs):
    return main_example(red_wong_wang_excio_example_base,
                        WWDeco2013Builder, RedWongWangExcIOTVBNESTInterfaceBuilder,
                        **kwargs)


def red_wong_wang_excio_inhi_example(**kwargs):
    return main_example(red_wong_wang_excio_inhi_example_bae,
                        WWDeco2014Builder, RedWongWangExcIOInhITVBNESTInterfaceBuilder,
                        **kwargs)


if __name__ == "__main__":
    import sys
    if sys.argv[-1] == "1":
        red_wong_wang_excio_inhi_example(tvb_to_spikeNet_mode="rate")
    else:
        red_wong_wang_excio_example(tvb_to_spikeNet_mode="rate")
