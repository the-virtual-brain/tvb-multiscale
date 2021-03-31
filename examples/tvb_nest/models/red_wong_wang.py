# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_nest.interfaces.models.red_wong_wang import \
    RedWongWangExcIOTVBNESTInterfaceBuilder, RedWongWangExcIOInhITVBNESTInterfaceBuilder
from tvb_multiscale.tvb_nest.nest_models.models.ww_deco import WWDeco2013Builder, WWDeco2014Builder

from examples.tvb_nest.example import main_example
from examples.models.red_wong_wang import \
    red_wong_wang_excio_example as red_wong_wang_excio_example_base, \
    red_wong_wang_excio_inhi_example as red_wong_wang_excio_inhi_example_bae


def red_wong_wang_excio_example(**kwargs):
    return main_example(red_wong_wang_excio_example_base,
                        WWDeco2013Builder, RedWongWangExcIOTVBNESTInterfaceBuilder(),
                        **kwargs)


def red_wong_wang_excio_inhi_example(**kwargs):
    return main_example(red_wong_wang_excio_inhi_example_bae,
                        WWDeco2014Builder, RedWongWangExcIOInhITVBNESTInterfaceBuilder(),
                        **kwargs)


if __name__ == "__main__":
    import sys
    if sys.argv[-1] == "1":
        red_wong_wang_excio_inhi_example(model="RATE")
    else:
        red_wong_wang_excio_example(model="RATE")
