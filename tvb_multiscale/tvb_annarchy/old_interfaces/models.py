# -*- coding: utf-8 -*-

from tvb_multiscale.tvb_annarchy.config import CONFIGURED, initialize_logger
from tvb_multiscale.tvb_annarchy.old_interfaces.base import TVBANNarchyInterface

from tvb.simulator.models.reduced_wong_wang_exc_io import ReducedWongWangExcIO
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.models.wilson_cowan_constraint import WilsonCowan as WilsonCowanSimModel
from tvb.simulator.models.generic_2d_oscillator_multiscale import Generic2dOscillator as Generic2dOscillatorSimModel


LOG = initialize_logger(__name__)


class Linear(TVBANNarchyInterface):
    tvb_model = ReducedWongWangExcIO()

    def __init__(self, config=CONFIGURED):
        super(Linear, self).__init__(config)
        LOG.info("%s created!" % self.__class__)


class RedWWexcIO(TVBANNarchyInterface):
    tvb_model = ReducedWongWangExcIO()

    def __init__(self, config=CONFIGURED):
        super(RedWWexcIO, self).__init__(config)
        LOG.info("%s created!" % self.__class__)


class RedWWexcIOinhI(TVBANNarchyInterface):
    tvb_model = ReducedWongWangExcIOInhI()

    def __init__(self, config=CONFIGURED):
        super(RedWWexcIOinhI, self).__init__(config)
        LOG.info("%s created!" % self.__class__)


class Generic2dOscillator(TVBANNarchyInterface):
    tvb_model = Generic2dOscillatorSimModel()

    def __init__(self, config=CONFIGURED):
        super(Generic2dOscillator, self).__init__(config)
        LOG.info("%s created!" % self.__class__)


class WilsonCowan(TVBANNarchyInterface):
    tvb_model = WilsonCowanSimModel()

    def __init__(self, config=CONFIGURED):
        super(WilsonCowan, self).__init__(config)
        LOG.info("%s created!" % self.__class__)
