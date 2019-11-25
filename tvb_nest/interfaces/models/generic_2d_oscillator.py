# -*- coding: utf-8 -*-

import numpy as np

from tvb_nest.config import CONFIGURED
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_nest.interfaces.base import TVBNESTInterface
from tvb.simulator.models.oscillator import Generic2dOscillator as TVBGeneric2dOscillator


LOG = initialize_logger(__name__)


class Generic2dOscillator(TVBNESTInterface):
    tvb_model = TVBGeneric2dOscillator()

    # TODO: confirm the following:
    w_tvb_sv_to_V_m = lambda state: -60.0 + 5 * state
    w_V_m_to_tvb_sv = lambda V_m: (60.0 + V_m) / 5

    def __init__(self, config=CONFIGURED.nest, V_th=-55.0, V_reset=-60.0):
        self.V0 = np.abs(V_th)
        self.V_range = np.abs(np.abs(V_reset)-self.V0)

        super(Generic2dOscillator, self).__init__(config)
        LOG.info("%s created!" % self.__class__)

    def configure(self, tvb_model):
        # TODO: confirm the following:
        self.w_tvb_sv_to_V_m = lambda state: -self.V0 + self.V_range * state
        self.w_V_m_to_tvb_sv = lambda V_m: (self.V0 + V_m) / self.V_range
        super(Generic2dOscillator, self).configure(tvb_model)
