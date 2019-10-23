# -*- coding: utf-8 -*-

from tvb_nest.config import CONFIGURED
from tvb_scripts.utils.log_error_utils import initialize_logger
from tvb_nest.interfaces.base import TVBNESTInterface
from tvb_nest.simulator_tvb.model_reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI

LOG = initialize_logger(__name__)


class RedWWexcIOinhI(TVBNESTInterface):
    tvb_model = ReducedWongWangExcIOInhI()

    # TODO: confirm the following:
    w_nest_spikes_to_tvb_rate = 1.0  # (assuming spikes/ms in TVB)
    # Convert rate to a number in the [0,1] interval,
    # assuming a maximum rate of 1000.0 Hz, or 1 spike/sec
    w_nest_spikes_to_tvb_sv = 1.0  # (assuming spikes/ms in TVB)
    w_tvb_sv_to_nest_spike_rate = 1000.0  # (spike rate in NEST is in spikes/sec, whereas dt is in ms)
    w_tvb_sv_to_nest_current = 1000.0  # (1000.0 (nA -> pA), because I_e, and dc_generator amplitude in NEST are in pA)

    def __init__(self, config=CONFIGURED):
        super(RedWWexcIOinhI, self).__init__(config)
        LOG.info("%s created!" % self.__class__)

    def configure(self, tvb_model):
        super(RedWWexcIOinhI, self).configure(tvb_model)
        # TODO: solve the following inconsistency in the case that J_N is different among regions:
        # The index of J_N refers to the source TVB region in case of dc_generator,
        # but to the target NEST node, in case of direct application to is I_e parameter
        # Therefore, for the moment the direct application is more consistent.
        self.w_tvb_sv_to_nest_current *= self.tvb_model.J_N
