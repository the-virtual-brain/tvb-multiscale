from tvb_multiscale.tvb_netpyne.netpyne_models.builders.base import NetpyneNetworkBuilder
# from tvb_multiscale.core.spiking_models.builders.templates import tvb_delay, scale_tvb_weight


class BasalGangliaIzhikevichNetworkBuilder(NetpyneNetworkBuilder):

    def connect_two_populations(self, pop_src, src_inds_fun, pop_trg, trg_inds_fun, conn_spec, syn_spec):

        if syn_spec.get('receptor_type') == 'gaba':
            syn_spec['weight'] *= -1 # due to the specific implemetation of Izhikevich cell, inhibitory connection is represented as negative weight
        super(BasalGangliaIzhikevichNetworkBuilder, self).connect_two_populations(pop_src, src_inds_fun, pop_trg, trg_inds_fun, conn_spec, syn_spec)