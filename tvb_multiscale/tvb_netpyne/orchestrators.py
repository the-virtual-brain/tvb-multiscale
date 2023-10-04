from logging import Logger
from tvb.basic.neotraits.api import Attr

from tvb_multiscale.core.neotraits import HasTraits
from tvb_multiscale.core.orchestrators.spikeNet_app import SpikeNetSerialApp  # , SpikeNetParallelApp
from tvb_multiscale.core.orchestrators.tvb_app import TVBSerialApp as TVBSerialAppBase
from tvb_multiscale.core.orchestrators.serial_orchestrator import SerialOrchestrator


from tvb_multiscale.tvb_netpyne.config import Config, CONFIGURED, initialize_logger
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.netpyne_factory import load_netpyne
from tvb_multiscale.tvb_netpyne.netpyne_models.network import NetpyneNetwork
from tvb_multiscale.tvb_netpyne.netpyne_models.builders.base import NetpyneNetworkBuilder
from tvb_multiscale.tvb_netpyne.interfaces.builders import TVBNetpyneInterfaceBuilder


class NetpyneApp(HasTraits):

    """NetpyneApp class"""

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Config class instance.""",
        required=True,
        default=CONFIGURED
    )

    spikeNet_builder = Attr(
        label="Netpyne Network Builder",
        field_type=NetpyneNetworkBuilder,
        doc="""Instance of Netpyne Network Builder.""",
        required=False
    )

    spiking_network = Attr(
        label="NetPyNE Network",
        field_type=NetpyneNetwork,
        doc="""Instance of NetpyneNetwork class.""",
        required=False
    )

    _spikeNet_builder_type = NetpyneNetworkBuilder
    _spikeNet_type = NetpyneNetwork

    @property
    def netpyne_instance(self):
        return self.spiking_cosimulator

    def synaptic_weight_scale(self, is_coupling_mode_tvb):
        # TODO: this is not specific to serial app. Once Parallel app is ready, move to some common ancestor,
        #  to use in both cases (see also: TVBNetpyneSerialOrchestrator.build_interfaces())
        if is_coupling_mode_tvb:
            return 1e-2
        else:  # "spikeNet"
            return 1

    @property
    def netpyne_network(self):
        return self.spiking_network

    @property
    def netpyne_model_builder(self):
        return self.spikeNet_builder

    def start(self):
        self.spiking_cosimulator = load_netpyne(self.config)

    def configure(self):
        super(NetpyneApp, self).configure()
        self.spikeNet_builder.netpyne_synaptic_weight_scale = self.synaptic_weight_scale(is_coupling_mode_tvb=False)
        # TODO: check self.spikeNet_builder.netpyne_instance = self.spiking_cosimulator
    def clean_up(self):
        self.spiking_cosimulator.finalize()

    def stop(self):
        pass


class NetpyneSerialApp(NetpyneApp, SpikeNetSerialApp):

    """NetpyneSerialApp class"""

    def configure(self):
        SpikeNetSerialApp.configure(self)
        NetpyneApp.configure(self)

    def run(self, *args, **kwargs):
        SpikeNetSerialApp.run(self)
        self.configure()
        self.build()


class TVBSerialApp(TVBSerialAppBase):

    """TVBSerialApp class"""


    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    interfaces_builder = Attr(
        label="TVBNESTInterfaces builder",
        field_type=TVBNetpyneInterfaceBuilder,
        doc="""Instance of TVBNESTInterfaces' builder class.""",
        required=False
    )

    spiking_network = Attr(
        label="NEST Network",
        field_type=NetpyneNetwork,
        doc="""Instance of NESTNetwork class.""",
        required=False
    )

    _default_interface_builder_type = TVBNetpyneInterfaceBuilder


class TVBNetpyneSerialOrchestrator(SerialOrchestrator):

    config = Attr(
        label="Configuration",
        field_type=Config,
        doc="""Configuration class instance.""",
        required=True,
        default=CONFIGURED
    )

    logger = Attr(
        label="Logger",
        field_type=Logger,
        doc="""logging.Logger instance.""",
        required=True,
        default=initialize_logger(__name__, config=CONFIGURED)
    )

    tvb_app = Attr(
        label="TVBSerial app",
        field_type=TVBSerialApp,
        doc="""Application for running TVB serially.""",
        required=False,
        default=TVBSerialApp()
    )

    spikeNet_app = Attr(
        label="NetPyNE Network app",
        field_type=NetpyneSerialApp,
        doc="""Application for running a Spiking Network (co)simulator serially.""",
        required=False,
        default=NetpyneSerialApp()
    )

    def link_spikeNet_to_TVB_cosimulator(self):
        super(TVBNetpyneSerialOrchestrator, self).link_spikeNet_to_TVB_cosimulator()
        # for parallel spiking simulation
        self.tvb_app.cosimulator.is_root_host = self.spikeNet_app.spiking_cosimulator.isRootNode
        self.tvb_app.cosimulator.synchronize_spiking_hosts = self.spikeNet_app.spiking_cosimulator.gatherFromNodes

    def build(self):
        self.tvb_app.interfaces_builder.synaptic_weight_scale_func = self.spikeNet_app.synaptic_weight_scale
        self.tvb_app.interfaces_builder.synaptic_model_funcs = \
            self.spikeNet_app.spikeNet_builder.proxy_node_synaptic_model_funcs
        super(TVBNetpyneSerialOrchestrator, self).build()
