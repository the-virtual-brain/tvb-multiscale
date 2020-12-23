import os
import shutil

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb_multiscale.tvb_annarchy.config import CONFIGURED, Config
from tvb_multiscale.tvb_annarchy.annarchy_models.builders.models.basal_ganglia_izhikevich import \
    BasalGangliaIzhikevichBuilder

from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.simulator import Simulator
from tvb.simulator.models.reduced_wong_wang_exc_io_inh_i import ReducedWongWangExcIOInhI
from tvb.simulator.monitors import Raw  # , Bold  # , EEG


def test(dt=0.1, noise_strength=0.001, config=CONFIGURED):
    # Select the regions for the fine scale modeling with ANNarchy spiking networks
    anarchy_nodes_ids = list(range(10))  # the indices of fine scale regions modeled with ANNarchy
    # In this example, we model parahippocampal cortices (left and right) with ANNarchy
    connectivity = Connectivity.from_file(CONFIGURED.DEFAULT_CONNECTIVITY_ZIP)
    connectivity.configure()

    # Create a TVB simulator and set all desired inputs
    # (connectivity, model, surface, stimuli etc)
    # We choose all defaults in this example
    simulator = Simulator()
    simulator.integrator.dt = dt
    # simulator.integrator.noise.nsig = np.array([noise_strength])
    simulator.model = ReducedWongWangExcIOInhI()

    simulator.connectivity = connectivity
    mon_raw = Raw(period=simulator.integrator.dt)
    simulator.monitors = (mon_raw,)

    # Build a ANNarchy network model with the corresponding builder
    # Using all default parameters for this example
    anarchy_model_builder = BasalGangliaIzhikevichBuilder(simulator, anarchy_nodes_ids, config=config)
    anarchy_model_builder.configure()
    for prop in ["min_delay", "tvb_dt", "tvb_model", "tvb_connectivity", "tvb_weights", "tvb_delays",
                 "number_of_nodes", "number_of_spiking_nodes", "spiking_nodes_labels",
                 "number_of_populations", "populations_models", "populations_nodes",
                 "populations_scales", "populations_sizes", "populations_params",
                 "populations_connections_labels", "populations_connections_models", "populations_connections_nodes",
                 "populations_connections_weights", "populations_connections_delays",
                 "populations_connections_receptor_types", "populations_connections_conn_spec",
                 "nodes_connections_labels", "nodes_connections_models",
                 "nodes_connections_source_nodes", "nodes_connections_target_nodes",
                 "nodes_connections_weights", "nodes_connections_delays", "nodes_connections_receptor_types",
                 "nodes_connections_conn_spec"]:
        print("%s:\n%s\n\n" % (prop, str(getattr(anarchy_model_builder, prop))))


def teardown_function():
    output_folder = Config().out._out_base
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)


if __name__ == "__main__":
    test()
