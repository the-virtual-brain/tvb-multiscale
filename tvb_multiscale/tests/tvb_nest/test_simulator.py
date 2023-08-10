# -*- coding: utf-8 -*-
import os
import shutil

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import matplotlib as mpl
mpl.use('Agg')

import numpy as np

from tvb_multiscale.tvb_nest.config import Config

from docker.launch_example import launch_example_nest


# def prepare_launch_default_simulation():
#     return launch_example_nest(False, plot_write=False, simulation_length=36.0)
#
#
# def test_connectivity_weights_shape(number_of_regions=None, weights=None):
#     if weights is None:
#         simulator, results = prepare_launch_default_simulation()
#         number_of_regions = simulator.connectivity.number_of_regions
#         weights = simulator.connectivity.tract_lengths
#     assert weights.shape == (number_of_regions, number_of_regions)
#
#
# def test_connectivity_tract_lengths_shape(number_of_regions=None,  tract_lengths=None):
#     if tract_lengths is None:
#         simulator, results = prepare_launch_default_simulation()
#         number_of_regions = simulator.connectivity.number_of_regions
#         tract_lengths = simulator.connectivity.tract_lengths
#     assert tract_lengths.shape == (number_of_regions, number_of_regions)
#
#
# def test_results_shape(simulator=None, results=None):
#     if results is None:
#         simulator, results = prepare_launch_default_simulation()
#     assert not np.isinf(results.ravel()).all()
#     assert not np.isnan(results.ravel()).all()
#     assert results.shape == (360+simulator.synchronization_n_step,
#                              simulator.model.nvar, simulator.connectivity.number_of_regions, 1)
#
#
# def teardown_function():
#     output_folder = Config().out._out_base
#     if os.path.exists(output_folder):
#         shutil.rmtree(output_folder)
#
#
# # if __name__ == "__main__":
# #     simulator, results = prepare_launch_default_simulation()
# #     test_connectivity_weights_shape(simulator.connectivity.number_of_regions,
# #                                     simulator.connectivity.weights)
# #     test_connectivity_tract_lengths_shape(simulator.connectivity.number_of_regions,
# #                                           simulator.connectivity.tract_lengths)
# #     test_results_shape(simulator, results)
