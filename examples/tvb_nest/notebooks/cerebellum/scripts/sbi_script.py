# -*- coding: utf-8 -*-

import torch
from sbi.inference.base import infer, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils
from sbi import analysis as analysis

from examples.tvb_nest.notebooks.cerebellum.scripts.base import *
from examples.tvb_nest.notebooks.cerebellum.scripts.tvb import run_workflow


def build_priors(config):
    if config.PRIORS_MODE.lower() == "uniform":
        priors = utils.torchutils.BoxUniform(low=torch.as_tensor(config.prior_min),
                                             high=torch.as_tensor(config.prior_max))
    else:
        priors_normal = torch.distributions.Normal(loc=torch.as_tensor(config.prior_loc),
                                                   scale=torch.as_tensor(config.prior_sc))
        #     priors = torch.distributions.MultivariateNormal(loc=torch.as_tensor(config.prior_loc),
        #                                                     scale_tril=torch.diag(torch.as_tensor(config.prior_sc)))
        priors = torch.distributions.Independent(priors_normal, 1)
    return priors


def sample_priors_for_sbi(config=None):
    config = assert_config(config)
    dummy_sim = lambda priors: priors
    priors = build_priors(config)
    simulator, priors = prepare_for_sbi(dummy_sim, priors)
    priors_samples, sim_res = simulate_for_sbi(dummy_sim, proposal=priors,
                                               num_simulations=config.N_SIMULATIONS,
                                               num_workers=config.SBI_NUM_WORKERS)
    return priors_samples, sim_res


def priors_samples_per_batch(priors_samples=None, config=None, write_to_files=True):
    config = assert_config(config)
    if priors_samples is None:
        priors_samples = sample_priors_for_sbi(config)[0]
    batch_samples = []
    filepath, extension = os.path.splitext(os.path.join(config.out.FOLDER_RES, config.BATCH_PRIORS_SAMPLES_FILE))
    for iR in range(config.N_SIM_BATCHES):
        batch_samples.append(priors_samples[iR::config.N_SIM_BATCHES])
        if write_to_files:
            torch.save(batch_samples[-1], "%s_%03d%s" % (filepath, iR, extension))
    return batch_samples


def priors_samples_per_batch_for_iG(priors_samples=None, iG=None, config=None, write_to_files=True):
    config = assert_config(config)
    if iG is not None and "iG" not in config.BATCH_PRIORS_SAMPLES_FILE:
        BATCH_PRIORS_SAMPLES_FILE, extension = os.path.splitext(config.BATCH_PRIORS_SAMPLES_FILE)
        config.BATCH_PRIORS_SAMPLES_FILE = "%s_iG%02d%s" % (BATCH_PRIORS_SAMPLES_FILE, iG, extension)
    return priors_samples_per_batch(priors_samples, config, write_to_files)


def load_priors_samples_per_batch(iB, config=None):
    config = assert_config(config)
    filepath, extension = os.path.splitext(os.path.join(config.out.FOLDER_RES, config.BATCH_PRIORS_SAMPLES_FILE))
    return torch.load("%s_%03d%s" % (filepath, iB, extension))


def load_priors_samples_per_batch_per_iG(iB, iG=None, config=None):
    config = assert_config(config)
    if iG is not None and "iG" not in config.BATCH_PRIORS_SAMPLES_FILE:
        BATCH_PRIORS_SAMPLES_FILE, extension = os.path.splitext(config.BATCH_PRIORS_SAMPLES_FILE)
        config.BATCH_PRIORS_SAMPLES_FILE = "%s_iG%02d%s" % (BATCH_PRIORS_SAMPLES_FILE, iG, extension)
    return load_priors_samples_per_batch(iB, config)


def write_batch_sim_res_to_file(sim_res, iB, config=None):
    config = assert_config(config)
    filepath, extension = os.path.splitext(os.path.join(config.out.FOLDER_RES, config.BATCH_SIM_RES_FILE))
    np.save("%s_%03d%s" % (filepath, iB, extension), sim_res, allow_pickle=True)


def write_batch_sim_res_to_file_per_iG(sim_res, iB, iG=None, config=None):
    config = assert_config(config)
    if iG is not None and "iG" not in config.BATCH_SIM_RES_FILE:
        BATCH_SIM_RES_FILE, extension = os.path.splitext(config.BATCH_SIM_RES_FILE)
        config.BATCH_SIM_RES_FILE = "%s_iG%02d%s" % (BATCH_SIM_RES_FILE, iG, extension)
    return write_batch_sim_res_to_file(sim_res, iB, config)


def simulate_TVB_for_sbi_batch(iB, iG=None, config=None, write_to_file=True):
    config = assert_config(config)
    # Get the default values for the parameter except for G
    params = OrderedDict()
    for pname, pval in zip(config.PRIORS_PARAMS_NAMES, config.model_params.values()):
        params[pname] = pval
    batch_samples = load_priors_samples_per_batch_per_iG(iB, iG, config)
    n_simulations = batch_samples.shape[0]
    sim_res = []
    for iS in range(n_simulations):
        priors_params = params.copy()
        if iG is not None:
            priors_params["G"] = config.Gs[iG]
        for prior, prior_name in zip(batch_samples[iS], config.PRIORS_PARAMS_NAMES):
            try:
                numpy_prior = prior.numpy()
            except:
                numpy_prior = prior
            priors_params[prior_name] = numpy_prior
        print("\n\nSimulating for parameters:\n%s\n" % str(priors_params))
        sim_res.append(run_workflow(**priors_params, plot_flag=False)[0])
    if write_to_file:
        write_batch_sim_res_to_file_per_iG(sim_res, iB, iG, config)
    return sim_res


# def simulate_TVB_for_sbi(priors, priors_params_names, **params):
#     priors_params = params.copy()
#     # Convert all tensor parameters to numpy arrays
#     for prior, prior_name in zip(priors, priors_params_names):
#         try:
#             numpy_prior = prior.numpy()
#         except:
#             numpy_prior = prior
#         priors_params[prior_name] = numpy_prior
#     # Run the simulation and return only the PSD output to be fit:
#     return run_workflow(**priors_params, plot_flag=False)[0]
#
#
# def sbi_fit(iG, config=None):
#
#     tic = time.time()
#
#     if config is None:
#         # Create a configuration if one is not given
#         config = configure(plot_flag=False)[0]
#
#     # Get the default values for the parameter except for G
#     params = OrderedDict()
#     for pname, pval in zip(config.PRIORS_PARAMS_NAMES, config.model_params.values()):
#         params[pname] = pval
#     print("params =\n", params)
#
#     # Load the target
#     PSD_target = np.load(config.PSD_TARGET_PATH, allow_pickle=True).item()
#     # Duplicate the target for the two M1 regions (right, left) and the two S1 barrel field regions (right, left)
#     #                                        right                       left
#     psd_targ_conc = np.concatenate([PSD_target["PSD_M1_target"], PSD_target["PSD_M1_target"],
#                                     PSD_target["PSD_S1_target"], PSD_target["PSD_S1_target"]])
#
#     # Get G for this run:
#     G = config.Gs[iG]
#
#     # Define the simulation function for sbi for this G
#     simulate_for_sbi_for_g = lambda priors: simulate_TVB_for_sbi(priors,
#                                                                  priors_params_names=config.PRIORS_PARAMS_NAMES,
#                                                                  G=G, **params)
#
#     # Build the priors
#     priors = build_priors(config)
#
#     print("\n\nFitting for G = %g!\n" % G)
#     tic = time.time()
#     for iR in range(config.N_RUNS):
#         print("\nFitting run %d " % iR)
#
#         # Train the neural network to approximate the posterior:
#         posterior = infer(simulate_for_sbi_for_g, priors,
#                           method=config.SBI_METHOD,
#                           num_simulations=config.N_SIMULATIONS,
#                           num_workers=config.SBI_NUM_WORKERS)
#
#         print("\nSampling posterior...")
#         if iR:
#             samples_fit = torch.cat((samples_fit, posterior.sample((config.N_SAMPLES_PER_RUN,), x=psd_targ_conc)), 0)
#         else:
#             samples_fit = posterior.sample((config.N_SAMPLES_PER_RUN,), x=psd_targ_conc)
#
#     print("Done in %g sec!" % (time.time() - tic))
#
#     # Compute the sample mean, add to the results dictionary and write to file:
#     if os.path.isfile(config.SAMPLES_GS_PATH):
#         samples_fit_Gs = np.load(config.SAMPLES_GS_PATH, allow_pickle=True).item()
#     else:
#         samples_fit_Gs = {}
#     samples_fit_Gs[G] = {}
#     samples_fit_Gs[G]['samples'] = samples_fit.numpy()
#     samples_fit_Gs[G]['mean'] = samples_fit.mean(axis=0).numpy()
#     np.save(config.SAMPLES_GS_PATH, samples_fit_Gs, allow_pickle=True)
#
#     # Plot posterior:
#     print("\nPlotting posterior...")
#     limits = []
#     for pmin, pmax in zip(config.prior_min, config.prior_max):
#         limits.append([pmin, pmax])
#     fig, axes = analysis.pairplot(samples_fit,
#                                   limits=limits,
#                                   ticks=limits,
#                                   figsize=(10, 10),
#                                   points=np.array(list(params.values())),
#                                   points_offdiag={'markersize': 6},
#                                   points_colors=['r'] * config.n_priors)
#     plt.savefig(os.path.join(config.figures.FOLDER_FIGURES, 'sbi_pairplot_%g.png' % G))
#
#     # Run one simulation with the posterior means:
#     print("\nSimulating with posterior means...")
#     params.update(dict(zip(config.PRIORS_PARAMS_NAMES, samples_fit_Gs[G]['mean'])))
#     PSD, results, simulator, output_config = run_workflow(PSD_target=PSD_target, plot_flag=True, G=G, **params)
#
#     duration = time.time() - tic
#     print("\n\nFinished after %g sec!" % duration)
#     print("\n\nFind results in %s!" % config.out.FOLDER_RES)
#
#     return samples_fit_Gs, results, fig, simulator, output_config


if __name__ == "__main__":
    import sys

    # samples_fit_Gs, results, fig, simulator, output_config = sbi_fit(int(sys.argv[-1]))
    config = configure()[0]
    if len(sys.argv) == 1:
        iB = int(sys.argv[-1])
        iG = None
    else:
        iB = int(sys.argv[-1])
        iG = int(sys.argv[-2])

    sim_res = simulate_TVB_for_sbi_batch(iB, iG, config=config, write_to_file=True)
