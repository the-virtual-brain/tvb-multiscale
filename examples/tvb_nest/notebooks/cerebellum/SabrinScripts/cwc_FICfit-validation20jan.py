from examples.tvb_nest.notebooks.cerebellum.scripts.scripts import *
import sbi
print(sbi.__version__)
config, plotter = configure()
for iG in range(3):
    for iB in range(500):
        filename = "bsr_iG%02d_%03d.npy" % (iG, iB)
        print(filename)
        filepath = os.path.join(config.out.FOLDER_RES, filename)
        try:
            dat = np.load(filepath)
        except Exception as e:
            print("No file for iG=%d, iB=%d!\n%s" % (iG, iB, str(e)))
            continue
        print(dat.shape)
def posterior_samples_filepath_train(config,N_TrainSamples ,iG=None,filepath=None, extension=None):
    if iG is None:
        return config.POSTERIOR_SAMPLES_PATH
    if filepath is None or extension is None:
        filepath, extension = os.path.splitext(os.path.join(config.out.FOLDER_RES,
                                                            config.POSTERIOR_SAMPLES_PATH))
    return "%s_iG%02d_%04d_Train%s" % (filepath, iG, N_TrainSamples ,extension)
def write_posterior_samples(samples,N_TrainSamples, map=None, iG=None ,config=None):
    config = assert_config(config, return_plotter=False)
    filepath = posterior_samples_filepath_train(config,N_TrainSamples,iG)
    if os.path.isfile(filepath):
        samples_fit = np.load(filepath, allow_pickle=True).item()
    else:
        samples_fit = {}
        # Get G for this run:
        samples_fit["G"] = config.Gs[iG]
        samples_fit['samples'] = []
        samples_fit['mean'] = []
        samples_fit['std'] = []
        samples_fit['map'] = []
    samples_fit['samples'].append(samples.numpy())
    samples_fit['mean'].append(samples.mean(axis=0).numpy())
    samples_fit['std'].append(samples.std(axis=0).numpy())
    if map is not None:
        samples_fit['map'].append(map.numpy())
    np.save(filepath, samples_fit, allow_pickle=True)
    return samples_fit

def sbi_infer_first_half(priors, priors_samples, sim_res, verbose):
    # Initialize the inference algorithm class instance:
    inference = SNPE(prior=priors)
    # Append to the inference the priors samples and simulations results
    # and train the network:
    density_estimator = inference.append_simulations(priors_samples, sim_res).train()
    keep_building = -10
    posterior = None
    exception = "None"
    while keep_building < 0:
        try:
            # Build the posterior:
            if verbose:
                print("Building the posterior...")
            posterior = inference.build_posterior(density_estimator)
            keep_building = 0
        except Exception as e:
            exception = e
            warnings.warn(str(e) + "\nTrying again for the %dth time!" % (10 + keep_building + 2))
            keep_building += 1
    #if posterior is None:
        #raise Exception(exception)
    #posterior.set_default_x(target)
    return posterior


def sbi_infer_second_half(posterior, target,n_samples_per_run, verbose):
    if posterior is None:
        raise Exception(exception)
    posterior.set_default_x(target)
    return posterior.sample((n_samples_per_run,)), posterior.map()
def Train_and_Test(n_train_samples,n_test_sample,iG,verbose):
    tic = time.time()
    
    priors, priors_samples, sim_res = load_priors_and_simulations_for_sbi(iG, config=config)
    n_samples = sim_res.shape[0]
    all_inds = list(range(n_samples))
    sampl_inds = random.sample(all_inds, n_train_samples)
    posteriorx=sbi_infer_first_half(priors, priors_samples[sampl_inds], sim_res[sampl_inds],verbose )
    write_posterior(posteriorx,iG, iR=None, config=config)
    for i in range(n_test_sample):
        posterior_samples, map = sbi_infer_second_half(posteriorx, sim_res[i].numpy(),1000, verbose)
         
        samples_fit = write_posterior_samples(posterior_samples,n_train_samples, map,iG ,config)
    if config.VERBOSE:
        print("Done with fitting with all samples in sec!")

    # Plot posterior:
    plot_infer_for_iG(iG, iR=None, samples=samples_fit, config=config);

    if config.VERBOSE:
        print("\n\nFinished after sec!" )
        print("\n\nFind results in %s!" % config.out.FOLDER_RES)
    return posterior_samples


if __name__ == "__main__":
    parser = args_parser("cwc_FICfit-validation20jan")
    parser.add_argument('--script_id', '-scr',
                        dest='script_id', metavar='script_id',
                        type=int, required=False, default=1, # nargs=1,
                        help="")
    parser.add_argument('--Ts', '-ts',
                        dest='Ts', metavar='Ts',
                        type=int, required=False, default=-1, # nargs=1,
                        help="")
    parser.add_argument('--iG', '-ig',
                        dest='iG', metavar='iG',
                        type=int, required=False, default=-1, # nargs=1,
                        help="")

    args, parser_args, parser = parse_args(parser, def_args=DEFAULT_ARGS)
    verbose = args.get('verbose', DEFAULT_ARGS['verbose'])
    config = configure(**args)[0]
    iG = parser_args.iG
    Ts = parser_args.Ts
    Train_and_Test(Ts,760,iG,config.VERBOSE)
