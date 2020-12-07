import numpy as np
from matplotlib import pyplot
from tvb_multiscale.core.utils.data_structures_utils import combine_DataArray_dims


def plot_correlations(corrs, plotter, **kwargs):
    from xarray import DataArray
    data, dims, coords = combine_DataArray_dims(corrs, [(0, 2), (1, 3)], join_string=", ", return_array=False)
    figsize = kwargs.pop("figsize", plotter.config.DEFAULT_SIZE)
    pyplot.figure()
    DataArray(data, dims=dims, name=corrs.name). \
        plot(x=dims[0], y=dims[1], cmap="jet",
             xticks=np.arange(data.shape[0]), yticks=np.arange(data.shape[1]),
             figsize=figsize, **kwargs)
    ax = pyplot.gca()
    ax.set_xticklabels(coords[dims[0]], rotation=45, ha="right")
    ax.set_yticklabels(coords[dims[1]])
    ax.set_aspect(1./ax.get_data_ratio())
    pyplot.tight_layout()
    plotter.base._save_figure(figure_name="Populations' Spikes' Correlation Coefficient")
    if not plotter.base.config.SHOW_FLAG:
       pyplot.close()