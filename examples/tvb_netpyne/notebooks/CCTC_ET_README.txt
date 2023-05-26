CCTC_essential_tremor.py as a python file is a temporary workaround for one technical issue. Will be converted to .ipynb once fixed.
In order to run the example:
1. Clone the NetPyNE network of cortico-cerebello-thalamo-cortical (CCTC) network from separate repo:
    git clone https://github.com/suny-downstate-medical-center/thalamic_VIM_essential_tremor.git ~/packages/tvb-multiscale/tvb_multiscale/tvb_netpyne/netpyne_models/models/thalamic_VIM_ET
2. In `tvb_multiscale/tvb_netpyne/netpyne_models/models/thalamic_VIM_ET/src/netParams.py` set `isTVB` flag to True
3. Compile mpd-files:
    nrnivmodl tvb_multiscale/tvb_netpyne/netpyne_models/models/thalamic_VIM_ET/mod
4. Run the example
    python examples/tvb_netpyne/notebooks/CCTC_essential_tremor.py