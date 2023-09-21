#
# TVB & NEST, ANNarchy and NetPyNE (NEURON) simulations environment
#
# Install Python 3.10, NEST, ANNarchy, NEURON, NetPyNE, Elephant, PySpike, ray, and then tvb-root.
# Finally link tvb-multiscale from Github sources
#

# Ubuntu 22.04 from NeuroDebian 02/09/2023:
# https://hub.docker.com/layers/library/neurodebian/nd22.04/images/sha256-be7132784be260e212cf3adfa3dbcfad197e3b374d64199b6ad7f9437824f973?context=explore
FROM neurodebian:nd22.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8

# TODO: Confirm this change:
#Package python-dev is not available, but is referred to by another package.
#This may mean that the package is missing, has been obsoleted, or
#is only available from another source
#However the following packages replace it:
#  python2-dev python2 python-dev-is-python3
RUN apt-get update; apt-get install -y automake libtool build-essential git vim octave \
                       wget python3 libpython3.10 libpython3-dev libncurses5-dev libreadline-dev libgsl0-dev cython3 \
                       python3-dev python3-numpy python3-scipy python3-matplotlib python3-pip python3-nose \
                       ipython3 python3-venv python3-virtualenv python3-pyqt5 python-dev-is-python3 libgsl-dev libncurses-dev \
                       python3-pkgconfig subversion cmake zip unzip make; \
    apt-get clean

RUN wget -q http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz \
    && tar xf mpich-3.1.4.tar.gz \
    && cd mpich-3.1.4 \
    && ./configure --disable-fortran --enable-fast=all,O3 --prefix=/usr \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd .. \
    && rm -rf mpich-3.1.4 \
    && rm mpich-3.1.4.tar.gz

# Create user different than root
RUN useradd -ms /bin/bash docker
USER docker
ARG HOME=/home/docker
RUN mkdir $HOME/packages
RUN mkdir $HOME/build
ENV PACKAGES=$HOME/packages
ENV BUILD=$HOME/build
ENV BASEENV=/usr
RUN mkdir $HOME/.local
ENV LOCALENV=$HOME/.local
ENV PYTHONPATH=$PYTHONPATH:$BASEENV:$LOCALENV
ENV PATH="${BASEENV}/bin":"${LOCALENV}/bin":${PATH};

# Basic python dependencies:
RUN $BASEENV/bin/pip3 install --upgrade pip; \
    $BASEENV/bin/pip3 install --upgrade nose ipython numpy scipy matplotlib; \
    $BASEENV/bin/pip3 install parameters quantities neo "django<1.9" django-tagging future hgapi gitpython sumatra nixio numba gunicorn;

# Install from sources libneurosim (NEST boost dependency)
WORKDIR $PACKAGES
RUN git clone --depth 1 https://github.com/INCF/libneurosim.git; \
    cd libneurosim; ./autogen.sh;
WORKDIR $BUILD
RUN mkdir libneurosim; \
    cd libneurosim; \
    PYTHON=$BASEENV/bin/python $PACKAGES/libneurosim/configure --prefix=$BASEENV; \
    make; make install; ls $BASEENV/lib $BASEENV/include;
RUN ln -s /usr/lib/python3.10/config-x86_64-linux-gnu/libpython3.10.so $LOCALENV/lib/;

# Install NEST, latest stable version 3.5
ARG NEST_VER=3.5
ARG NEST=nest-simulator
ARG NEST_SRC=$PACKAGES/$NEST
ARG NEST_INSTALL_DIR=$BUILD/nest
ARG NEST_PYTHON_PREFIX=${NEST_INSTALL_DIR}/lib/python3.10/site-packages
ARG NEST_MODULE_PATH=${NEST_INSTALL_DIR}/lib/nest

# Install NEST 3.5:
# https://github.com/nest/nest-simulator/commit/4e7a7559fd737a4d54a79d0006fc9ebec88ddefd
WORKDIR $PACKAGES
# ARG LAST_SHA_NEST=LATEST
RUN git clone https://github.com/nest/nest-simulator.git
RUN cd $NEST; \
    git fetch origin && git checkout 4e7a7559fd737a4d54a79d0006fc9ebec88ddefd; \
    cmake -DCMAKE_INSTALL_PREFIX=${NEST_INSTALL_DIR} \
          -Dwith-mpi=ON  \
          -Dwith-libneurosim=$LOCALENV \
          -DPYTHON_LIBRARY=/usr/lib \
          -DPYTHON_INCLUDE_DIR=/usr/include/python3.10m \
          ${NEST_SRC}; \
    make; make install

# Install NESTServerClient
RUN $BASEENV/bin/pip3 install flask_cors RestrictedPython;
WORKDIR $PACKAGES
RUN git clone https://github.com/nest/nest-client.git
RUN cd nest-client; \
     $BASEENV/bin/python3 setup.py install --user;

# Install nestml
WORKDIR $PACKAGES
RUN $BASEENV/bin/pip3 install setuptools; \
    git clone https://github.com/nest/nestml.git; \
    cd nestml; \
    $BASEENV/bin/python setup.py install --user;

# Other (large) packages added for scientific work:
RUN $BASEENV/bin/pip3 install plotly
#RUN $BASEENV/bin/pip3 install torch
#RUN $BASEENV/bin/pip3 install sbi
# RUN $BASEENV/bin/pip3 install sbibm

# Install TVB specific packages
WORKDIR $PACKAGES
RUN $BASEENV/bin/pip3 install pandas xarray pytest pytest-cov pytest-benchmark scikit-learn sqlalchemy numba dill; \
    $BASEENV/bin/pip3 install --ignore-installed entrypoints; \
    $BASEENV/bin/pip3 install werkzeug==2.0.1;

# Install TVB-multiscale specific packages
WORKDIR $PACKAGES
# Install NEURON and NetPyNE
RUN $BASEENV/bin/pip3 install neuron
RUN $BASEENV/bin/pip3 install netpyne
# Install ANNarchy
RUN $BASEENV/bin/pip3 install ANNarchy;
# Install pyNN
# RUN pip install pyNN

# Install PySpike
WORKDIR $PACKAGES
RUN git clone --depth 1 https://github.com/mariomulansky/PySpike.git; \
    cd PySpike; \
    $BASEENV/bin/python3 setup.py build_ext install --user;
# Install Elephant
WORKDIR $PACKAGES
RUN $BASEENV/bin/pip3 install elephant;

# Clone and install tvb-root
## TODO: Use Pypi for tvb-framework!!!
## First install tvb-framework from Pypi:
#RUN $BASEENV/bin/pip3 install tvb-framework
## Then, overwrite the tvb-library installation, and add the rest of the necessary tvb packages:
# For the moment though, let's try to install some dependencies of tvb-framework:
RUN $BASEENV/bin/pip3 install alembic;
RUN $BASEENV/bin/pip3 install bctpy;
RUN $BASEENV/bin/pip3 install cherrypy;
RUN $BASEENV/bin/pip3 install flask;
RUN $BASEENV/bin/pip3 install flask-restx;
RUN $BASEENV/bin/pip3 install formencode;
RUN $BASEENV/bin/pip3 install gevent;
RUN $BASEENV/bin/pip3 install python-keycloak;
RUN $BASEENV/bin/pip3 install google.auth;
RUN $BASEENV/bin/pip3 install oauthlib;
RUN $BASEENV/bin/pip3 install requests-oauthlib;
WORKDIR $PACKAGES
RUN git clone --depth 1 --no-single-branch https://github.com/the-virtual-brain/tvb-root.git;
ARG LAST_SHA_TVB=LATEST
RUN cd tvb-root; \
    git pull --allow-unrelated-histories; \
    cd tvb_framework; \
    $BASEENV/bin/python setup.py develop --user; \
    cd ../tvb_library; \
    $BASEENV/bin/python setup.py develop --user; \
    cd ../tvb_contrib; \
    $BASEENV/bin/python setup.py develop --user;

# Clone and install tvb-multiscale
WORKDIR $PACKAGES
RUN git clone --depth 1 --no-single-branch https://github.com/the-virtual-brain/tvb-multiscale.git;
ARG LAST_SHA_MULTISCALE=LATEST
RUN cd tvb-multiscale; \
    git pull origin --allow-unrelated-histories; \
    $BASEENV/bin/python setup.py develop --no-deps --user;

# Install Jupyter notebook specific packages
# TODO: Resolve the need for notebook version 6.4.12
RUN $BASEENV/bin/pip3 install jupyterlab notebook==6.4.12 ipympl jupyter_contrib_nbextensions --user;
RUN $LOCALENV/bin/jupyter-contrib nbextension install --user;
RUN $LOCALENV/bin/jupyter-nbextension enable --py widgetsnbextension;  # --sys-prefix
# Due to this:
# https://github.com/microsoft/azuredatastudio/issues/24436
# we need to downgrade traitlets:
RUN $BASEENV/bin/pip3 install traitlets==5.9.0

# TODO: UNDO THIS!
# Downgrade matplotlib as a temporary fix for error:
#File /usr/lib/python3/dist-packages/mpl_toolkits/mplot3d/axes3d.py:25
#     23 import matplotlib.collections as mcoll
#     24 import matplotlib.colors as mcolors
#---> 25 import matplotlib.docstring as docstring
#     26 import matplotlib.scale as mscale
#     27 import matplotlib.transforms as mtransforms
#
#ModuleNotFoundError: No module named 'matplotlib.docstring'
RUN $BASEENV/bin/pip3 install matplotlib==3.5.3  # The version required by tvb-framework

# Checkout specific branch of tvb-multiscale
WORKDIR $PACKAGES
RUN cd tvb-multiscale; \
    git checkout nrp; \
    git pull --allow-unrelated-histories;

# Compile tvb-netpyne .mod files
WORKDIR $PACKAGES
RUN cd tvb-multiscale; \
    $LOCALENV/bin/nrnivmodl tvb_multiscale/tvb_netpyne/netpyne/mod;

WORKDIR $PACKAGES

# Install ray:
RUN $BASEENV/bin/pip3 install -U "ray"

# Activate the right python env.
ENV NEST_INSTALL_DIR=${NEST_INSTALL_DIR}
ENV NEST_PYTHON_PREFIX=${NEST_PYTHON_PREFIX}
ENV PYTHONPATH=$PYTHONPATH:${NEST_PYTHON_PREFIX}
ENV PATH="${NEST_INSTALL_DIR}/bin":${PATH};
# Note that the following are automatically reset if NEST is imported
# via a call to  tvb_nest.simulator_nest.nest_factory.load_nest()
ENV NEST_DATA_DIR=${NEST_INSTALL_DIR}/share/nest
ENV SLI_PATH=${NEST_INSTALL_DIR}/share/nest/sli
ENV NEST_MODULE_PATH=${NEST_MODULE_PATH}
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NEST_MODULE_PATH}
ENV NEST_DOC_DIR=${NEST_INSTALL_DIR}/share/doc/nest

# Some configurations for NEST server:
RUN mkdir $HOME/tmp;
ENV NEST_SERVER_LOGFILE=$HOME/tmp/nest-server.log
ENV NEST_SERVER_STDOUT=1

# ----------------------------------------------------------------------------------------------------------------------
# Run the following lines in case you want to install your own NEST module with NEST builders
# The commented lines will install modules in the MODULES_LIST upon building this image
## Install local tvb-multiscale:
#WORKDIR $PACKAGES
#RUN mkdir tvb-multiscale
#COPY ./tvb-multiscale/ tvb-multiscale/
#RUN cd tvb-multiscale; $BASEENV/bin/python3 setup.py develop --no-deps
ENV MYMODULES_DIR=$PACKAGES/tvb-multiscale/tvb_multiscale/tvb_nest/nest/modules
ENV MYMODULES_BLD_DIR=$BUILD/nest_modules_builds
RUN cp -r ${MYMODULES_DIR} ${MYMODULES_BLD_DIR}
#ARG MYMODULES_LIST="izhikevich_hamker iaf_cond_ww_deco"
#ARG MYMODULES_DIR=$PACKAGES/tvb-multiscale/tvb_multiscale/tvb_nest/nest/modules
#ARG NEST_CONFIG=${NEST_INSTALL_DIR}/bin/nest-config
#RUN for MYMODULE_NAME in $MYMODULES_LIST; do export MYMODULE_DIR=${MYMODULES_DIR}/${MYMODULE_NAME}; \
#        export MYMODULE_BLD=${MYMODULES_BLD_DIR}/${MYMODULE_NAME}module_bld; \
#        mkdir -p ${MYMODULE_BLD}; cd ${MYMODULE_BLD}; \
#            cmake -Dwith-nest=$NEST_CONFIG ${MYMODULE_DIR}; \
#            make; make install; \
#    done
# ----------------------------------------------------------------------------------------------------------------------

# Final configurations
# tvb42
RUN $LOCALENV/bin/jupyter-notebook --generate-config; \
    echo "c.NotebookApp.password='sha1:12bff019c253:9daecd92c2e9bdb10b3b8a06767a74a0fe078d7c'">>/home/docker/.jupyter/jupyter_notebook_config.py

WORKDIR /home/docker/

# For running examples
# CMD $BASEENV/bin/python /home/docker/packages/tvb-multiscale/docker/installation_test.py
# CMD $BASEENV/bin/python /home/docker/packages/tvb-multiscale/examples/tvb_nest/basal_ganglia_izhikevich.py.py
# CMD $BASEENV/bin/python /home/docker/packages/tvb-multiscale/examples/tvb_nest/example.py

# Test NEST installation
# RUN cd $NEST; make installcheck
# CMD $BASEENV/bin/python /home/docker/packages/tvb-multiscale/docker/module_build_test.py

# pynestml installation test:
# CMD cd /home/docker/packages/nestml; $LOCALENV/bin/pytest setup.py test

# pyNN-NEST installation test:
# CMD $LOCALENV/bin/python -c "import pyNN.nest as sim; sim.setup(); sim.end(); print(sim)";

#For running tests
#CMD $LOCALENV/bin/pytest /home/docker/packages/tvb-multiscale/tests

#For running tests with coverage
#CMD $LOCALENV/bin/pytest --cov=nest_tests_coverage /home/docker/packages/tvb-multiscale/tests

#Run jupyter notebooks
CMD $LOCALENV/bin/jupyter-notebook --ip 0.0.0.0 --no-browser --allow-root

# CMD /bin/bash