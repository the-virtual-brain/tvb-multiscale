#
# TVB & NEST, ANNarchy and NetPyNE (NEURON) simulations environment
#
# Install Python 3.12, NEST, ANNarchy, NEURON, NetPyNE, Elephant, PySpike, ray, and then tvb-root.
# Finally link tvb-multiscale from Github sources
#

# Ubuntu Noble 24.04 LTS from NeuroDebian 2025:
# https://hub.docker.com/layers/library/neurodebian/noble/images/sha256-2f888be4d7159ed9a03884f72b0a18d428983c3c093c51acf701c4f3a8f58db7
# https://hub.docker.com/layers/library/neurodebian/nd22.04/images/sha256-be7132784be260e212cf3adfa3dbcfad197e3b374d64199b6ad7f9437824f973?context=explore
FROM neurodebian:noble

ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8

# TODO: Confirm this change:
#Package python-dev is not available, but is referred to by another package.
#This may mean that the package is missing, has been obsoleted, or
#is only available from another source
#However the following packages replace it:
#  python2-dev python2 python-dev-is-python3
RUN apt-get update; apt-get install -y software-properties-common;
RUN add-apt-repository ppa:deadsnakes/ppa;
RUN apt-get update; apt-get install -y automake libtool build-essential git vim octave \
                       ca-certificates locales tzdata \
                       wget python3 libpython3.12 libpython3.12-dev libncurses5-dev libreadline-dev libgsl0-dev cython3 \
                       python3-venv python3-dev python3-numpy python3-scipy python3-matplotlib python3-pip python3-nose \
                       ipython3 python3.12-venv python3-virtualenv python3-pyqt5 python-dev-is-python3 libgsl-dev \
                       libboost-dev  libncurses-dev libltdl-dev libopenmpi-dev pkg-config nano libhdf5-dev swig \
                       python3-pkgconfig subversion cmake ccache zip unzip make gcc g++ pandoc; \
    apt-get clean

#RUN wget -q http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz \
#    && tar xf mpich-3.1.4.tar.gz \
#    && cd mpich-3.1.4 \
#    && ./configure --disable-fortran --enable-fast=all,O3 --prefix=/usr \
#    && make -j$(nproc) \
#    && make install \
#    && ldconfig \
#    && cd .. \
#    && rm -rf mpich-3.1.4 \
#    && rm mpich-3.1.4.tar.gz

# Create user different than root
RUN useradd -ms /bin/bash docker
USER docker
ARG HOME=/home/docker
RUN mkdir $HOME/packages
RUN mkdir $HOME/build
ENV PACKAGES=$HOME/packages
ENV BUILD=$HOME/build
ENV VENV=$HOME/venv
ENV PYTHONPATH=$PYTHONPATH:$VENV
ENV PATH="${VENV}/bin":${PATH};

# Basic python dependencies:
RUN python3.12 -m venv $VENV && python3.12 -m venv --system-site-packages $VENV;
RUN $VENV/bin/pip3 install --upgrade pip wheel "packaging>=24.2"; \
    $VENV/bin/pip3 install --no-cache-dir Cython numpy boost setuptools gsl meson-python ninja; \
    $VENV/bin/pip3 install -v --no-cache-dir --no-build-isolation --force-reinstall --check-build-dependencies --no-binary pygsl pygsl; \
    $VENV/bin/pip3 install --upgrade nose ipython  scipy matplotlib; \
    $VENV/bin/pip3 install parameters quantities neo "django<1.9" django-tagging future hgapi gitpython sumatra nixio numba gunicorn;

# TODO: Find out if this is needed and whether the path to libpython3.12.so is correct irrespective of architecture:
# Install from sources libneurosim (NEST boost dependency)
WORKDIR $PACKAGES
RUN git clone --depth 1 https://github.com/INCF/libneurosim.git; \
    cd libneurosim; ./autogen.sh;
WORKDIR $BUILD
RUN mkdir libneurosim; \
    cd libneurosim; \
    PYTHON=$VENV/bin/python3 $PACKAGES/libneurosim/configure --prefix=$VENV; \
    make; make install; ls $VENV/lib $VENV/include;
RUN export LIBPYTHONPATH=$($VENV/bin/python3 -c "import glob; print(glob.glob('/usr/lib/python3.12/config*/libpython3.12.so')[0])"); \
    echo LIBPYTHONPATH=$LIBPYTHONPATH; \
    ln -s $LIBPYTHONPATH $VENV/lib/; \
    ls  $VENV/lib;

# Install NEST, latest stable version 3.9
ARG NEST_VER=3.10
ARG NEST=nest-simulator
ARG NEST_SRC=$PACKAGES/$NEST
ARG NEST_INSTALL_DIR=$BUILD/nest
ARG NEST_PYTHON_PREFIX=${NEST_INSTALL_DIR}/lib/python3.12/site-packages
ARG NEST_MODULE_PATH=${NEST_INSTALL_DIR}/lib/nest

# Install NEST:
# https://github.com/nest/nest-simulator/commit/a32cd8728fbb100cc1ce2bc7955196b2bd15403c
WORKDIR $PACKAGES
# ARG LAST_SHA_NEST=LATEST
RUN git clone https://github.com/nest/nest-simulator.git
RUN cd $NEST; \
    git fetch origin && git checkout a32cd8728fbb100cc1ce2bc7955196b2bd15403c; \
    cmake -DCMAKE_INSTALL_PREFIX=${NEST_INSTALL_DIR} \
          -Dwith-boost=ON \
          -Dwith-ltdl=ON \
          -Dwith-gsl=ON \
          -Dwith-readline=ON \
          -Dwith-mpi=OFF  \
          -Dwith-openmp=ON \
          -Dwith-libneurosim=$VENV \
          -DPYTHON_LIBRARY=$VENV/lib \
          -Dwith-python=ON \
          -DPYTHON_INCLUDE_DIR=/usr/include/python3.12m \
          -DPYTHON_INCLUDE_DIR2=$VENV/include/python3.12 \
          ${NEST_SRC}; \
    make; make install

# Install nestml and NEST Desktop
RUN $VENV/bin/pip3 install --target d:$PACKAGES nestml nest-desktop

# Install NESTServerClient
RUN $VENV/bin/pip3 install flask_cors RestrictedPython;
WORKDIR $PACKAGES
RUN git clone https://github.com/nest/nest-client.git
RUN cd nest-client; \
     $VENV/bin/python3 setup.py install;

# Other (large) packages added for scientific work:
RUN $VENV/bin/pip3 install plotly
#RUN $VENV/bin/pip3 install torch
#RUN $VENV/bin/pip3 install sbi
# RUN $VENV/bin/pip3 install sbibm

# Install TVB-multiscale specific packages
WORKDIR $PACKAGES
# Install NEURON and NetPyNE
RUN $VENV/bin/pip3 install neuron
RUN $VENV/bin/pip3 install netpyne
# Install ANNarchy
RUN $VENV/bin/pip3 install ANNarchy;
# Install pyNN
# RUN pip install pyNN
# Install Elephant
WORKDIR $PACKAGES
RUN $VENV/bin/pip3 install elephant;
# Install Pyspike:
# Only via github:
# https://github.com/mariomulansky/PySpike/issues/82#issuecomment-2838013901
WORKDIR $PACKAGES
RUN git clone https://github.com/mariomulansky/PySpike.git;
RUN cd PySpike && \
    mv pyproject.toml pyproject.toml.bak && \
    $VENV/bin/python setup.py build_ext --inplace
ENV PYTHONPATH=$PYTHONPATH:$PACKAGES/PySpike

# Install TVB specific packages
WORKDIR $PACKAGES
RUN $VENV/bin/pip3 install pandas xarray pytest pytest-cov pytest-benchmark scikit-learn sqlalchemy numba dill; \
    $VENV/bin/pip3 install --ignore-installed entrypoints; \
    $VENV/bin/pip3 install werkzeug;

# Install some additional packages for TVB:
# For tvb-library:
RUN $VENV/bin/pip3 install deprecated autopep8 mako>=1.1.4 networkx numexpr cryptography pyaescrypt kubernetes;
# RUN $VENV/bin/pip3 install syncrypto
# for tvb-framework:
RUN $VENV/bin/pip3 install alembic bctpy cherrypy flask-restx formencode gevent nibabel;
RUN $VENV/bin/pip3 install python-keycloak requests-toolbelt>=0.10 siibra simplejson;
RUN $VENV/bin/pip3 install tvb-data tvb-gdist;

# Clone and install tvb-root
WORKDIR $PACKAGES
RUN git clone --depth 1 --no-single-branch https://github.com/the-virtual-brain/tvb-root.git;
ARG LAST_SHA_TVB=LATEST
RUN cd tvb-root; \
    git pull --allow-unrelated-histories; \
    cd tvb_framework; \
    $VENV/bin/pip3 install -e . --no-deps; \
    cd ../tvb_storage; \
    $VENV/bin/pip3 install -e . --no-deps; \
    cd ../tvb_library; \
    $VENV/bin/pip3 install -e . --no-deps; \
    cd ../tvb_contrib; \
    $VENV/bin/pip3 install -e . --no-deps; \
    cd ../tvb_bin; \
    $VENV/bin/pip3 install -e .; \
    cd ../tvb_build; \
    $VENV/bin/pip3 install -e . --no-deps;

# Clone and install tvb-multiscale
WORKDIR $PACKAGES
RUN git clone --depth 1 --no-single-branch https://github.com/the-virtual-brain/tvb-multiscale.git;
ARG LAST_SHA_MULTISCALE=LATEST
RUN cd tvb-multiscale; \
    git pull origin --allow-unrelated-histories; \
    $VENV/bin/python3 setup.py develop --no-deps;

WORKDIR $PACKAGES

# Install ray:
RUN $VENV/bin/pip3 install -U "ray"

# Install Jupyter notebook specific packages
RUN $VENV/bin/pip3 install jupyterlab notebook ipympl;

# TODO: Find how to solve the mpl_toolkits warning problem:
# RUN $VENV/bin/pip3 install --upgrade matplotlib

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
#RUN cd tvb-multiscale; $VENV/bin/python3 setup.py develop --no-deps
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
RUN $VENV/bin/jupyter-notebook --generate-config; \
    echo "c.NotebookApp.password='sha1:12bff019c253:9daecd92c2e9bdb10b3b8a06767a74a0fe078d7c'">>/home/docker/.jupyter/jupyter_notebook_config.py

WORKDIR /home/docker/

# For running examples
# CMD $VENV/bin/python3 /home/docker/packages/tvb-multiscale/docker/installation_test.py
# CMD $VENV/bin/python3 /home/docker/packages/tvb-multiscale/examples/tvb_nest/basal_ganglia_izhikevich.py.py
# CMD $VENV/bin/python3 /home/docker/packages/tvb-multiscale/examples/tvb_nest/example.py

# Test NEST installation
# RUN cd $NEST; make installcheck
# CMD $VENV/bin/python3 /home/docker/packages/tvb-multiscale/docker/module_build_test.py

# pynestml installation test:
# CMD cd /home/docker/packages/nestml; $VENV/bin/pytest setup.py test

# pyNN-NEST installation test:
# CMD $VENV/bin/python3 -c "import pyNN.nest as sim; sim.setup(); sim.end(); print(sim)";

# For running tests
# CMD $VENV/bin/pytest /home/docker/packages/tvb-multiscale/tests

# For running tests with coverage
# CMD $VENV/bin/pytest --cov=nest_tests_coverage /home/docker/packages/tvb-multiscale/tests

# Run jupyter lab
# CMD $VENV/bin/jupyter-lab --ip 0.0.0.0 --no-browser --allow-root

#Run jupyter notebooks
CMD $VENV/bin/jupyter-notebook --ip 0.0.0.0 --no-browser --allow-root

# CMD /bin/bash
