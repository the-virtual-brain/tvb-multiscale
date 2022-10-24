#
# TVB & NEST simulations environment
#
# Prepare conda env, install NEST, then tvb-library from Pypi
# Finally link tvb-scripts and tvb-nest from Github sources
#

FROM jupyter/minimal-notebook
MAINTAINER lia.domide@codemart.ro

ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8

ARG a
USER root
RUN apt-get update; apt-get install -y automake libtool build-essential git vim octave \
                       wget python3.7 libpython3.7 libpython3-dev libncurses5-dev libreadline-dev libgsl0-dev cython3 \
                       python3-pip python3-numpy python3-scipy python3-matplotlib python3-nose \
                       ipython3 python3-venv python3-pyqt5 python3-dev libgsl-dev libncurses-dev \
                       subversion cmake zip unzip; \
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

RUN mkdir -m777 $HOME/env; mkdir -m777 $HOME/packages

USER $NB_UID
# Create a dedicated Python env for working in it
ENV VENV=$HOME/env/neurosci
RUN mkdir -m777 $VENV

RUN conda install cython setuptools h5py
RUN pip install parameters quantities neo future hgapi gitpython sumatra nixio
RUN pip install --upgrade nose ipython

# Install NEST3, latest dev version (October 2020)
ENV NEST_VER=dev
ENV NEST=nest-simulator
ENV NEST_SRC=$HOME/packages/$NEST
ENV NEST_INSTALL_DIR=$VENV/nest_build
ENV NEST_PYTHON_PREFIX=${NEST_INSTALL_DIR}/lib/python3.7/site-packages
ENV NEST_MODULE_PATH=${NEST_INSTALL_DIR}/lib/nest

WORKDIR $HOME/packages

RUN wget https://zenodo.org/record/3688773/files/tvb-data.zip?download=1 -O tvb_data.zip; \
    mkdir tvb_data; unzip tvb_data.zip -d tvb_data; rm tvb_data.zip; \
    cd tvb_data; \
    python setup.py develop

RUN git clone https://github.com/nest/nest-simulator.git
RUN cd $NEST; \
    git fetch origin && git checkout 1e0ce51e17e48933a95463cc35a5c2c5d99ca366; \
    cmake -DCMAKE_INSTALL_PREFIX=$NEST_INSTALL_DIR \
    -Dwith-mpi=ON  \
#    -Dwith-libneurosim=$VENV \
#    -DPYTHON_EXECUTABLE=$CONDAENV/bin/python3 \
#    -DPYTHON_LIBRARY=/usr/lib \
#    -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m \
    $NEST_SRC; \
    make; make install

USER root
RUN apt-get update; apt-get install -y gfortran # lvm-7-dev
RUN chown $NB_UID:$NB_UID -R $HOME/packages
USER $NB_UID

# Install TVB specific packages
WORKDIR $HOME/packages
RUN pip install pandas xarray pytest pytest-cov pytest-benchmark oct2py elephant sqlalchemy; \
    pip install --ignore-installed entrypoints; \
    pip install scikit-learn scipy
RUN conda install -y matplotlib==3.2.2
RUN conda install -y numba

WORKDIR $HOME/packages
RUN git clone --depth 1 https://github.com/the-virtual-brain/tvb-root.git
ARG LAST_SHA_ROOT=LATEST
RUN cd tvb-root; \
    cd tvb_library; \
    python setup.py develop; \
    cd ../tvb_framework; \
    python setup.py develop; \
    cd ../tvb_contrib; \
    python setup.py develop;

WORKDIR $HOME/packages
# Install nestml
RUN git clone --depth 1 https://github.com/nest/nestml.git; \
    cd nestml; \
    python setup.py install

RUN git clone --depth 1 https://github.com/the-virtual-brain/tvb-multiscale.git
ARG LAST_SHA_MULTISCALE=LATEST
RUN cd tvb-multiscale; \
    git pull; \
    python setup.py develop --no-deps;

RUN mkdir -m777 /home/jovyan/.tvb-temp /home/jovyan/.local
RUN mkdir -m777 /home/jovyan/.local/share
RUN mkdir -m777 /home/jovyan/.local/share/jupyter
RUN mkdir -m777 /home/jovyan/.local/share/jupyter/runtime

# Register conda env python as jupyerhub kernel
#RUN pip uninstall ipykernel
#RUN pip install --user ipykernel
#RUN python -m ipykernel install --user

WORKDIR $HOME

ENV NEST_INSTALL_DIR=${NEST_INSTALL_DIR}
ENV NEST_PYTHON_PREFIX=${NEST_INSTALL_DIR}/lib/python3.8/site-packages
ENV PYTHONPATH=$PYTHONPATH:$VENV:${NEST_PYTHON_PREFIX}
# Note that the following are automatically reset if NEST is imported
# via a call to  tvb_nest.simulator_nest.nest_factory.load_nest()
ENV NEST_DATA_DIR=${NEST_INSTALL_DIR}/share/nest
ENV SLI_PATH=${NEST_INSTALL_DIR}/share/nest/sli
ENV NEST_MODULE_PATH=${NEST_MODULE_PATH}
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NEST_MODULE_PATH}
ENV NEST_DOC_DIR=${NEST_INSTALL_DIR}/share/doc/nest

## Install local tvb-multiscale:
ENV MYMODULES_DIR=$HOME/packages/tvb-multiscale/tvb_multiscale/tvb_nest/nest/modules
ENV MYMODULES_BLD_DIR=$HOME/packages/nest_modules_builds
RUN cp -r ${MYMODULES_DIR} ${MYMODULES_BLD_DIR}

ARG WORK_DIR=/home/jovyan/packages/notebooks
ARG TVB_NEST_Examples=/home/jovyan/packages/notebooks/TVB-NEST-Examples

RUN mkdir -m777 $WORK_DIR
RUN mkdir -m777 $TVB_NEST_Examples
RUN cp -r /home/jovyan/packages/tvb-multiscale/docs/documented_example_notebooks $TVB_NEST_Examples
RUN cp -r /home/jovyan/packages/tvb-multiscale/docs/VirtualCourse_2021-01-07 $TVB_NEST_Examples
RUN cp -r /home/jovyan/packages/tvb-multiscale/examples $TVB_NEST_Examples
RUN chmod -R 777 $WORK_DIR
RUN chmod -R 777 $MYMODULES_BLD_DIR
RUN chmod -R 777 $MYMODULES_DIR
RUN chmod -R 777 $TVB_NEST_Examples
RUN chmod -R 777 $NEST_INSTALL_DIR
RUN chmod -R 777 $SLI_PATH
RUN chmod -R 777 $NEST_MODULE_PATH
RUN chmod -R 777 /home/jovyan/packages/tvb-multiscale/examples/tvb_nest
RUN chown -R jovyan: $TVB_NEST_Examples
RUN chown -R jovyan: $NEST_INSTALL_DIR
RUN chown -R jovyan: /home/jovyan/packages/tvb-multiscale/examples/tvb_nest

# Extend the output data rate limit for all images to be loaded in the notebooks
#RUN echo "c.NotebookApp.iopub_data_rate_limit = 10000000" >> /etc/jupyter/jupyter_notebook_config.py
#RUN echo "c.NotebookApp.trust_xheaders = True" >> /etc/jupyter/jupyter_notebook_config.py

WORKDIR $WORK_DIR
# CMD is defined in the base image: jupyter/minimal-notebook