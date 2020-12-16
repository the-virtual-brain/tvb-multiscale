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

USER root
RUN apt-get update; apt-get install -y automake libtool build-essential git vim octave \
                       wget python3.7 libpython3.7 libpython3-dev libncurses5-dev libreadline-dev libgsl0-dev cython3 \
                       python3.7-dev python3-pip python3-numpy python3-scipy python3-matplotlib python3-nose \
                       ipython3 python3-venv python-virtualenv python3-pyqt5 python3-dev libgsl-dev libncurses-dev \
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
ENV CONDAENV_NAME nestenv
ENV CONDAENV /opt/conda/envs/nestenv
ENV CONDAENV_PYTHON /opt/conda/envs/nestenv/bin/python
ENV CONDAENV_PIP /opt/conda/envs/nestenv/bin/pip

RUN conda create -n $CONDAENV_NAME python=3
RUN conda install --name $CONDAENV_NAME cython setuptools
RUN $CONDAENV_PIP install parameters quantities neo "django<1.9" django-tagging future hgapi gitpython sumatra nixio
RUN $CONDAENV_PIP install --upgrade nose ipython
# django<1.9 is needed for pynest - september 2019 - NEST 2.18

# Install from sources libneurosim (NEST boost dependency)
WORKDIR $HOME/packages
RUN git clone --depth 1 https://github.com/INCF/libneurosim.git; \
    cd libneurosim; ./autogen.sh; \
    mkdir $VENV/build
WORKDIR $VENV/build
RUN mkdir libneurosim; \
    cd libneurosim; \
    PYTHON=$CONDAENV_PYTHON $HOME/packages/libneurosim/configure --prefix=$VENV; \
    make; make install; ls $VENV/lib $VENV/include; \
    ln -s /usr/lib/python3.7/config-x86_64-linux-gnu/libpython3.7.so $VENV/lib/


# Install NEST, latest stable version (July 2019)
ENV NEST_VER=2.20.0
ENV NEST=nest-simulator-$NEST_VER
ENV NEST_SRC=$HOME/packages/$NEST
ENV NEST_INSTALL_DIR=$VENV/nest_build
ENV NEST_PYTHON_PREFIX=${NEST_INSTALL_DIR}/lib/python3.7/site-packages
ENV NEST_MODULE_PATH=${NEST_INSTALL_DIR}/lib/nest

WORKDIR $HOME/packages
RUN wget  https://github.com/nest/nest-simulator/archive/v2.20.0.tar.gz -O $HOME/packages/$NEST.tar.gz; \
    tar xzf $NEST.tar.gz; rm $NEST.tar.gz; \
    cd $NEST; \
    cmake -DCMAKE_INSTALL_PREFIX=$NEST_INSTALL_DIR \
    -Dwith-python=3 \
    -Dwith-libneurosim=$VENV \
    -DPYTHON_EXECUTABLE=$CONDAENV/bin/python3 \
    -DPYTHON_LIBRARY=$CONDAENV/lib \
    -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.8 \
    $NEST_SRC; \
    make; make install

# Install nestml
RUN git clone --depth 1 https://github.com/nest/nestml.git; \
    cd nestml; \
    $CONDAENV_PYTHON setup.py install

# Install TVB specific packages
WORKDIR $HOME/packages
RUN $CONDAENV_PIP install pandas xarray pytest pytest-cov pytest-benchmark oct2py scikit-learn scipy; \
    $CONDAENV_PIP install --ignore-installed entrypoints

RUN wget https://zenodo.org/record/3688773/files/tvb-data.zip?download=1 -O tvb_data.zip; \
    mkdir tvb_data; unzip tvb_data.zip -d tvb_data; rm tvb_data.zip; \
    cd tvb_data; \
    $CONDAENV_PYTHON setup.py develop

RUN git clone --depth 1 --branch tvb-multiscale-cosim https://github.com/the-virtual-brain/tvb-root.git
ARG LAST_SHA_ROOT=LATEST
RUN cd tvb-root; \
    git pull origin tvb-multiscale-cosim --allow-unrelated-histories; \
    cd scientific_library; \
    $CONDAENV_PYTHON setup.py develop;

RUN git clone --depth 1 https://github.com/the-virtual-brain/tvb-multiscale.git
ARG LAST_SHA_MULTISCALE=LATEST
RUN cd tvb-multiscale; \
    git pull; \
    $CONDAENV_PYTHON setup.py develop --no-deps;

RUN mkdir -m777 /home/jovyan/.tvb-temp /home/jovyan/.local
RUN mkdir -m777 /home/jovyan/.local/share
RUN mkdir -m777 /home/jovyan/.local/share/jupyter
RUN mkdir -m777 /home/jovyan/.local/share/jupyter/runtime

# Register conda env python as jupyerhub kernel
RUN $CONDAENV_PIP install --user ipykernel
RUN $CONDAENV_PYTHON -m ipykernel install --user

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
ENV MYMODULES_DIR=$HOME/packages/tvb-multiscale/tvb_nest/nest/modules
ENV MYMODULES_BLD_DIR=$HOME/packages/nest_modules_builds
RUN cp -r ${MYMODULES_DIR} ${MYMODULES_BLD_DIR}

ARG WORK_DIR=/home/jovyan/packages/notebooks
ARG TVB_NEST_Examples=/home/jovyan/packages/notebooks/TVB-NEST-Examples

RUN mkdir -m777 $WORK_DIR
RUN mkdir -m777 $TVB_NEST_Examples
RUN cp -r /home/jovyan/packages/tvb-multiscale/docs/documented_example_notebook/* $TVB_NEST_Examples
RUN chmod -R 777 $WORK_DIR
RUN chmod -R 777 $TVB_NEST_Examples
RUN chmod -R 777 /home/jovyan/packages/tvb-multiscale/tvb_nest/examples
RUN chown -R jovyan: $TVB_NEST_Examples
RUN chown -R jovyan: /home/jovyan/packages/tvb-multiscale/tvb_nest/examples

WORKDIR $WORK_DIR
# CMD is defined in the base image: jupyter/minimal-notebook