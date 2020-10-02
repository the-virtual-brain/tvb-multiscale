#
# TVB & NEST simulations environment
#
# Install Python 3.7, NEST3, then tvb-library from Pypi
# Finally link tvb-multiscale from Github sources
#

FROM neurodebian:buster

ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8

RUN apt-get update; apt-get install -y build-essential git vim wget cmake libtool zip unzip curl \
                       python3.7 python3-dev libncurses-dev libreadline-dev libgsl-dev cython3 \
                       python3-pip python3-numpy python3-scipy python3-matplotlib python3-nose \
                       ipython3 python3-venv python3-pyqt5; \
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
ARG ENV_DIR=$HOME/env
ARG PACKAGES_DIR=$HOME/packages
RUN mkdir $ENV_DIR; mkdir $PACKAGES_DIR; mkdir $PACKAGES_DIR/tvb-nest

# Create a dedicated Python env for working in it
ENV VENV=$ENV_DIR/neurosci
RUN python3 -m venv $VENV && python3 -m venv --system-site-packages $VENV; \
    $VENV/bin/pip3 install --upgrade pip; \
    $VENV/bin/pip3 install parameters quantities neo future hgapi sumatra nixio numba; \
    $VENV/bin/pip3 install --upgrade nose ipython

# Install from sources libneurosim (NEST boost dependency)
WORKDIR $PACKAGES_DIR
RUN git clone --depth 1 https://github.com/INCF/libneurosim.git; \
    cd libneurosim; ./autogen.sh; \
    mkdir $VENV/build
WORKDIR $VENV/build
RUN mkdir libneurosim; \
    cd libneurosim; \
    PYTHON=$VENV/bin/python $PACKAGES_DIR/libneurosim/configure --prefix=$VENV; \
    make; make install; ls $VENV/lib $VENV/include; \
    ln -s /usr/lib/python3.7/config-x86_64-linux-gnu/libpython3.7.so $VENV/lib/

# Install NEST3, dev version (September 2020)
ARG NEST=nest-simulator
ENV NEST_SRC=$PACKAGES_DIR/$NEST
ENV NEST_INSTALL_DIR=$VENV/nest_build
ENV NEST_PYTHON_PREFIX=${NEST_INSTALL_DIR}/lib/python3.7/site-packages
ENV NEST_MODULE_PATH=${NEST_INSTALL_DIR}/lib/nest

# WITH_LIBNEUROSIM could be OFF
ARG WITH_LIBNEUROSIM=$VENV

WORKDIR $PACKAGES_DIR
# For Lionel it will be a Forked repo
ARG NEST_REPO_URL=https://github.com/nest/nest-simulator.git
ARG BRANCH=master
RUN git clone --depth 1 --branch $BRANCH $NEST_REPO_URL; \
    cd $NEST; \
    cmake -DCMAKE_INSTALL_PREFIX=${NEST_INSTALL_DIR} \
          -Dwith-mpi=ON  \
          -Dwith-openmp=ON  \
          -Dwith-readline=ON  \
          -Dwith-ltdl=ON  \
          -Dwith-python=ON  \
          -Dcythonize-pynest=ON  \
          -Dwith-libneurosim=$WITH_LIBNEUROSIM \
          ${NEST_SRC}; \
    make; make install
RUN if [ "x$BRANCH" = "master" ] ; then cp -r $NEST_SRC/nestkernel/spatial $NEST_INSTALL_DIR/include/nest ; fi

# Install nestml
RUN $VENV/bin/pip3 install setuptools; \
    git clone --depth 1 https://github.com/nest/nestml.git; \
    cd nestml; \
    $VENV/bin/python setup.py install

# Install TVB specific packages
WORKDIR $PACKAGES_DIR
RUN $VENV/bin/pip3 install pandas xarray jupyterlab pytest pytest-cov pytest-benchmark oct2py scikit-learn scipy elephant sqlalchemy; \
    $VENV/bin/pip3 install --ignore-installed entrypoints numpy; \
    $VENV/bin/pip3 install mpi4py

RUN wget https://zenodo.org/record/3688773/files/tvb-data.zip?download=1 -O tvb_data.zip; \
    mkdir tvb_data; unzip tvb_data.zip -d tvb_data; rm tvb_data.zip; \
    cd tvb_data; \
    $VENV/bin/python setup.py develop

ARG TVB_ROOT_BRANCH=tvb-multiscale-cosim
RUN git clone --depth 1 --branch $TVB_ROOT_BRANCH https://github.com/the-virtual-brain/tvb-root.git
ARG LAST_SHA_ROOT=LATEST
RUN cd tvb-root; \
    cd scientific_library; \
    $VENV/bin/python setup.py develop; \
    cd ../framework_tvb; \
    $VENV/bin/python setup.py develop; \
    cd ../tvb_contrib; \
    $VENV/bin/python setup.py develop

USER root
RUN apt-get install -y octave
USER docker

# Final configurations
# tvb42
RUN $VENV/bin/jupyter notebook --generate-config; \
    echo "c.NotebookApp.password='sha1:12bff019c253:9daecd92c2e9bdb10b3b8a06767a74a0fe078d7c'">>/home/docker/.jupyter/jupyter_notebook_config.py

WORKDIR $HOME
# Activate the rights python env
ENV PYTHONPATH $NEST_PYTHON_PREFIX:$PACKAGES_DIR/tvb-nest:$VENV
ENV SLI_PATH $VENV
RUN echo "source $VENV/bin/activate" >> .bashrc

#Run jupyter notebooks
CMD $VENV/bin/jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

#CMD /bin/bash