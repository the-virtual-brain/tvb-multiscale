#
# TVB & NEST simulations environment
#
# Install Python 2.7, NEST, then tvb-library from Pypi
# Finally link tvb-scripts and tvb-nest from Github sources
#

FROM neurodebian:buster
MAINTAINER lia.domide@codemart.ro

ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8

RUN apt-get update; apt-get install -y automake libtool build-essential openmpi-bin libopenmpi-dev git vim  \
                       wget python libpython-dev libncurses5-dev libreadline-dev libgsl0-dev cython \
                       python-all-dev python-pip python-numpy python-scipy python-matplotlib python-nose \
                       python-virtualenv python-pyqt5 libgsl-dev libncurses-dev subversion python-mpi4py cmake; \
                    apt-get clean

# Create user different than root
RUN useradd -ms /bin/bash docker
USER docker
ARG HOME=/home/docker
RUN mkdir $HOME/env; mkdir $HOME/packages

# Create a dedicated Python env for working in it
ENV VENV=$HOME/env/neurosci
RUN virtualenv --system-site-packages $VENV; \
    $VENV/bin/pip install --upgrade pip; \
    $VENV/bin/pip install parameters quantities neo "django<1.9" django-tagging future hgapi gitpython sumatra nixio; \
    $VENV/bin/pip install --upgrade nose

# Install from sources libneurosim (NEST boost dependency)
WORKDIR $HOME/packages
RUN git clone --depth 1 https://github.com/INCF/libneurosim.git; \
    cd libneurosim; ./autogen.sh; \
    mkdir $VENV/build
WORKDIR $VENV/build
RUN mkdir libneurosim; \
    cd libneurosim; \
    PYTHON=$VENV/bin/python $HOME/packages/libneurosim/configure --prefix=$VENV; \
    make; make install; ls $VENV/lib $VENV/include; \
    ln -s /usr/lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so $VENV/lib/

# Install NEST, latest stable version (July 2019)
ARG NEST_VER=2.18.0
ARG NEST=nest-simulator-$NEST_VER
ARG NEST_SRC=$HOME/packages/$NEST
ARG NEST_INSTALL_DIR=$VENV/nest_build

WORKDIR $HOME/packages
RUN wget  https://github.com/nest/nest-simulator/archive/v$NEST_VER.tar.gz -O $HOME/packages/$NEST.tar.gz; \
    tar xzf $NEST.tar.gz; rm $NEST.tar.gz; \
    cd $NEST; \
    cmake -DCMAKE_INSTALL_PREFIX=${NEST_INSTALL_DIR} \
          -Dwith-mpi=ON  \
          ###-Dwith-music=ON \
          -Dwith-libneurosim=$VENV \
          -DPYTHON_LIBRARY=$VENV/lib/libpython2.7.so \
          -DPYTHON_INCLUDE_DIR=/usr/include/python2.7 \
          ${NEST_SRC}; \
    make; make install
# RUN cd $NEST; make installcheck

# Install TVB specific packages
WORKDIR $HOME/packages

RUN $VENV/bin/pip install --upgrade pip; \
    $VENV/bin/pip install tvb-library jupyterlab; \
    $VENV/bin/pip install --ignore-installed entrypoints; \
    $VENV/bin/pip uninstall -y tvb-data

RUN git clone --depth 1 https://github.com/the-virtual-brain/tvb-data.git; \
    cd tvb-data; \
    $VENV/bin/python setup.py develop

# Final configurations
# tvb42
RUN $VENV/bin/jupyter notebook --generate-config; \
    echo "c.NotebookApp.password='sha1:12bff019c253:9daecd92c2e9bdb10b3b8a06767a74a0fe078d7c'">>/home/docker/.jupyter/jupyter_notebook_config.py

RUN mkdir tvb-multiscale
COPY . tvb-multiscale/
RUN cd tvb-multiscale; $VENV/bin/python setup.py develop --no-deps

# Run the following lines in case you want to install your own NEST module with NEST models
ARG MYMODULE_NAME=iaf_cond_deco2014
ARG MYMODULE_DIR=$HOME/packages/tvb-multiscale/tvb_nest/simulator_nest/models/modules/${MYMODULE_NAME}module
ARG MYMODULE_BLD=${MYMODULE_NAME}module_bld
RUN mkdir ${MYMODULE_BLD}; cd ${MYMODULE_BLD}; \
    cmake -Dwith-nest=${NEST_INSTALL_DIR}/bin/nest-config ${MYMODULE_DIR}; \
    make; make install

WORKDIR /home/docker/
# Activate the rights python env
ENV PYTHONPATH=$PYTHONPATH:$VENV:${NEST_INSTALL_DIR}/lib/python2.7/site-packages
ENV SLI_PATH ${NEST_INSTALL_DIR}/share/nest/sli
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NEST_INSTALL_DIR}/lib/nest/
RUN echo "source ${VENV}/bin/activate" >> .bashrc

CMD $VENV/bin/jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
# CMD $VENV/bin/python /home/docker/packages/tvb-multiscale/docker/installation_test.py
# CMD $VENV/bin/python /home/docker/packages/tvb-multiscale/docker/module_build_test.py
# CMD $VENV/bin/python /home/docker/packages/tvb-multiscale/tvb_nest/examples/red_ww_exc_io_inh_i.py

#srun -C mc --time 01:00:00 --partition=normal
#       shifter run --mount=type=bind,source=$HOME,destination=$HOME thevirtualbrain/tvb-nest /home/docker/env/neurosci/bin/python installation_test.py

#docker run -d -p 8888:8888 thevirtualbrain/tvb-nest /home/docker/env/neurosci/bin/jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

#docker run thevirtualbrain/tvb-nest -v /WORK/TVB/VBC/tvb-multiscale:/home/docker/packages/tvb-multiscale
#    /home/docker/env/neurosci/bin/python /home/docker/packages/tvb-multiscale/tvb_nest/examples/red_ww_exc_io_inh_i.py