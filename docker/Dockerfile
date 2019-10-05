#
# TVB & NEST simulations environment
#
# Install Python 3.7, NEST, then tvb-library from Pypi
# Finally link tvb-scripts and tvb-nest from Github sources
#

FROM neurodebian:buster
MAINTAINER lia.domide@codemart.ro

ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8

RUN apt-get update; apt-get install -y automake libtool build-essential openmpi-bin libopenmpi-dev git vim octave \
                       wget python3.7 libpython3.7 libpython3-dev libncurses5-dev libreadline-dev libgsl0-dev cython3 \
                       python3.7-dev python3-pip python3-numpy python3-scipy python3-matplotlib python3-nose \
                       ipython3 python3-venv python-virtualenv python3-pyqt5 python-dev libgsl-dev libncurses-dev subversion python3-mpi4py cmake; \
                    apt-get clean

# Create user different than root
RUN useradd -ms /bin/bash docker
USER docker
ARG HOME=/home/docker
RUN mkdir $HOME/env; mkdir $HOME/packages

# Create a dedicated Python env for working in it
ENV VENV=$HOME/env/neurosci
RUN python3 -m venv $VENV && python3 -m venv --system-site-packages $VENV; \
    $VENV/bin/pip3 install --upgrade pip; \
    $VENV/bin/pip3 install parameters quantities neo "django<1.9" django-tagging future hgapi gitpython sumatra nixio; \
    $VENV/bin/pip3 install --upgrade nose ipython
# django<1.9 is needed for pynest - september 2019 - NEST 2.18

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
    ln -s /usr/lib/python3.7/config-x86_64-linux-gnu/libpython3.7.so $VENV/lib/

# Install NEST, latest stable version (July 2019)
ENV NEST_VER=2.18.0
ENV NEST=nest-simulator-$NEST_VER

WORKDIR $HOME/packages
RUN wget  https://github.com/nest/nest-simulator/archive/v$NEST_VER.tar.gz -O $HOME/packages/$NEST.tar.gz; \
    tar xzf $NEST.tar.gz; rm $NEST.tar.gz; \
    cd $NEST; \
    cmake -DCMAKE_INSTALL_PREFIX=$VENV \
          -Dwith-mpi=ON  \
          -Dwith-python=3 \
          ###-Dwith-music=ON \
          -Dwith-libneurosim=$VENV \
          -DPYTHON_LIBRARY=/usr/lib \
          -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m \
          $HOME/packages/$NEST; \
    make; make install
# RUN cd $NEST; make installcheck

# Install TVB specific packages
WORKDIR $HOME/packages

RUN $VENV/bin/pip3 install jupyterlab pytest pytest-cov pytest-benchmark oct2py scikit-learn scipy; \
    $VENV/bin/pip3 install --ignore-installed entrypoints

RUN git clone --depth 1 https://github.com/the-virtual-brain/tvb-library.git; \
    cd tvb-library; git pull origin neotraits_py3; \
    $VENV/bin/python setup.py develop

RUN git clone --depth 1 https://github.com/the-virtual-brain/tvb-data.git; \
    cd tvb-data; \
    $VENV/bin/python setup.py develop

RUN git clone --depth 1 https://github.com/the-virtual-brain/tvb-multiscale.git; \
    cd tvb-multiscale; \
    $VENV/bin/python setup.py develop

# Final configurations
# tvb42
RUN $VENV/bin/jupyter notebook --generate-config; \
    echo "c.NotebookApp.password='sha1:12bff019c253:9daecd92c2e9bdb10b3b8a06767a74a0fe078d7c'">>/home/docker/.jupyter/jupyter_notebook_config.py

WORKDIR /home/docker/
COPY installation_test.py /home/docker/

# Activate the rights python env
ENV PYTHONPATH $VENV
ENV SLI_PATH $VENV
RUN echo "source $VENV/bin/activate" >> .bashrc

#CMD $VENV/bin/jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
#CMD $VENV/bin/python installation_test.py

#For running tests
CMD cd /home/docker/packages/tvb-multiscale; $VENV/bin/pytest tvb_nest/tests

#For running tests with coverage
#CMD $VENV/bin/pytest --cov /home/docker/packages/tvb-multiscale/tvb_nest/tests

#srun -C mc --time 01:00:00 --partition=normal
#       shifter run --mount=type=bind,source=$HOME,destination=$HOME thevirtualbrain/tvb-nest /home/docker/env/neurosci/bin/python installation_test.py

#docker run -d -p 8888:8888 thevirtualbrain/tvb-nest /home/docker/env/neurosci/bin/jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

#docker run thevirtualbrain/tvb-nest -v /WORK/TVB/VBC/tvb-multiscale:/home/docker/packages/tvb-multiscale
#    /home/docker/env/neurosci/bin/python /home/docker/packages/tvb-multiscale/tvb_nest/examples/red_ww_exc_io_inh_i.py