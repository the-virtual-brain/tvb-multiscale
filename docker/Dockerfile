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
                       python-virtualenv ipython libgsl-dev libncurses-dev subversion python-mpi4py cmake; \
                    apt-get clean

RUN useradd -ms /bin/bash docker
USER docker
ENV HOME=/home/docker
RUN mkdir $HOME/env; mkdir $HOME/packages

# Create a dedicated Python env for working in it
ENV VENV=$HOME/env/neurosci
RUN virtualenv --system-site-packages $VENV
RUN $VENV/bin/pip install --upgrade pip; \
    $VENV/bin/pip install parameters quantities neo "django<1.9" django-tagging future hgapi gitpython sumatra nixio; \
    $VENV/bin/pip install --upgrade nose ipython

# Install from sources libneurosim (NEST boost dependency)
WORKDIR $HOME/packages
RUN git clone --depth 1 https://github.com/INCF/libneurosim.git
RUN cd libneurosim; ./autogen.sh
RUN mkdir $VENV/build
WORKDIR $VENV/build
RUN mkdir libneurosim; \
    cd libneurosim; \
    PYTHON=$VENV/bin/python $HOME/packages/libneurosim/configure --prefix=$VENV; \
    make; make install; ls $VENV/lib $VENV/include
RUN ln -s /usr/lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so $VENV/lib/

# Install NEST, latest stable version (July 2019)
ENV NEST_VER=2.18.0
ENV NEST=nest-simulator-$NEST_VER

WORKDIR $HOME/packages
RUN wget  https://github.com/nest/nest-simulator/archive/v$NEST_VER.tar.gz -O $HOME/packages/$NEST.tar.gz
RUN tar xzf $NEST.tar.gz; rm $NEST.tar.gz
RUN cd $NEST; \
    cmake -DCMAKE_INSTALL_PREFIX=$VENV \
          -Dwith-mpi=ON  \
          ###-Dwith-music=ON \
          -Dwith-libneurosim=$VENV \
          -DPYTHON_LIBRARY=$VENV/lib/libpython2.7.so \
          -DPYTHON_INCLUDE_DIR=/usr/include/python2.7 \
          $HOME/packages/$NEST; \
    make; make install
# RUN cd $NEST; make installcheck

# Install TVB specific packages
WORKDIR $HOME/packages

RUN $VENV/bin/pip install --upgrade pip; \
    $VENV/bin/pip install tvb-library

RUN git clone --depth 1 https://github.com/the-virtual-brain/tvb-data.git; \
    $VENV/bin/pip uninstall -y tvb-data; \
    cd tvb-data; \
    $VENV/bin/python setup.py develop

RUN git clone --depth 1 https://github.com/dionperd/tvb-scripts.git; \
    cd tvb-scripts; \
    git pull;\
    $VENV/bin/python setup.py develop

RUN git clone --depth 1 https://github.com/dionperd/tvb-multiscale.git; \
    cd tvb-multiscale; \
    git pull;\
    $VENV/bin/python setup.py develop
# Test tvb-multiscale by running this:
#RUN $VENV/bin/python -m tvb_nest.examples.red_ww_exc_io_inh_i.py

# tvb42
RUN $VENV/bin/pip install jupyterlab
RUN $VENV/bin/jupyter notebook --generate-config
RUN echo "c.NotebookApp.password='sha1:12bff019c253:9daecd92c2e9bdb10b3b8a06767a74a0fe078d7c'">>/home/docker/.jupyter/jupyter_notebook_config.py

# Activate the rights python env
WORKDIR /home/docker/
RUN echo "source $VENV/bin/activate" >> .bashrc
RUN mkdir /home/docker/.tvb-temp; chmod 777 /home/docker/; chmod 777 /home/docker/.tvb-temp
CMD $VENV/bin/jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

#srun -C mc --time 23:59:00 --partition=normal
#       shifter run --mount=type=bind,source=$HOME,destination=$HOME
#       thevirtualbrain/tvb-nest /home/docker/env/neurosci/bin/python -m console_run.py

#docker run -d -p 8888:8888 thevirtualbrain/tvb-nest