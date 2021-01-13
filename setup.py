# -*- coding: utf-8 -*-
#
#
#

"""
Install tvb-nest and tvb-scripts packages.

Execute:
    python setup.py install/develop

"""

import shutil
import setuptools


VERSION = "1.0.2"

INSTALL_REQUIREMENTS = ["pandas", "xarray", "elephant", "dill"]

setuptools.setup(name='tvb-nest',
                 version=VERSION,
                 packages=setuptools.find_packages(),
                 include_package_data=True,
                 install_requires=INSTALL_REQUIREMENTS,
                 description='A package for multiscale simulations with TVB and NEST.',
                 license="GPL v3",
                 author="Dionysios Perdikis, Lia Domide, TVB Team",
                 author_email='tvb.admin@thevirtualbrain.org',
                 url='http://www.thevirtualbrain.org',
                 download_url='https://github.com/the-virtual-brain/tvb-multiscale',
                 keywords='tvb brain simulator nest neuroscience human animal neuronal dynamics builders delay')

shutil.rmtree('tvb_nest.egg-info', True)
