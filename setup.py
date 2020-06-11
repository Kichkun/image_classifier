#!/usr/bin/env python

import os
from distutils.core import setup

folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(name='image_classifier',
      version='0.1',
      description='Realization of image classifier',
      author='Kishkun Anastasia',
      author_email='',
      package_dir={},
      packages=["vgg_classifier"],
      install_requires=install_requires
      )