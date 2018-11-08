#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='PRF',
      version='0.1dev',
      description='Probabilistic Random Forest',
      author='Itamar Reis, Dalya Baron',
      author_email='itamarreis@mail.tau.ac.il, dalyabaron@gmail.com',
      packages=['PRF'],
      zip_safe=False,
      install_requires=['numpy', 'scipy',
                      'numba', 'joblib']

     )
