from setuptools import setup, find_packages

MAJOR = 0
MINOR = 0
PATCH = 1
VERSION = '%d.%d.%d' % (MAJOR, MINOR, PATCH)

setup(name='pyspa',
      version=VERSION,
      description='Functions for system discretization',
      packages=find_packages(exclude=['tests']),
      install_requires=[
          'joblib>=0.13.2',
          'numpy>=1.16.2',
          'scipy>=1.2.1',
          'scikit-learn>=0.20.3'],
      test_suite='tests'
      )
