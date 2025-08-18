from setuptools import find_packages
from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='avp_human_data',
    version='1.0.0',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='egocentric human data collection with apple vision pro',
    install_requires=required,
)
