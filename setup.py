#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="xgtool3",
    version="2.1.2",
    author="Keiichi Hashimoto",
    author_email="k1bridgebook@g.ecc.u-tokyo.ac.jp",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "xarray",
        "pandas",
        "dask",
        "cftime",
    ],
)
