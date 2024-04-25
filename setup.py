#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fgspectra: GitHub <https://github.com/simonsobs/fgspectra>."""

from setuptools import setup

setup(
    name="fgspectra",
    version="1.1.0",
    description="Foreground SED and power spectrum library",
    author="Simons Observatory fgspectra crew",
    author_email="",
    packages=["fgspectra"],
    python_requires=">3.9",
    install_requires=[
        "scipy",
        "pyyaml",
    ],
    include_package_data=True,
)
