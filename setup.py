#!/usr/bin/env python

#===============================================================================
# Copyright (c) 2020, Anand K Subramanian.
# All rights reserved.

import setuptools
import misso

""" Setup for CPU Version """
with open("README.md", 'r') as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as fh:
    requirements = fh.readlines()
    requirements = [r.split('\n')[0] for r in requirements]

setuptools.setup(
    name="misso",
    version=misso.__version__,
    author="Anand K Subramanian",
    author_email="anandkrish894@gmail.com",
    description="Python package to compute mutual information matrix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AntixK/MISSO",
    license="MIT",
    packages=["misso"],
    install_requires= requirements,
    classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Intended Audience :: Science/Research",
            "Natural Language :: English"
    ],
    python_requires=">=3.6",
)
#===========================================================================#

# """ Setup for GPU Version """
