#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import find_packages, setup


def readfile(filename):
    with open(filename, encoding="utf-8") as fp:
        filecontents = fp.read()
    return filecontents

setup(
    name="astrovision",
    version="0.0.1",
    description="Astronomy images datasets for use in PyTorch",
    author="Adam Moss",
    author_email="adam.moss@nottingham.ac.uk",
    maintainer="Adam Moss",
    maintainer_email="adam.moss@nottingham.ac.uk",
    url="https://github.com/adammoss/astro-vision-datasets/",
    license="MIT",
    packages=find_packages(),
    provides=["astrovision"],
    install_requires=readfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt")
    )
)
