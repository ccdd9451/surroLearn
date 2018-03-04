#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup, find_packages

setup(
    name="surroLearn",
    version="1.0",
    packages=find_packages(),
    scripts=['bin/learn'],
    install_requires=[
        "fire",
        "mock",
    ],
)
