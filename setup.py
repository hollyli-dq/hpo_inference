#!/usr/bin/env python3
"""
Setup script for the Bayesian Partial Order Planning package.
"""

from setuptools import setup, find_packages

with open("bayesian_pop_README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bayesian-pop",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Bayesian Partial Order Planning using Hierarchical Partial Orders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bayesian-pop",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "networkx>=2.5",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "flake8>=3.8.0",
            "black>=20.8b1",
        ],
    },
) 