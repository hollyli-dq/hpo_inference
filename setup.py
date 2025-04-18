from setuptools import setup, find_packages

setup(
    name="hpo_inference",
    version="0.1.0",
    packages=find_packages(include=['hpo_inference', 'hpo_inference.*']),
    package_data={
        'hpo_inference': ['config/*.yaml'],
    },
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "networkx>=2.8.0",
        "requests>=2.28.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    author="Dongqing Li",
    author_email="dongqing.li @kell.ox.ac.uk",
    description="A package for HPO inference using MCMC and bayesian approaches",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hpo_inference",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
) 