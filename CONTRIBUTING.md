# Contributing guide

Thank you for your interest in contributing to PyMC and PyMC-experimental!

This page outlines the steps to follow if you wish to contribute to the pymc-experimental repo and clone the repo locally.

## Install locally
**1**: Create a folder `pymc-devs` in your local machine and follow the [cloning PyMC locally](https://www.pymc.io/projects/docs/en/latest/contributing/pr_tutorial.html).

Since PyMC-experimental should integrate with the latest version of PyMC, it is recommended that any development work on PyMC-experimental must also work with the latest version of PyMC.

You should now have a local copy of PyMC under `pymc-devs/pymc`.

**2**: Fork the PyMC-experimental repo and clone it locally:

```
git clone git@github.com:<your GitHub handle>/pymc-experimental.git
cd pymc-experimental
git remote add upstream git@github.com:pymc-devs/pymc-experimental.git
```

Create a new conda environment by first installing the dependencies in the main PyMC repo:
```
conda env create -n pymc-experimental -f /path/to/pymc-devs/pymc/conda-envs/environment-dev.yml
conda activate pymc-experimental
pip install -e /path/to/pymc-devs/pymc

# ignores the specific pymc version to install pymc-experimental
pip install -e /path/to/pymc-devs/pymc-experimental --no-deps --ignore-installed pymc
```

## Develop the feature

**1** Develop the feature on your feature branch:
```
git checkout -b my-exp-feature
```



You should now have a local copy of PyMC-experimental under `pymc-devs/pymc-experimental`.

**Final steps**: Review contributing guide in [PyMC's main page](https://www.pymc.io/projects/docs/en/latest/contributing/index.html).

Page in construction, for now go to https://github.com/pymc-devs/pymc-experimental#questions.
