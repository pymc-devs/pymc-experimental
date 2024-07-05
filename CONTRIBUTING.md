# Contributing guide

Thank you for your interest in contributing to PyMC and PyMC-experimental!

This page outlines the steps to follow if you wish to contribute to the pymc-experimental repo and clone the repo locally.

## Install locally
**1**: Create a folder `pymc-devs` in your local machine and follow the steps on [cloning PyMC locally](https://www.pymc.io/projects/docs/en/latest/contributing/pr_tutorial.html).

Since PyMC-experimental should integrate with the latest version of PyMC, it is recommended that any development work on PyMC-experimental must also work with the latest version of PyMC.

You should now have a local copy of PyMC under `pymc-devs/pymc`.

**2**: Fork the PyMC-experimental repo and clone it locally:

```
git clone git@github.com:<your GitHub handle>/pymc-experimental.git
cd pymc-experimental
git remote add upstream git@github.com:pymc-devs/pymc-experimental.git
```

You should now have a local copy of PyMC-experimental under `pymc-devs/pymc-experimental`.

Create a new conda environment by first installing the dependencies in the main PyMC repo:
```
conda env create -n pymc-experimental -f /path/to/pymc-devs/pymc/conda-envs/environment-dev.yml
conda activate pymc-experimental
pip install -e /path/to/pymc-devs/pymc

# ignores the specific pymc version to install pymc-experimental
pip install -e /path/to/pymc-devs/pymc-experimental --ignore-installed pymc
```

**3** Check origin and upstream is correct.

**PyMC**
```
cd /path/to/pymc-devs/pymc
git remote -v
```
Output:
```
origin  git@github.com:<your GitHub handle>/pymc.git (fetch)
origin  git@github.com:<your GitHub handle>/pymc.git (push)
upstream        git@github.com:pymc-devs/pymc.git (fetch)
upstream        git@github.com:pymc-devs/pymc.git (push)
```

**PyMC-experimental**
```
cd /path/to/pymc-devs/pymc-experimental
git remote -v
```
Output:
```
origin  git@github.com:<your GitHub handle>/pymc-experimental.git (fetch)
origin  git@github.com:<your GitHub handle>/pymc-experimental.git (push)
upstream        git@github.com:pymc-devs/pymc-experimental.git (fetch)
upstream        git@github.com:pymc-devs/pymc-experimental.git (push)
```



## Git integration [(from PyMC's main page)](https://www.pymc.io/projects/docs/en/latest/contributing/pr_tutorial.html)

**1** Develop the feature on your feature branch:
```
git checkout -b my-exp-feature
```

**2** Before committing, run pre-commit checks:
```
pip install pre-commit
pre-commit run --all      # 👈 to run it manually
pre-commit install        # 👈 to run it automatically before each commit
```

**3** Add changed files using git add and then git commit files:
```
git add modified_files
git commit
```
to record your changes locally.

**4** After committing, it is a good idea to sync with the base repository in case there have been any changes:
```
# pymc
cd /path/to/pymc-devs/pymc
git fetch upstream
git rebase upstream/main

# (pymc-dev team) Please double check this
pip install -e /path/to/pymc-devs/pymc

# pymc-exp
cd /path/to/pymc-devs/pymc-experimental
git fetch upstream
git rebase upstream/main
```
Then push the changes to the fork in your GitHub account with:
```
git push -u origin my-exp-feature
```

**5** Go to the GitHub web page of your fork of the PyMC repo. Click the ‘Pull request’ button to send your changes to the project’s maintainers for review. This will send a notification to the committers.

## Final steps

Review contributing guide in [PyMC's main page](https://www.pymc.io/projects/docs/en/latest/contributing/index.html).

FAQ [page](https://github.com/pymc-devs/pymc-experimental#questions).

Discussions [page](https://github.com/pymc-devs/pymc-experimental/discussions/5).
