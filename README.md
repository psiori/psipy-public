# PSIORI Machine Learning Toolbox

[![Build Status](https://dev.azure.com/psiori/psipy/_apis/build/status/psipy?branchName=develop)](https://dev.azure.com/psiori/psipy/_build/latest?definitionId=6&branchName=develop)

## Usage

A good starting point is the documentation hosted on [psipy.azurewebsites.net](https://psipy.azurewebsites.net). It can also be built locally using `make doc`.

Generally it holds, that psipy is just a normal python package, but not distributed through pypi. To get started on your local machine, run the following lines, maybe inside a virtual environment (`conda` or `python -m venv`):

```Shell
[python3|python3.8] --version  # requires python 3.8, use e.g. brew install pythons@3.8 if missing on your development machine!
git clone git@github.com:psiori/psipy.git
pip install -e "./psipy[dev,automl,gym]"
```

## Contributing

With psipy being the core library for most internal python projects, it can sometimes be hard to keep track of the latest changes. To avoid that, we aim for short lived branches, small pull requests and frequent merges to develop.

- No project specific release branches.
- Feature branches are kept small and are frequently merged to `develop`.
- Releases (versions published to customer) always are a tagged version off of `master`.

Sticking to those principles will make changes to the shared codebase frequent, but small. Projects under active development should frequently and therefore easily update to the latest psipy `develop` state without the need for major refactors. Projects need to plan for the time needed for such merges. The idea of those principles is to avoid prolonged feature branches which get hard to merge at some point or actually never get merged at all.

If a project requires new psipy features after some time of inactivity, it needs to update to the latest head. While this might produce some overhead in that project, it will also ensure an easier to maintain and faster to advance shared codebase.

### Documentation

Code documentation should live close to code to keep it maintained. Usage examples are in the best case doctests and therefore both runnable and executed by pytest. Documentation is automatically published from `develop` ([psipy.azurewebsites.net](https://psipy.azurewebsites.net)) and can be published manually from PRs to a staging environment ([psipy-staging.azurewebsites.net](https://psipy-staging.azurewebsites.net), see [#203](https://github.com/psiori/psipy/pull/203) for more details).

### Pre-commit hooks

psipy provides [pre-commit](https://pre-commit.com/) hooks for developers. After having installed psipy, run `pre-commit install` to setup git hook scripts. From now on, `flake8`, `mypy` and `black` (only checks) will be run on all staged files before every commit as well as checks for trailing whitespaces, newlines at end of files and large filesizes. Hooks are configured in `.pre-commit-config.yaml`.

In case you actively want to ignore all hooks, use `git commit --no-verify`. For ignoring only specific hooks, you can use the `SKIP` environment variable, e.g. `SKIP=flake8 git commit -m "foo"`.

## License

**The following is currently not working / only partially implemented.**

A decorator is attached to private methods, which validates a license key file against a public key. The license key file is time dependent.
