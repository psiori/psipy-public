# PSIORI's Machine Learning Library -- The Public Part

The psipy library is a private collection of machine learning algorithms and tools developed and used by PSIORI. It is designed to be a modular and extensible framework for building and deploying software solutions that incorporate or are based on machine learning components. This public version of the library is a subset of the full private library and currently includes only large parts of the Reinforcement Learning (RL) module. It may be expanded to include additional modules in the future.

## Installation

To install the psipy library, you can use pip. First, ensure you have Python 3.8 or later installed on your system. 

We suggest creating a virtual environment for your project using psipy or when working on psipy itself. You can use the following command to create a virtual environment and activate it:

```Shell
python3.8 -m venv .venv
source .venv/bin/activate
```

Then, you can install psipy directly from the GitHub repository:

```Shell
[python] --version  # make sure python 3.8 is used!
git clone git@github.com:psiori/psipy-public.git
pip install -e "./psipy-public[dev,gym]"
```
The option '-e' is used to install the package in editable mode, which allows you to make changes to the code and have them reflected in the installed package without having to reinstall.

The options '[dev,gym]' are used to install the dependencies for the development and gym environments. Please be aware that we switched to Farama-Foundations's fork [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) of OpenAI's gym when they took over maintenance of the original gym library.


## Usage

To get started with using psipy, you can explore the provided tutorials. One of the best ways to familiarize yourself with the library is by running the batch tutorial Jupyter notebook. Here's how you can do that:

1. Navigate to the examples directory in your terminal:

   ```
   cd psipy-public/examples/rl/tutorials
   ```

2. Launch Jupyter Notebook:

   ```
   jupyter notebook
   ```

3. In the Jupyter interface that opens in your web browser, locate and click on the "batch_learning_tutorials.ipynb" file to open it.

4. You can now run the cells in the notebook to see how psipy works with batch reinforcement learning tasks.

This tutorial will guide you through the basics of using psipy for reinforcement learning tasks, specifically focusing on batch learning scenarios.

Further examples, also including python scripts outside jupyter notebooks, can be found in the [examples](./examples) folder.

## Documentation

The documentation can be built locally using `make doc`.

## Contributing

With psipy being the core library for most internal python projects, it can sometimes be hard to keep track of the latest changes. To avoid that, we aim for short lived branches, small pull requests and frequent merges to develop.

- No project specific release branches.
- Feature branches are kept small and are frequently merged to `develop`.
- Releases (versions published to customer) always are a tagged version off of `master`.

Sticking to those principles will make changes to the shared codebase frequent, but small. Projects under active development should frequently and therefore easily update to the latest psipy `develop` state without the need for major refactors. Projects need to plan for the time needed for such merges. The idea of those principles is to avoid prolonged feature branches which get hard to merge at some point or actually never get merged at all.

If a project requires new psipy features after some time of inactivity, it needs to update to the latest head. While this might produce some overhead in that project, it will also ensure an easier to maintain and faster to advance shared codebase.

### Documentation

Code documentation should live close to code to keep it maintained. Usage examples are in the best case doctests and therefore both runnable and executed by pytest.

### Pre-commit hooks

psipy provides [pre-commit](https://pre-commit.com/) hooks for developers. After having installed psipy, run `pre-commit install` to setup git hook scripts. From now on, `flake8`, `mypy` and `black` (only checks) will be run on all staged files before every commit as well as checks for trailing whitespaces, newlines at end of files and large filesizes. Hooks are configured in `.pre-commit-config.yaml`.

In case you actively want to ignore all hooks, use `git commit --no-verify`. For ignoring only specific hooks, you can use the `SKIP` environment variable, e.g. `SKIP=flake8 git commit -m "foo"`.

## License

The code is licensed under the BSD 3-Clause License. See the [LICENSE](./LICENSE) file for more details.
