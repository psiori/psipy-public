#!/usr/bin/env bash

# cnvrg prerun.sh
pip install -e ".[automl,dev,gym]"
pre-commit install
