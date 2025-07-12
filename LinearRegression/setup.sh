#!/bin/bash

ENV_NAME="LinearRegression"

conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

pip install -r requirements.txt