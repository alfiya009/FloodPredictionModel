#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/vercel/path0"
python -m pip install --upgrade pip
python -m pip install wheel setuptools
python -m pip install -r requirements.txt