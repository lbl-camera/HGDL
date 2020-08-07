#!/bin/bash
rm -rf hgdl.egg-info build dist
find . -type d -name '__pycache__' -exec rm -r {} +
find . -type d -name '.ipynb_checkpoints' -exec rm -r {} +
find . -type d -name 'dask-worker-space' -exec rm -r {} +
