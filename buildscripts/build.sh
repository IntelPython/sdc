#!/bin/bash

buildscripts="$(dirname $(realpath $0))"
output_folder="$(dirname $buildscripts)/sdc-build"
recipe="$buildscripts/sdc-conda-recipe"

# cd buildscripts

# echo $buildscripts $output_folder

set -ex

# conda create -y -n sdc_env python=3.7 conda-build -c anaconda -c intel/label/beta -c intel -c defaults --override-channels
conda env update -f "$buildscripts/environment-build.yml"

# conda build --no-test --python 3.7 --numpy 1.18 --output-folder /workspaces/sdc/sdc-build -c intel/label/beta -c intel -c defaults --override-channels /workspaces/sdc/buildscripts/sdc-conda-recipe
conda build \
    --no-test \
    $@ \
    --output-folder "$output_folder" \
    -c intel/label/beta -c intel -c defaults --override-channels \
    "$recipe"
