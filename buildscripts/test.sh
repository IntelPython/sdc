#!/bin/bash

buildscripts="$(dirname $(realpath $0))"
output_folder="$(dirname $buildscripts)/sdc-build"
recipe="$buildscripts/sdc-conda-recipe"

set -ex

conda env update -f "$buildscripts/environment-build.yml"

output=$(conda build --output-folder "$output_folder" --output "$recipe" | head -n 1)

# conda build --test -c file:///workspaces/sdc/sdc-build -c intel/label/beta -c intel -c defaults /workspaces/sdc/sdc-build/linux-64/sdc-0.32.0-py36h0922cd1_28.tar.bz2
conda build \
    --test \
    -c "file://$output_folder" -c intel/label/beta -c intel -c defaults --override-channels \
    "$output"
