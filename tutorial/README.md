# HPAT_tutorial

To get the full experience you need to setup your python envorinment as follows:
* conda create -n hpatidp -c intel python=3.6 tbb-devel numpy pandas scipy cython jinja2 daal daal-include impi-devel
* conda activate hpatidp # or source activate idp
* conda install jupyter notebook matplotlib boost cmake pyarrow gcc_linux-64 gxx_linux-64 gfortran_linux-64
* conda install -c numba/label/dev -c intel llvmlite
* git clone https://github.com/numba/numba
* pushd numba && python setup.py install && popd
* git clone https://github.com/IntelLabs/hpat
* pushd hpat && python setup.py install && popd
* git clone https://github.com/IntelPython/daal4py hpat
* pushd daal4py && python setup.py install && popd

Now  you can use the notebook!
* jupyter notbook
