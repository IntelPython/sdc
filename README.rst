**********************************
Intel® Scalable Dataframe Compiler
**********************************

.. image:: https://travis-ci.com/IntelPython/sdc.svg?branch=master
    :target: https://travis-ci.com/IntelPython/sdc
    :alt: Travis CI

.. image:: https://dev.azure.com/IntelPython/HPAT/_apis/build/status/IntelPython.sdc?branchName=master
    :target: https://dev.azure.com/IntelPython/HPAT/_build/latest?definitionId=2&branchName=master
    :alt: Azure Pipelines

.. image:: https://coveralls.io/repos/github/IntelPython/sdc/badge.svg?branch=master
    :target: https://coveralls.io/github/IntelPython/sdc?branch=master
    :alt: Coveralls

Numba* Extension For Pandas* Operations Compilation
###################################################

Intel® Scalable Dataframe Compiler (Intel® SDC), which is an extension of `Numba* <https://numba.pydata.org/>`_ 
that enables compilation of `Pandas* <https://pandas.pydata.org/>`_ operations. It automatically vectorizes and parallelizes 
the code by leveraging modern hardware instructions and by utilizing all available cores. 

Intel SDC's documentation can be found `here <https://intelpython.github.io/sdc-doc/>`_.

Installing Binary Packages (conda)
----------------------------------
::

   conda install -c intel -c intel/label/test sdc


Example
#######

Here is a Pi calculation example in Intel SDC:

.. code:: python

    import sdc
    import numpy as np
    import time

    @sdc.jit
    def calc_pi(n):
        t1 = time.time()
        x = 2 * np.random.ranf(n) - 1
        y = 2 * np.random.ranf(n) - 1
        pi = 4 * np.sum(x**2 + y**2 < 1) / n
        print("Execution time:", time.time()-t1, "\nresult:", pi)
        return pi

    calc_pi(2 * 10**8)

Save this in a file named `pi.py` and run (on 8 cores)::

    mpiexec -n 8 python pi.py

This should demonstrate about 100x speedup compared to regular Python version
without `@sdc.jit` and `mpiexec`.


References
##########

These academic papers describe the underlying methods in Intel SDC:

- `HPAT paper at ICS'17 <http://dl.acm.org/citation.cfm?id=3079099>`_
- `HPAT at HotOS'17 <http://dl.acm.org/citation.cfm?id=3103004>`_
- `HiFrames on arxiv <https://arxiv.org/abs/1704.02341>`_


Building Intel® SDC from Source on Linux
----------------------------------------

We use `Anaconda <https://www.anaconda.com/download/>`_ distribution of
Python for setting up Intel SDC build environment.

If you do not have conda, we recommend using Miniconda3::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH

It is possible to build Intel SDC via conda-build or setuptools. Follow one of the
cases below to install Intel SDC and its dependencies on Linux.

Building on Linux with conda-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    PYVER=<3.6 or 3.7>
    conda create -n CBLD python=$PYVER conda-build
    source activate CBLD
    git clone https://github.com/IntelPython/sdc
    cd sdc
    # build Intel SDC
    conda build --python $PYVER --override-channels -c numba -c conda-forge -c defaults buildscripts/sdc-conda-recipe

Building on Linux with setuptools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    PYVER=<3.6 or 3.7>
    conda create -n SDC -q -y -c numba -c conda-forge -c defaults numba mpich pyarrow=0.15.0 arrow-cpp=0.15.0 gcc_linux-64 gxx_linux-64 gfortran_linux-64 scipy pandas boost python=$PYVER
    source activate SDC
    git clone https://github.com/IntelPython/sdc
    cd sdc
    # build SDC
    python setup.py install

In case of issues, reinstalling in a new conda environment is recommended.

Building Intel® SDC from Source on Windows
------------------------------------------

Building Intel® SDC on Windows requires Build Tools for Visual Studio 2019 (with component MSVC v140 - VS 2015 C++ build tools (v14.00)):

* Install `Build Tools for Visual Studio 2019 (with component MSVC v140 - VS 2015 C++ build tools (v14.00)) <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019>`_.
* Install `Miniconda for Windows <https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_.
* Start 'Anaconda prompt'

It is possible to build Intel SDC via conda-build or setuptools. Follow one of the
cases below to install Intel SDC and its dependencies on Windows.

Building on Windows with conda-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    set PYVER=<3.6 or 3.7>
    conda create -n CBLD -q -y python=%PYVER% conda-build conda-verify vc vs2015_runtime vs2015_win-64
    conda activate CBLD
    git clone https://github.com/IntelPython/sdc.git
    cd sdc
    conda build --python %PYVER% --override-channels -c numba -c defaults -c intel buildscripts\sdc-conda-recipe

Building on Windows with setuptools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    conda create -n SDC -c numba -c defaults -c intel -c conda-forge python=<3.6 or 3.7> numba impi-devel pyarrow=0.15.0 arrow-cpp=0.15.0 scipy pandas boost
    conda activate SDC
    git clone https://github.com/IntelPython/sdc.git
    cd sdc
    set INCLUDE=%INCLUDE%;%CONDA_PREFIX%\Library\include
    set LIB=%LIB%;%CONDA_PREFIX%\Library\lib
    %CONDA_PREFIX%\Library\bin\mpivars.bat quiet
    python setup.py install

.. "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

Troubleshooting Windows Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If the ``cl`` compiler throws the error fatal ``error LNK1158: cannot run 'rc.exe'``,
  add Windows Kits to your PATH (e.g. ``C:\Program Files (x86)\Windows Kits\8.0\bin\x86``).
* Some errors can be mitigated by ``set DISTUTILS_USE_SDK=1``.
* For setting up Visual Studio, one might need go to registry at
  ``HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7``,
  and add a string value named ``14.0`` whose data is ``C:\Program Files (x86)\Microsoft Visual Studio 14.0\``.
* Sometimes if the conda version or visual studio version being used are not latest then building Intel SDC can throw some vague error about a keyword used in a file. So make sure you are using the latest versions.


Building documentation
----------------------
Building Intel SDC User's Guide documentation requires pre-installed Intel SDC package along with compatible Pandas* version as well as Sphinx* 2.2.1 or later.

You can install Sphinx* using either ``conda`` or ``pip``:
::

    conda install sphinx
    pip install sphinx

Currently the build precedure is based on ``make`` located at ``./sdc/docs/`` folder. While it is not generally required we recommended that you clean up the system from previous documentaiton build by running
::

    make clean
    
To build HTML documentation you will need to run
::

    make html

The built documentation will be located in the ``.sdc/docs/build/html`` directory. To preview the documentation open ``index.html``
file. 

Sphinx* Generation Internals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The documentation generation is controlled by ``conf.py`` script automatically invoked by Sphinx. 
See `Sphinx documentation <http://www.sphinx-doc.org/en/master/usage/configuration.html>`_ for details.

The API Reference for Intel SDC User's Guide is auto-generated by inspecting ``pandas`` and ``sdc`` modules. That's why these modules must be pre-installed for documentation generation using Sphinx*. However, there is a possibility to skip API Reference auto-generation by setting environment variable ``SDC_DOC_NO_API_REF_STR=1``.

If the environment variable ``SDC_DOC_NO_API_REF_STR`` is unset then Sphinx's ``conf.py`` invokes ``generate_api_reference()`` function located in ``./sdc/docs/source/buildscripts/apiref_generator`` module. This function parses ``pandas`` and ``sdc`` docstrings for each API, combines those into single docstring and writes it into RST file with respective Pandas* API name. The auto-generated RST files are
located at ``./sdc/docs/source/_api_ref`` directory.

.. note:
    Sphinx will automatically clean the ``_api_ref`` directory on the next invocation of the documenation build.
      
Intel SDC docstring decoration rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Since SDC API Reference is auto-generated from respective Pandas* and Intel SDC docstrings there are certain rules that must be
followed to accurately generate the API description.

1. Every SDC API must have the docstring.
    If developer does not provide the docstring then Sphinx will not be able to match Pandas docstring with respective SDC one. In this     situation Sphinx assumes that SDC does not support such API and will include respective note in the API Reference that 
    **This API is currently unsupported**.
    
2. Follow 'one function - one docstring' rule.
    You cannot have one docstring for multiple APIs, even if those are very similar. Auto-generator assumes every SDC API is covered by
    respective docstring. If Sphinx does not find the docstring for particular API then it assumes that SDC does not support such API 
    and will include respective note in the API Reference that **This API is currently unsupported**.
    
3. Description (introductory section, the very first few paragraphs without a title) is taken from Pandas*.
Intel SDC developers should not include API description in SDC docstring. 
    But developers are encouraged to follow Pandas API description naming conventions 
    so that the combined docstring appears consistent.
    
4. Parameters, Returns, and Raises sections' description is taken from Pandas* docstring. 
SDC developers should not include such descriptions in their SDC docstrings.
    Rather developers are encouraged to follow Pandas naming conventions 
    so that the combined docstring appears consistent.
    
5. Every SDC docstring must be of the follwing structure:
    ::

        """
        Intel Scalable Dataframe Compiler User Guide
        ********************************************
        Pandas API: <full pandas name, e.g. pandas.Series.nlargest>

        <Intel SDC specific sections>

        Intel Scalable Dataframe Compiler Developer Guide
        *************************************************
        <Developer's Guide specific sections> 
        """

The first two lines must be the User Guide header. This is an indication to Sphinx that this section is intended for public API
and it will be combined with repsective Pandas API docstring.
    
Line 3 must specify what Pandas API this Intel SDC docstring does correspond to. It must start with ``Pandas API:`` followed by
full Pandas API name that corresponds to this SDC docstring. Remember to include full name, for example, ``nlargest`` is not
sufficient for auto-generator to perform the match. The full name must be ``pandas.Series.nlargest``.
    
After User Guide sections in the docstring there can be another header indicating that the remaining part of the docstring belongs to
Developer's Guide and must not be included into User's Guide.

6. Examples, See Also, References sections are **NOT** taken from Pandas docstring. SDC developers are expected to complete these sections in SDC doctrings.
    This is so because respective Pandas sections are sometimes too Pandas specific and are not relevant to SDC. SDC developers have to
    rewrite those sections in Intel SDC style. Do not forget about User Guide header and Pandas API name prior to adding SDC specific
    sections.
    
7. Examples section is mandatory for every SDC API. 'One API - at least one example' rule is applied.
    Examples are essential part of user experience and must accompany every API docstring. 

8. Embed examples into Examples section from ``./sdc/examples``.
    Rather than writing example in the docstring (which is error-prone) embed relevant example scripts into the docstring. For example,
    here is an example how to embed example for ``pandas.Series.get()`` function into respective Intel SDC docstring:
    
    ::
    
        """
        ...
        Examples
        --------
        .. literalinclude:: ../../../examples/series_getitem.py
           :language: python
           :lines: 27-
           :caption: Getting Pandas Series elements
           :name: ex_series_getitem

        .. code-block:: console

            > python ./series_getitem.py
            55        
    
    In the above snapshot the script ``series_getitem.py`` is embedded into the docstring. ``:lines: 27-`` allows to skip lengthy
    copyright header of the file. ``:caption:`` provides meaningful description of the example. It is a good tone to have the caption
    for every example. ``:name:`` is the Sphinx name that allows referencing example from other parts of the documentation. It is a good 
    tone to include this field. Please follow the naming convention ``ex_<example file name>`` for consistency.
    
    Accompany every example with the expected output using ``.. code-block:: console`` decorator.
    
     
        **Every Examples section must come with one or more examples illustrating all major variations of supported API parameter  combinations. It is highly recommended to illustrate SDC API limitations (e.g. unsupported parameters) in example script comments.**

9. See Also sections are highly encouraged. 
    This is a good practice to include relevant references into the See Also section. Embedding references which are not directly 
    related to the topic may be distructing if those appear across API description. A good style is to have a dedicated section for 
    relevant topics. 
    
    See Also section may include references to relevant SDC and Pandas as well as to external topics.
    
    A special form of See Also section is References to publications. Pandas documentation sometimes uses References section to refer to
    external projects. While it is not prohibited to use References section in SDC docstrings, it is better to combine all references
    under See Also umbrella.
    
10. Notes and Warnings must be decorated with ``.. note::`` and ``.. warning::`` respectively.
    Do not use
    ::
        Notes
        -----
        
        Warning
        -------
    
    Pay attention to indentation and required blank lines. Sphinx is very sensitive to that.
    
11. If SDC API does not support all variations of respective Pandas API then Limitations section is mandatory.
    While there is not specific guideline how Limitations section must be written, a good style is to follow Pandas Parameters section
    description style and naming conventions.

12. Before committing your code for public SDC API you are expected to: 

    - have SDC docstring implemented;
    - have respective SDC examples implemented and tested
    - API Reference documentation generated and visually inspected. New warnings in the documentation build are not allowed.

Running unit tests
------------------
::

    python sdc/tests/gen_test_data.py
    python -m unittest

