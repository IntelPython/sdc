.. _DevelopersGuide:

Developer's Guide
=================

HPAT implements Pandas and Numpy API as a DSL.
Data structures are implemented as Numba extensions, and
compiler stages are responsible for different levels of abstraction.
For example, `Series data type support <https://github.com/IntelLabs/hpat/blob/master/hpat/hiframes/pd_series_ext.py>`_
and `Series transformations <https://github.com/IntelLabs/hpat/blob/master/hpat/hiframes/hiframes_typed.py>`_
implement the `Pandas Series API <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`_.
Follow the pipeline for a simple function like `Series.sum()`
for initial understanding of the transformations.

HPAT Technology Overview
------------------------

This `slide deck <https://drive.google.com/open?id=1jLikSEAqOFf8kKO8vgT7ru6dKU1LGiDR>`_
provides an overview of HPAT technology and software architecture.

These papers provide deeper dive in technical ideas (might not be necessary for many developers):

- `HPAT paper on automatic parallelization for distributed memory <http://dl.acm.org/citation.cfm?id=3079099>`_
- `HPAT paper on system architecture versus Spark <http://dl.acm.org/citation.cfm?id=3103004>`_
- `HPAT Dataframe DSL approach <https://arxiv.org/abs/1704.02341>`_
- `ParallelAccelerator DSL approach <https://users.soe.ucsc.edu/~lkuper/papers/parallelaccelerator-ecoop17.pdf>`_


Numba
-----

HPAT sits on top of Numba and is heavily tied to many of its features.
Therefore, understanding Numba's internal details and being able to develop Numba extensions
is necessary.


- Start with `basic overview of Numba use <http://numba.pydata.org/numba-doc/latest/user/5minguide.html>`_ and try the examples.
- `User documentation <http://numba.pydata.org/numba-doc/latest/user/index.html>`_ is generally helpful for overview of features.
- | `ParallelAccelerator documentation <http://numba.pydata.org/numba-doc/latest/user/parallel.html>`_
    provides overview of parallel analysis and transformations in Numba (also used in HPAT).
- `Setting up Numba for development <http://numba.pydata.org/numba-doc/latest/developer/contributing.html>`_
- | `Numba architecture page <http://numba.pydata.org/numba-doc/latest/developer/architecture.html>`_
    is a good starting point for understanding the internals.
- | Learning Numba IR is crucial for understanding transformations.
    See the `IR classes <https://github.com/numba/numba/blob/master/numba/ir.py>`_.
    Setting `NUMBA_DEBUG_ARRAY_OPT=1` shows the IR at different stages
    of ParallelAccelerator and HPAT transformations. Run `a simple parallel
    example <http://numba.pydata.org/numba-doc/latest/user/parallel.html#explicit-parallel-loops>`_
    and make sure you understad the IR at different stages.
- | `Exending Numba page <http://numba.pydata.org/numba-doc/latest/extending/index.html>`_
    provides details on how to provide native implementations for data types and functions.
    The low-level API should be avoided as much as possible for ease of development and
    code readability. The `unicode support <https://github.com/numba/numba/blob/master/numba/unicode.py>`_
    in Numba is an example of a modern extension for Numba (documentation planned).
- | A more complex extension is `the new dictionary implementation in
    Numba <https://github.com/numba/numba/blob/master/numba/dictobject.py>`_ (documentation planned).
    It has examples of calling into C code which is implemented as
    `a C extension library <https://github.com/numba/numba/blob/master/numba/_dictobject.c>`_.
    For a simpler example of calling into C library, see HPAT's I/O features like
    `get_file_size <https://github.com/IntelLabs/hpat/blob/master/hpat/io.py#L12>`_.
- | `Developer reference manual <http://numba.pydata.org/numba-doc/latest/developer/index.html>`_
    provides more details if necessary.

How to update the documentation
--------------------------------

The HPAT documentation is generated from RST files using Sphinx. If you want to add to the documentation follow these steps:

1. If you don't have sphinx installed then install sphinx in your conda environment::
   
    conda install sphinx
    
2. You should also install sphinx_bootstrap_theme because that is the theme used for hPAT documentation::

    pip install sphinx_bootstrap_theme
	
2. To add new documentation you either can modify an existing RST file or create a new RST file. The location of that file will be:
   docs/source/

   
3. If you create a new RST file you should also modify the index.rst to add your filename in the table of contents. index.rst is in the same directory as other RST files. Adding the filename to this list will introduce a new section in online documentation.


4. To build the developer'sdocumentation::

    cd docs
    make developerhtml
    
    If you want to build user documentation::
    
    cd docs
    make html
    
   
This will generate all html files in _build/html directory. 

   
5. If you want to generate pdf you should have latexmk installed. After making sure you have latexmk installed follow these steps::

    cd docs
    make latex
    cd _build/latex
    make all-pdf 

The pdf will be generated in _build/latex directory as HPAT.pdf.
