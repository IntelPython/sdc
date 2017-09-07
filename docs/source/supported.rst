.. _supported:

Supported Python Programs
=========================

HPAT compiles and parallelizes the functions annotated with the `@hpat.jit`
decorator. Therefore, file I/O and computations on large datasets should be
inside the jitted functions. The supported data structures for large datasets
are `Numpy <http://www.numpy.org/>`_ arrays and
`Pandas <http://pandas.pydata.org/>`_ dataframes.

Enabling Parallelization
------------------------

To enable parallelization, HPAT needs to recognize the large datasets and their
computation in the program. Hence, only the supported data-parallel operators of
Numpy and Pandas can be used for large datasets and computations. The sequential
computation on small data can be any code that
`Numba supports <http://numba.pydata.org/numba-doc/latest/index.html>`_.

Supported Numpy Functions
-------------------------

Below is the list of the data-parallel Numpy operators that HPAT can optimize
and parallelize.

1. Numpy `element-wise` or `point-wise` array operations:

    * Unary operators: ``+`` ``-`` ``~``
    * Binary operators: ``+`` ``-`` ``*`` ``/`` ``/?`` ``%`` ``|`` ``>>`` ``^``
        ``<<`` ``&`` ``**`` ``//``
    * Comparison operators: ``==`` ``!=`` ``<`` ``<=`` ``>`` ``>=``
    * data-parallel math operations: ``add``, ``subtract``, ``multiply``,
        ``divide``, ``logaddexp``, ``logaddexp2``, ``true_divide``,
        ``floor_divide``, ``negative``, ``power``, ``remainder``,
        ``mod``, ``fmod``, ``abs``, ``absolute``, ``fabs``, ``rint``, ``sign``,
        ``conj``, ``exp``, ``exp2``, ``log``, ``log2``, ``log10``, ``expm1``,
        ``log1p``, ``sqrt``, ``square``, ``reciprocal``, ``conjugate``
    * Trigonometric functions: ``sin``, ``cos``, ``tan``, ``arcsin``,
        ``arccos``, ``arctan``, ``arctan2``, ``hypot``, ``sinh``, ``cosh``,
        ``tanh``, ``arcsinh``, ``arccosh``, ``arctanh``, ``deg2rad``,
        ``rad2deg``, ``degrees``, ``radians``
    * Bit manipulation functions: ``bitwise_and``, ``bitwise_or``,
        ``bitwise_xor``, ``bitwise_not``, ``invert``, ``left_shift``,
        ``right_shift``

2. Numpy reduction functions ``sum`` and ``prod``.

3. Numpy array creation functions ``zeros``, ``ones``

4. Random number generator functions: ``rand``, ``randn``,
    ``ranf``, ``random_sample``, ``sample``, ``random``,
    ``standard_normal``, ``chisquare``, ``weibull``, ``power``, ``geometric``,
    ``exponential``, ``poisson``, ``rayleigh``, ``normal``, ``uniform``,
    ``beta``, ``binomial``, ``f``, ``gamma``, ``lognormal``, ``laplace``,
    ``randint``, ``triangular``.

4. Numpy ``dot`` function between a matrix and a vector, or two vectors.

Optional arguments are not supported unless if explicitly mentioned here.
For operations on multi-dimensional arrays, automatic broadcast of
dimensions of size 1 is not supported.


Explicit Parallel Loops
-----------------------

Sometimes a program cannot be written in terms of data-parallel operators easy
and explicit parallel loops are required.
In this case, one can use HPAT's ``prange`` instead of ``range`` to specify that a
loop can be parallelized. The user is required to make sure that the loop does
not have cross iteration dependencies except the supported reductions.
Currently, only sum using the ``+=`` operator is supported.
The example below demonstrates a parallel loop with a
reduction::

    from HPAT import jit, prange
    @jit
    def prange_test(n):
        A = np.random.ranf(n)
        s = 0
        for i in prange(len(A)):
            s += A[i]
        return s

Supported Pandas Functions
--------------------------

Below is the list of the Pandas operators that HPAT supports. Since Numba
doesn't support Pandas, only these operations can be used for both large and
small datasets.

1. Dataframe creation with the ``DataFrame`` constructor. Only a dictionary is
    supported as input. For example::

        df = pd.DataFrame({'A': np.ones(n), 'B': np.random.ranf(n)})

2. Accessing columns using both getitem (e.g. ``df['A']``) and attribute (``df.A``) is supported. 
   
3. Using columns similar to Numpy arrays and performing data-parallel operations listed previously is supported.

4. s

File I/O
--------

Currently, HPAT only supports I/O for the `HDF5 <http://www.h5py.org/>`_ format.
