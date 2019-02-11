.. _supported:

User Guide
==========

HPAT automatically parallelizes a subset of Python that is commonly used for
data analytics and machine learning. This section describes this subset
and how parallelization is performed.

HPAT compiles and parallelizes the functions annotated with the `@hpat.jit`
decorator. The decorated functions are replaced with generated parallel
binaries that run on bare metal.
The supported data structures for large datasets
are `Numpy <http://www.numpy.org/>`_ arrays and
`Pandas <http://pandas.pydata.org/>`_ dataframes.

Automatic Parallelization
-------------------------

HPAT parallelizes programs automatically based on the `map-reduce` parallel
pattern. Put simply, this means the compiler analyzes the program to
determine whether each array should be distributed or not. This analysis uses
the semantics of array operations as the program below demonstrates::

    @hpat.jit
    def example_1D(n):
        f = h5py.File("data.h5", "r")
        A = f['A'][:]
        return np.sum(A)

This program reads a one-dimensional array called `A` from file and sums its
values. Array `A` is the output of an I/O operation and is input to `np.sum`.
Based on semantics of I/O and `np.sum`, HPAT determines that `A` can be
distributed since I/O can output a distributed array and `np.sum` can
take a distributed array as input.
In `map-reduce` terminology, `A` is output of a `map` operator and is input
to a `reduce` operator. Hence,
HPAT distributes `A` and all operations associated with `A`
(i.e. I/O and `np.sum`) and generates a parallel binary.
This binary replaces the `example_1D` function in the Python program.

HPAT can only analyze and parallelize the supported data-parallel operations of
Numpy and Pandas (listed below). Hence, only the supported operations can be
used for distributed datasets and computations.
The sequential computation on small data can be any code that
`Numba supports <http://numba.pydata.org/numba-doc/latest/index.html>`_.

Array Distribution
~~~~~~~~~~~~~~~~~~

Arrays are distributed in one-dimensional block (`1D_Block`) manner
among processors. This means that processors own equal chunks of each
distributed array, except possibly the last processor.
Multi-dimensional arrays are distributed along their first dimension by default.
For example, chunks of rows are distributed for a 2D matrix.
The figure below
illustrates the distribution of a 9-element one-dimensional Numpy array, as well
as a 9 by 2 array, on three processors:

.. image:: ../figs/dist.jpg
    :height: 500
    :width: 500
    :scale: 60
    :alt: distribution of 1D array
    :align: center

HPAT replicates the arrays that are not distributed.
This is called `REP` distribution for consistency.

Argument and Return Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HPAT assumes argument and return variables to jitted functions are
replicated. However, the user can annotate these variables to indicate
distributed data. In this case,
the user is responsile for handling of the distributed data chunks outside
the HPAT scope. For example, the data can come from other jitted functions::

    @hpat.jit(distributed={'A'})
    def example_return(n):
        A = np.arange(n)
        return A

    @hpat.jit(distributed={'B'})
    def example_arg(B):
        return B.sum()

    n = 100
    A = example_return(n)
    s = example_arg(A)

Distribution Report
~~~~~~~~~~~~~~~~~~~

The distributions found by HPAT can be printed using the
`hpat.distribution_report()` function. The distribution report for the above
example code is as follows::

    Array distributions:
        $A.23                1D_Block

    Parfor distributions:
        0                    1D_Block

This report suggests that the function has an array that is distributed in
1D_Block fashion. The variable name is renamed from `A` to `$A.23` through
the optimization passes. The report also suggests that there is a `parfor`
(data-parallel for loop) that is 1D_Block distributed.

Numpy dot() Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `np.dot` function has different distribution rules based on the number of
dimensions and the distributions of its input arrays. The example below
demonstrates two cases::

    @hpat.jit
    def example_dot(N, D):
        X = np.random.ranf((N, D))
        Y = np.random.ranf(N)
        w = np.dot(Y, X)
        z = np.dot(X, w)
        return z.sum()

    example_dot(1024, 10)
    hpat.distribution_report()

Here is the output of `hpat.distribution_report()`::

    Array distributions:
       $X.43                1D_Block
       $Y.45                1D_Block
       $w.44                REP

    Parfor distributions:
       0                    1D_Block
       1                    1D_Block
       2                    1D_Block

The first `dot` has a 1D array with `1D_Block` distribution as first input
(`Y`), while the second input is a 2D array with `1D_Block` distribution (`X`).
Hence, `dot` is a sum reduction across distributed datasets and therefore,
the output (`w`) is on the `reduce` side and is assiged `REP` distribution.

The second `dot` has a 2D array with `1D_Block` distribution (`X`) as first
input, while the second input is a REP array (`w`). Hence, the computation is
data-parallel across rows of `X`, which implies a `1D_Block` distibution for
output (`z`).

Variable `z` does not exist in the distribution report since
the compiler optimizations were able to eliminate it. Its values are generated
and consumed on-the-fly, without memory load/store overheads.

Supported Numpy Operations
--------------------------

Below is the list of the data-parallel Numpy operators that HPAT can optimize
and parallelize.

1. Numpy `element-wise` array operations:

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

2. Numpy reduction functions ``sum``, ``prod``, ``min``, ``max``, ``argmin``
   and ``argmax``. Currently, `int64` data type is not supported for
   ``argmin`` and ``argmax``.

3. Numpy array creation functions ``empty``, ``zeros``, ``ones``,
   ``empty_like``, ``zeros_like``, ``ones_like``, ``full_like``, ``copy``,
   ``arange`` and ``linspace``.

4. Random number generator functions: ``rand``, ``randn``,
   ``ranf``, ``random_sample``, ``sample``, ``random``,
   ``standard_normal``, ``chisquare``, ``weibull``, ``power``, ``geometric``,
   ``exponential``, ``poisson``, ``rayleigh``, ``normal``, ``uniform``,
   ``beta``, ``binomial``, ``f``, ``gamma``, ``lognormal``, ``laplace``,
   ``randint``, ``triangular``.

4. Numpy ``dot`` function between a matrix and a vector, or two vectors.

5. Numpy array comprehensions, such as::

    A = np.array([i**2 for i in range(N)])

Optional arguments are not supported unless if explicitly mentioned here.
For operations on multi-dimensional arrays, automatic broadcast of
dimensions of size 1 is not supported.


Explicit Parallel Loops
-----------------------

Sometimes explicit parallel loops are required since a program cannot be written
in terms of data-parallel operators easily.
In this case, one can use HPAT's ``prange`` in place of ``range`` to specify
that a loop can be parallelized. The user is required to make sure the
loop does not have cross iteration dependencies except for supported reductions.

The example below demonstrates a parallel loop with a reduction::

    from hpat import jit, prange
    @jit
    def prange_test(n):
        A = np.random.ranf(n)
        s = 0
        for i in prange(len(A)):
            s += A[i]
        return s

Currently, reductions using ``+=``, ``*=``, ``min``, and ``max`` operators are
supported.


File I/O
--------

Currently, HPAT supports I/O for the `HDF5 <http://www.h5py.org/>`_ and
`Parquet <http://parquet.apache.org/>`_ formats.
For HDF5, the syntax is the same as the `h5py <http://www.h5py.org/>`_ package.
For example::

    @hpat.jit
    def example():
        f = h5py.File("lr.hdf5", "r")
        X = f['points'][:]
        Y = f['responses'][:]

For Parquet, the syntax is the same as `pyarrow <https://arrow.apache.org/docs/python/>`_::

    import pyarrow.parquet as pq
    @hpat.jit
    def kde():
        t = pq.read_table('kde.parquet')
        df = t.to_pandas()
        X = df['points'].values

HPAT automatically parallelizes I/O of different nodes in a distributed setting
without any code changes.

HPAT needs to know the types of input arrays. If the file name is a constant
string, HPAT tries to look at the file at compile time and recognize the types.
Otherwise, the user is responsile for providing the types similar to
`Numba's typing syntax
<http://numba.pydata.org/numba-doc/latest/reference/types.html>`_. For
example::

     @hpat.jit(locals={'X': hpat.float64[:,:], 'Y': hpat.float64[:]})
     def example(file_name):
         f = h5py.File(file_name, "r")
         X = f['points'][:]
         Y = f['responses'][:]

Print
-----

Using ``print`` function is only supported for `REP` values. Print is called on
one processor only since all processors have the same copy.


Strings
-------

Currently, HPAT provides basic ASCII string support. Constant strings, equality
comparison of strings (``==`` and ``!=``), ``split`` function, extracting
characters (e.g. ``s[1]``), concatination, and convertion to `int` and `float`
are supported. Here are some examples::

    s = 'test_str'
    flag = (s == 'test_str')
    flag = (s != 'test_str')
    s_list = s.split('_')
    c = s[1]
    s = s+'_test'
    a = int('12')
    b = float('1.2')

Dictionaries
------------

HPAT supports basic integer dictionaries currently. ``DictIntInt`` is the type
for dictionaries with 64-bit integer keys and values, while ``DictInt32Int32``
is for 32-bit integer ones. Getting and setting values, ``pop`` and ``get``
operators, as well as ``min`` and ``max`` of keys is supported. For example::

    d = DictIntInt()
    d[2] = 3
    a = d[2]
    b = d.get(3, 0)
    d.pop(2)
    d[3] = 4
    a = min(d.keys())
