.. _performance:

Getting Performance With Intel® SDC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compilation Overheads
======================

.. todo::
    Discuss tradeoffs of the compilation vs. staying in interpreter/object mode. Demonstrate effects of the compilation time overhead, boxing/unboxing overheads. Explain why boxing and unboxing is required. Explain difference between Pandas dataframe structure layout and internal hpat hi-frame layout, e.g. on the example of unboxing strings. Discuss the difference with Apache Arrow approach
 
Automatic Parallelization
==========================

.. todo::
    Types of supported parallelism. Multiprocessing (MPI) and multi-threading (TBB, OMP, built-in scheduler), implications. Controls, decorators, hybrid parallelism. Tradeoffs choosing the type of parallelism
