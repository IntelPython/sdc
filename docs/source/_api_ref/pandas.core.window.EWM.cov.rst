.. _pandas.core.window.EWM.cov:

:orphan:

pandas.core.window.EWM.cov
**************************

Exponential weighted sample covariance.

:param other:
    Series, DataFrame, or ndarray, optional
        If not supplied then will default to self and produce pairwise
        output.

:param pairwise:
    bool, default None
        If False then only matching columns between self and other will be
        used and the output will be a DataFrame.
        If True then all pairwise combinations will be calculated and the
        output will be a MultiIndex DataFrame in the case of DataFrame
        inputs. In the case of missing elements, only complete pairwise
        observations will be used.

:param bias:
    bool, default False
        Use a standard estimation bias correction.
        \*\*kwargs
        Keyword arguments to be passed into func.

        :return: Series or DataFrame
            Return type is determined by the caller.



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

