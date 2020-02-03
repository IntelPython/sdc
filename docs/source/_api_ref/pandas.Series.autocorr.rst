.. _pandas.Series.autocorr:

:orphan:

pandas.Series.autocorr
**********************

Compute the lag-N autocorrelation.

This method computes the Pearson correlation between
the Series and its shifted self.

:param lag:
    int, default 1
        Number of lags to apply before performing autocorrelation.

:return: float
    The Pearson correlation between self and self.shift(lag).



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

