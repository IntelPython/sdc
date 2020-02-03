.. _pandas.Series.str.wrap:

:orphan:

pandas.Series.str.wrap
**********************

Wrap long strings in the Series/Index to be formatted in
paragraphs with length less than a given width.

This method has the same keyword parameters and defaults as
:class:`textwrap.TextWrapper`.

:param width:
    int
        Maximum line width.

:param expand_tabs:
    bool, optional
        If True, tab characters will be expanded to spaces (default: True).

:param replace_whitespace:
    bool, optional
        If True, each whitespace character (as defined by string.whitespace)
        remaining after tab expansion will be replaced by a single space
        (default: True).

:param drop_whitespace:
    bool, optional
        If True, whitespace that, after wrapping, happens to end up at the
        beginning or end of a line is dropped (default: True).

:param break_long_words:
    bool, optional
        If True, then words longer than width will be broken in order to ensure
        that no lines are longer than width. If it is false, long words will
        not be broken, and some lines may be longer than width (default: True).

:param break_on_hyphens:
    bool, optional
        If True, wrapping will occur preferably on whitespace and right after
        hyphens in compound words, as it is customary in English. If false,
        only whitespaces will be considered as potentially good places for line
        breaks, but you need to set break_long_words to false if you want truly
        insecable words (default: True).

:return: Series or Index



.. warning::
    This feature is currently unsupported by Intel Scalable Dataframe Compiler

