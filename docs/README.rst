Intel® SDC docstring decoration rules
#####################################

Since Intel® SDC API Reference is auto-generated from respective `Pandas*`_ and Intel® SDC docstrings there are certain rules that must be
followed to accurately generate the API description.

1. Every Intel® SDC API must have the docstring.
    If developer does not provide the docstring then `Sphinx*`_ will not be able to match `Pandas*`_ docstring with respective SDC one.
    In this situation `Sphinx*`_ assumes that SDC does not support such API and will include respective note in the API Reference that
    **This API is currently unsupported**.

2. Follow 'one function - one docstring' rule.
    You cannot have one docstring for multiple APIs, even if those are very similar. Auto-generator assumes every SDC API is covered by
    respective docstring. If `Sphinx*`_ does not find the docstring for particular API then it assumes that SDC does not support such API
    and will include respective note in the API Reference that **This API is currently unsupported**.

3. Description (introductory section, the very first few paragraphs without a title) is taken from `Pandas*`_.
    Intel® SDC developers should not include API description in SDC docstring.
    But developers are encouraged to follow Pandas API description naming conventions
    so that the combined docstring appears consistent.

4. Parameters, Returns, and Raises sections' description is taken from `Pandas*`_ docstring.
    Intel® SDC developers should not include such descriptions in their SDC docstrings.
    Rather developers are encouraged to follow Pandas naming conventions
    so that the combined docstring appears consistent.

5. Every SDC docstring must be of the follwing structure:
    ::

        """
        Intel Scalable Dataframe Compiler User Guide
        ********************************************
        Pandas API: <full pandas name, e.g. pandas.Series.nlargest>

        <Intel® SDC specific sections>

        Intel Scalable Dataframe Compiler Developer Guide
        *************************************************
        <Developer's Guide specific sections>
        """

The first two lines must be the User Guide header. This is an indication to `Sphinx*`_ that this section is intended for public API
and it will be combined with repsective Pandas API docstring.

Line 3 must specify what Pandas API this Intel® SDC docstring does correspond to. It must start with ``Pandas API:`` followed by
full Pandas API name that corresponds to this SDC docstring. Remember to include full name, for example, ``nlargest`` is not
sufficient for auto-generator to perform the match. The full name must be ``pandas.Series.nlargest``.

After User Guide sections in the docstring there can be another header indicating that the remaining part of the docstring belongs to
Developer's Guide and must not be included into User's Guide.

6. Examples, See Also, References sections are **NOT** taken from Pandas docstring. SDC developers are expected to complete these sections in SDC doctrings.
    This is so because respective Pandas sections are sometimes too Pandas specific and are not relevant to SDC. SDC developers have to
    rewrite those sections in Intel® SDC style. Do not forget about User Guide header and Pandas API name prior to adding SDC specific
    sections.

7. Examples section is mandatory for every SDC API. 'One API - at least one example' rule is applied.
    Examples are essential part of user experience and must accompany every API docstring.

8. Embed examples into Examples section from ``./sdc/examples``.
    Rather than writing example in the docstring (which is error-prone) embed relevant example scripts into the docstring. For example,
    here is an example how to embed example for ``pandas.Series.get()`` function into respective Intel® SDC docstring:

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
    for every example. ``:name:`` is the `Sphinx*`_ name that allows referencing example from other parts of the documentation. It is a good
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

    Pay attention to indentation and required blank lines. `Sphinx*`_ is very sensitive to that.

11. If SDC API does not support all variations of respective Pandas API then Limitations section is mandatory.
    While there is not specific guideline how Limitations section must be written, a good style is to follow Pandas Parameters section
    description style and naming conventions.

12. Before committing your code for public SDC API you are expected to:

    - have SDC docstring implemented;
    - have respective SDC examples implemented and tested
    - API Reference documentation generated and visually inspected. New warnings in the documentation build are not allowed.

Sphinx Generation Internals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The documentation generation is controlled by ``conf.py`` script automatically invoked by `Sphinx*`_.
See `Sphinx documentation <http://www.sphinx-doc.org/en/master/usage/configuration.html>`_ for details.

The API Reference for Intel® SDC User's Guide is auto-generated by inspecting ``pandas`` and ``sdc`` modules.
That's why these modules must be pre-installed for documentation generation using `Sphinx*`_.
However, there is a possibility to skip API Reference auto-generation by setting environment variable ``SDC_DOC_NO_API_REF_STR=1``.

If the environment variable ``SDC_DOC_NO_API_REF_STR`` is unset then Sphinx's ``conf.py``
invokes ``generate_api_reference()`` function located in ``./sdc/docs/source/buildscripts/apiref_generator`` module.
This function parses ``pandas`` and ``sdc`` docstrings for each API, combines those into single docstring and
writes it into RST file with respective `Pandas*`_ API name. The auto-generated RST files are
located at ``./sdc/docs/source/_api_ref`` directory.

.. note::
    `Sphinx*`_ will automatically clean the ``_api_ref`` directory on the next invocation of the documenation build.

