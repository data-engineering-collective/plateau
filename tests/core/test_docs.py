import inspect
import textwrap
from collections import defaultdict

import pytest

from plateau.core.docs import _PARAMETER_MAPPING, default_docs
from plateau.io.dask.bag import (
    build_dataset_indices__bag,
    read_dataset_as_dataframe_bag,
    read_dataset_as_metapartitions_bag,
    store_bag_as_dataset,
)
from plateau.io.dask.dataframe import (
    read_dataset_as_ddf,
    store_dataset_from_ddf,
    update_dataset_from_ddf,
)
from plateau.io.dask.delayed import (
    delete_dataset__delayed,
    read_dataset_as_delayed,
    read_dataset_as_delayed_metapartitions,
    store_delayed_as_dataset,
    update_dataset_from_delayed,
)
from plateau.io.eager import (
    build_dataset_indices,
    commit_dataset,
    create_empty_dataset_header,
    delete_dataset,
    garbage_collect_dataset,
    read_dataset_as_dataframes,
    read_dataset_as_metapartitions,
    read_table,
    store_dataframes_as_dataset,
    update_dataset_from_dataframes,
    write_single_partition,
)
from plateau.io.iter import (
    read_dataset_as_dataframes__iterator,
    read_dataset_as_metapartitions__iterator,
    store_dataframes_as_dataset__iter,
    update_dataset_from_dataframes__iter,
)


@pytest.mark.parametrize(
    "function",
    [
        read_dataset_as_metapartitions_bag,
        read_dataset_as_dataframe_bag,
        store_bag_as_dataset,
        build_dataset_indices__bag,
        read_dataset_as_ddf,
        store_dataset_from_ddf,
        update_dataset_from_ddf,
        delete_dataset__delayed,
        read_dataset_as_delayed_metapartitions,
        read_dataset_as_delayed,
        update_dataset_from_delayed,
        store_delayed_as_dataset,
        delete_dataset,
        read_dataset_as_dataframes,
        read_dataset_as_metapartitions,
        read_table,
        commit_dataset,
        store_dataframes_as_dataset,
        create_empty_dataset_header,
        write_single_partition,
        update_dataset_from_dataframes,
        build_dataset_indices,
        garbage_collect_dataset,
        read_dataset_as_metapartitions__iterator,
        read_dataset_as_dataframes__iterator,
        update_dataset_from_dataframes__iter,
        store_dataframes_as_dataset__iter,
    ],
)
def test_docs(function):
    docstrings = function.__doc__
    arguments = inspect.signature(function).parameters
    valid_docs = defaultdict(set)
    for arg in arguments:
        valid = _PARAMETER_MAPPING.get(arg, "Parameters") in docstrings
        if not valid:
            indented = textwrap.indent(
                _PARAMETER_MAPPING.get(arg, "Parameters"), " " * 4, lambda line: True
            )
            valid = indented.lstrip() in docstrings
        valid_docs[valid].add(arg)

    assert valid_docs[True]
    if valid_docs[False]:
        raise AssertionError(
            f"Wrong or missing docstrings for parameters {valid_docs[False]}.\n\n{docstrings}"
        )

    sorted_arguments = sorted(arguments)
    args_in_docs = [argument in docstrings for argument in sorted_arguments]

    assert all(args_in_docs), [
        sorted_arguments[ix] for ix, val in enumerate(args_in_docs) if val is False
    ]


def test_docs_duplicity():
    # This test ensures that if a keyword argument has been listed out in the docs as well as under _PARAMETER_MAPPING
    # then the parser does not end up pick the one under _PARAMETER_MAPPING
    @default_docs
    def dummy_function(store):
        """This is a dummy_function.

        Parameters
        -----------
        store: str
            This is an argument
        """

    resultant_docstrings = str(dummy_function.__doc__)
    assert "This is a dummy_function" in resultant_docstrings
    assert "This is an argument" in resultant_docstrings
    assert "Factory function producing a KeyValueStore" not in resultant_docstrings
