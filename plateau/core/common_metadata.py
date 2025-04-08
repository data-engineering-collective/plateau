import difflib
import logging
import pprint
from collections.abc import Sequence
from copy import copy, deepcopy
from functools import reduce
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import simplejson
from minimalkv import KeyValueStore

from plateau.core import naming
from plateau.core._compat import load_json
from plateau.core.naming import SINGLE_TABLE
from plateau.core.utils import ensure_string_type
from plateau.serialization._parquet import PARQUET_VERSION
from plateau.serialization._util import schema_metadata_bytes_to_object

_logger = logging.getLogger()

__all__ = (
    "SchemaWrapper",
    "make_meta",
    "validate_compatible",
    "empty_dataframe_from_schema",
    "normalize_type",
    "normalize_column_order",
)


class SchemaWrapper:
    """Wrapper object for pyarrow.Schema to handle forwards and backwards
    compatibility."""

    def __init__(self, schema, origin: str | set[str]):
        if isinstance(origin, str):
            origin = {origin}
        elif isinstance(origin, set):
            origin = copy(origin)
        if not all(isinstance(s, str) for s in origin):
            raise TypeError("Schema origin elements must be strings.")

        self.__schema = schema
        self.__origin = origin
        self._schema_compat()

    def with_origin(self, origin: str | set[str]) -> "SchemaWrapper":
        """Create new SchemaWrapper with given origin.

        Parameters
        ----------
        origin:
            New origin.
        """
        return SchemaWrapper(self.__schema, origin)

    def _schema_compat(self):
        # https://issues.apache.org/jira/browse/ARROW-5104
        schema = self.__schema
        if self.__schema is not None and self.__schema.pandas_metadata is not None:
            pandas_metadata = schema.pandas_metadata
            index_cols = pandas_metadata["index_columns"]
            if len(index_cols) > 1:
                raise NotImplementedError("Treatment of MultiIndex not implemented.")

            for _, col in enumerate(index_cols):
                # Range index is now serialized using start/end information. This special treatment
                # removes it from the columns which is fine
                if isinstance(col, dict):
                    pass
                # other indices are still tracked as a column
                else:
                    index_level_ix = schema.get_field_index(col)
                    # this may happen for the schema of an empty df
                    if index_level_ix >= 0:
                        schema = schema.remove(index_level_ix)

            schema = schema.remove_metadata()
            md = {b"pandas": _dict_to_binary(pandas_metadata)}
            schema = schema.with_metadata(md)
            self.__schema = schema

    def internal(self):
        return self.__schema

    @property
    def origin(self) -> set[str]:
        return copy(self.__origin)

    def __repr__(self):
        return self.__schema.__repr__()

    def __eq__(self, other):
        return self.equals(other)

    def __ne__(self, other):
        return not self.equals(other)

    def __getstate__(self):
        return (_schema2bytes(self.__schema), self.__origin)

    def __setstate__(self, state):
        self.__schema = _bytes2schema(state[0])
        self.__origin = state[1]

    def __getattr__(self, attr):
        return getattr(self.__schema, attr)

    def __hash__(self):
        # see https://issues.apache.org/jira/browse/ARROW-2719
        return hash(_schema2bytes(self.__schema))

    def __getitem__(self, i):
        return self.__schema[i]

    def __len__(self):
        return len(self.__schema)

    def equals(self, other, check_metadata=False):
        if isinstance(other, SchemaWrapper):
            return self.__schema.equals(other.__schema, check_metadata)
        else:
            return self.__schema.equals(other, check_metadata)

    equals.__doc__ = pa.Schema.equals.__doc__

    def remove(self, i):
        return SchemaWrapper(self.__schema.remove(i), self.__origin)

    remove.__doc__ = pa.Schema.set.__doc__

    def remove_metadata(self):
        return SchemaWrapper(
            self.__schema.remove_metadata(),
            {s + "__no_metadata" for s in self.__origin},
        )

    remove_metadata.__doc__ = pa.Schema.remove_metadata.__doc__

    def set(self, i, field):
        return SchemaWrapper(self.__schema.set(i, field), self.__origin)

    set.__doc__ = pa.Schema.set.__doc__


def gen_metadata(schema: SchemaWrapper) -> dict[str, Any]:
    assert isinstance(
        schema.internal(), pa.Schema
    ), "Internal schema must be a pyarrow schema"

    pandas_metadata = {
        "columns": [],
        "index_columns": [],  # this function is only called if no pandas metadata is present, ergo we do not have any index columns
        "pandas_version": pd.__version__,
        "creator": {
            "library": "plateau/core/common_metadata.py:gen_metadata",
            # instead of {'library': 'pyarrow', 'version': '19.0.1'}
        },
    }

    for field in schema:
        pandas_metadata["columns"].append(
            {
                "name": field.name,
                "field_name": field.name,
                # the following fields are NOT accessed when resorting the columns
                # "pandas_type": str(field.type),  # optional
                # "numpy_type": str(field.type),  # optional
                # "metadata": field.metadata,  # optional: decode if needed
            }
        )

    return pandas_metadata
    # > {'columns': [{'name': 'A', 'field_name': 'A'}, {'name': 'B', 'field_name': 'B'}, {'name': 'C', 'field_name': 'C'}, {'name': 'D', 'field_name': 'D'}, {'name': 'E', 'field_name': 'E'}, {'name': 'F', 'field_name': 'F'}], 'index_columns': [], 'pandas_version': '2.2.3'}


def normalize_column_order(schema, partition_keys=None):
    """Normalize column order in schema.

    Columns are sorted in the following way:

    1. Partition keys (as provided by ``partition_keys``)
    2. DataFrame columns in alphabetic order
    3. Remaining fields as generated by pyarrow, mostly index columns

    Parameters
    ----------
    schema: SchemaWrapper
        Schema information for DataFrame.
    partition_keys: Union[None, List[str]]
        Partition keys used to split the dataset.

    Returns
    -------
    schema: SchemaWrapper
        Schema information for DataFrame.
    """
    if not isinstance(schema, SchemaWrapper):
        schema = SchemaWrapper(schema, "__unknown__")

    if partition_keys is None:
        partition_keys = []
    else:
        partition_keys = list(partition_keys)

    pandas_metadata = schema.pandas_metadata or gen_metadata(schema)
    origin = schema.origin

    cols_partition = {}
    cols_payload = []
    cols_misc = []

    for cmd in pandas_metadata["columns"]:
        name = cmd.get("name")
        field_name = cmd["field_name"]
        field_idx = schema.get_field_index(field_name)

        if field_idx >= 0:
            field = schema[field_idx]
        else:
            field = None

        if name is None:
            cols_misc.append((cmd, field))
        elif name in partition_keys:
            cols_partition[name] = (cmd, field)
        else:
            cols_payload.append((name, cmd, field))

    ordered = []
    for k in partition_keys:
        if k in cols_partition:
            ordered.append(cols_partition[k])

    ordered += [(cmd, f) for _name, cmd, f in sorted(cols_payload, key=lambda x: x[0])]
    ordered += cols_misc

    pandas_metadata["columns"] = [cmd for cmd, _ in ordered]
    fields = [f for _, f in ordered if f is not None]

    metadata = schema.metadata or {}  # fallback
    metadata[b"pandas"] = _dict_to_binary(pandas_metadata)

    pa_schema = pa.schema(fields, metadata)
    return SchemaWrapper(pa_schema, origin)


def make_meta(obj, origin, partition_keys=None):
    """Create metadata object for DataFrame.

    .. note::
        This function can, for convenience reasons, also be applied to schema objects in which case they are just
        returned.

    .. warning::
        Information for categoricals will be stripped!

    :meth:`normalize_type` will be applied to normalize type information and :meth:`normalize_column_order` will be
    applied to to reorder column information.

    Parameters
    ----------
    obj: Union[DataFrame, Schema]
        Object to extract metadata from.
    origin: str
        Origin of the schema data, used for debugging and error reporting.
    partition_keys: Union[None, List[str]]
        Partition keys used to split the dataset.

    Returns
    -------
    schema: SchemaWrapper
        Schema information for DataFrame.
    """
    if isinstance(obj, SchemaWrapper):
        return obj
    elif isinstance(obj, pa.Schema):
        return normalize_column_order(
            SchemaWrapper(obj, origin), partition_keys=partition_keys
        )
    elif isinstance(obj, pa.Table):
        return obj.schema

    # normalize_column_order(
    #         SchemaWrapper(obj.schema, origin), partition_keys=partition_keys
    #     )

    if not isinstance(obj, pd.DataFrame):
        raise ValueError("Input must be a pyarrow schema, or a pandas dataframe")

    schema = pa.Schema.from_pandas(obj)
    pandas_metadata = schema.pandas_metadata

    # normalize types
    fields = {field.name: field.type for field in schema}
    for cmd in pandas_metadata["columns"]:
        name = cmd.get("name")
        if name is None:
            continue
        field_name = cmd["field_name"]
        field_idx = schema.get_field_index(field_name)
        field = schema[field_idx]
        (
            fields[field_name],
            cmd["pandas_type"],
            cmd["numpy_type"],
            cmd["metadata"],
        ) = normalize_type(
            field.type, cmd["pandas_type"], cmd["numpy_type"], cmd["metadata"]
        )
    metadata = schema.metadata
    metadata[b"pandas"] = _dict_to_binary(pandas_metadata)
    schema = pa.schema([pa.field(n, t) for n, t in fields.items()], metadata)
    return normalize_column_order(SchemaWrapper(schema, origin), partition_keys)


def normalize_type(
    t_pa: pa.DataType,
    t_pd: str | None,
    t_np: str | None,
    metadata: dict[str, Any] | None,
) -> tuple[pa.DataType, str | None, str | None, dict[str, Any] | None]:
    """This will normalize types as followed:

    - all signed integers (``int8``, ``int16``, ``int32``, ``int64``) will be converted to ``int64``
    - all unsigned integers (``uint8``, ``uint16``, ``uint32``, ``uint64``) will be converted to ``uint64``
    - all floats (``float32``, ``float64``) will be converted to ``float64``
    - all list value types will be normalized (e.g. ``list[int16]`` to ``list[int64]``, ``list[list[uint8]]`` to
      ``list[list[uint64]]``)
    - all dict value types will be normalized (e.g. ``dictionary<values=float32, indices=int16, ordered=0>`` to
      ``float64``)

    Parameters
    ----------
    t_pa
        pyarrow type object, e.g. ``pa.list_(pa.int8())``.
    t_pd
        pandas type identifier, e.g. ``"list[int8]"``.
    t_np
        numpy type identifier, e.g. ``"object"``.
    metadata
        metadata associated with the type, e.g. information about categorials.
    """
    if pa.types.is_signed_integer(t_pa):
        return pa.int64(), "int64", "int64", None
    elif pa.types.is_unsigned_integer(t_pa):
        return pa.uint64(), "uint64", "uint64", None
    elif pa.types.is_floating(t_pa):
        return pa.float64(), "float64", "float64", None
    elif pa.types.is_list(t_pa):
        assert t_pd is not None
        t_pa2, t_pd2, t_np2, metadata2 = normalize_type(
            t_pa.value_type, t_pd[len("list[") : -1], None, None
        )
        return pa.list_(t_pa2), f"list[{t_pd2}]", "object", None
    elif pa.types.is_dictionary(t_pa):
        # downcast to dictionary content, `t_pd` is useless in that case
        return normalize_type(t_pa.value_type, t_np, t_np, None)
    else:
        return t_pa, t_pd, t_np, metadata


def _get_common_metadata_key(dataset_uuid, table):
    return f"{dataset_uuid}/{table}/{naming.TABLE_METADATA_FILE}"


def read_schema_metadata(
    dataset_uuid: str, store: KeyValueStore, table: str = SINGLE_TABLE
) -> SchemaWrapper:
    """Read schema and metadata from store.

    Parameters
    ----------
    dataset_uuid
        Unique ID of the dataset in question.
    store
        Object that implements `.get(key)` to read data.
    table
        Table to read metadata for.

    Returns
    -------
    schema: Schema
        Schema information for DataFrame/table.
    """
    key = _get_common_metadata_key(dataset_uuid=dataset_uuid, table=table)
    return SchemaWrapper(_bytes2schema(store.get(key)), key)


def store_schema_metadata(
    schema: SchemaWrapper,
    dataset_uuid: str,
    store: KeyValueStore,
    table: str = SINGLE_TABLE,
) -> str:
    """Store schema and metadata to store.

    Parameters
    ----------
    schema
        Schema information for DataFrame/table.
    dataset_uuid
        Unique ID of the dataset in question.
    store
        Object that implements `.put(key, data)` to write data.
    table
        Table to write metadata for.

    Returns
    -------
    key: str
        Key to which the metadata was written to.
    """
    key = _get_common_metadata_key(dataset_uuid=dataset_uuid, table=table)
    return store.put(key, _schema2bytes(schema.internal()))


def _schema2bytes(schema: SchemaWrapper) -> bytes:
    buf = pa.BufferOutputStream()
    pq.write_metadata(schema, buf, version=PARQUET_VERSION, coerce_timestamps="us")
    return buf.getvalue().to_pybytes()


def _bytes2schema(data: bytes) -> SchemaWrapper:
    reader = pa.BufferReader(data)
    schema = pq.read_schema(reader)
    fields = []
    for idx in range(len(schema)):
        f = schema[idx]

        # schema data recovered from parquet always contains timestamp data in us-granularity, but pandas will use
        # ns-granularity, so we re-align the two different worlds here
        if f.type == pa.timestamp("us"):
            f = pa.field(f.name, pa.timestamp("ns"))

        fields.append(f)
    return pa.schema(fields, schema.metadata)


def _pandas_in_schemas(schemas):
    """Check if any schema contains pandas metadata."""
    has_pandas = False
    for schema in schemas:
        if schema.metadata and b"pandas" in schema.metadata:
            has_pandas = True
    return has_pandas


def _determine_schemas_to_compare(
    schemas: Sequence[SchemaWrapper], ignore_pandas: bool
) -> tuple[SchemaWrapper | None, list[tuple[SchemaWrapper, list[str]]]]:
    """Iterate over a list of `pyarrow.Schema` objects and prepares them for
    comparison by picking a reference and determining all null columns.

    .. note::

        If pandas metadata exists, the version stored in the metadata is overwritten with the currently
        installed version since we expect to stay backwards compatible

    Returns
    -------
    reference: Schema
        A reference schema which is picked from the input list. The reference schema is guaranteed
        to be a schema having the least number of null columns of all input columns. The set of null
        columns is guaranteed to be a true subset of all null columns of all input schemas. If no such
        schema can be found, an Exception is raised
    list_of_schemas: List[Tuple[Schema, List]]
        A list holding pairs of (Schema, null_columns) where the null_columns are all columns which are null and
        must be removed before comparing the schemas
    """
    has_pandas = _pandas_in_schemas(schemas) and not ignore_pandas
    schemas_to_evaluate: list[tuple[SchemaWrapper, list[str]]] = []
    reference = None
    null_cols_in_reference = set()

    for schema in set(schemas):
        if not isinstance(schema, SchemaWrapper):
            schema = SchemaWrapper(schema, "__unknown__")

        if has_pandas:
            metadata = schema.metadata
            if metadata is None or b"pandas" not in metadata:
                raise ValueError(
                    "Pandas and non-Pandas schemas are not comparable. "
                    "Use ignore_pandas=True if you only want to compare "
                    "on Arrow level."
                )
            pandas_metadata = load_json(metadata[b"pandas"].decode("utf8"))

            # we don't care about the pandas version, since we assume it's safe
            # to read datasets that were written by older or newer versions.
            pandas_metadata["pandas_version"] = f"{pd.__version__}"

            metadata_clean = deepcopy(metadata)
            metadata_clean[b"pandas"] = _dict_to_binary(pandas_metadata)
            current = SchemaWrapper(pa.schema(schema, metadata_clean), schema.origin)
        else:
            current = schema

        # If a field is null we cannot compare it and must therefore reject it
        null_columns = {field.name for field in current if field.type == pa.null()}  # type: ignore

        # Determine a valid reference schema. A valid reference schema is considered to be the schema
        # of all input schemas with the least empty columns.
        # The reference schema ought to be a schema whose empty columns are a true subset for all sets
        # of empty columns. This ensures that the actual reference schema is the schema with the most
        # information possible. A schema which doesn't fulfil this requirement would weaken the
        # comparison and would allow for false positives

        # Trivial case
        if reference is None:
            reference = current
            null_cols_in_reference = null_columns
        # The reference has enough information to validate against current schema.
        # Append it to the list of schemas to be verified
        elif null_cols_in_reference.issubset(null_columns):
            schemas_to_evaluate.append((current, null_columns))
        # current schema includes all information of reference and more.
        # Add reference to schemas_to_evaluate and update reference
        elif null_columns.issubset(null_cols_in_reference):
            schemas_to_evaluate.append((reference, null_cols_in_reference))
            reference = current
            null_cols_in_reference = null_columns
        # If there is no clear subset available elect the schema with the least null columns as `reference`.
        # Iterate over the null columns of `reference` and replace it with a non-null field of the `current`
        # schema which recovers the loop invariant (null columns of `reference` is subset of `current`)
        else:
            if len(null_columns) < len(null_cols_in_reference):
                reference, current = current, reference
                null_cols_in_reference, null_columns = (
                    null_columns,
                    null_cols_in_reference,
                )

            for col in null_cols_in_reference - null_columns:
                # Enrich the information in the reference by grabbing the missing fields
                # from the current iteration. This assumes that we only check for global validity and
                # isn't relevant where the reference comes from.
                reference = _swap_fields_by_name(reference, current, col)
                null_cols_in_reference.remove(col)
            schemas_to_evaluate.append((current, null_columns))

    assert (reference is not None) or (not schemas_to_evaluate)

    return reference, schemas_to_evaluate


def _swap_fields_by_name(reference, current, field_name):
    current_field = current.field(field_name)
    reference_index = reference.get_field_index(field_name)
    return reference.set(reference_index, current_field)


def _strip_columns_from_schema(schema, field_names):
    stripped_schema = schema

    for name in field_names:
        ix = stripped_schema.get_field_index(name)
        if ix >= 0:
            stripped_schema = stripped_schema.remove(ix)
        else:
            # If the returned index is negative, the field doesn't exist in the schema.
            # This is most likely an indicator for incompatible schemas and we refuse to strip the schema
            # to not obfurscate the validation result
            _logger.warning(
                f"Unexpected field `{name}` encountered while trying to strip `null` columns.\n"
                f"Schema was:\n\n`{schema}`"
            )
            return schema
    return stripped_schema


def _remove_diff_header(diff):
    diff = list(diff)
    for ix, el in enumerate(diff):
        # This marks the first actual entry of the diff
        # e.g. @@ -1,5 + 2,5 @@
        if el.startswith("@"):
            return diff[ix:]
    return diff


def _diff_schemas(first, second):
    # see https://issues.apache.org/jira/browse/ARROW-4176

    first_pyarrow_info = str(first.remove_metadata())
    second_pyarrow_info = str(second.remove_metadata())
    pyarrow_diff = _remove_diff_header(
        difflib.unified_diff(
            str(first_pyarrow_info).splitlines(), str(second_pyarrow_info).splitlines()
        )
    )

    first_pandas_info = first.pandas_metadata
    second_pandas_info = second.pandas_metadata
    pandas_meta_diff = _remove_diff_header(
        difflib.unified_diff(
            pprint.pformat(first_pandas_info).splitlines(),
            pprint.pformat(second_pandas_info).splitlines(),
        )
    )

    diff_string = (
        "Arrow schema:\n"
        + "\n".join(pyarrow_diff)
        + "\n\nPandas_metadata:\n"
        + "\n".join(pandas_meta_diff)
    )

    return diff_string


def validate_compatible(schemas, ignore_pandas=False):
    """Validate that all schemas in a given list are compatible.

    Apart from the pandas version preserved in the schema metadata, schemas must be completely identical. That includes
    a perfect match of the whole metadata (except the pandas version) and pyarrow types.

    Use :meth:`make_meta` and :meth:`normalize_column_order` for type and column order normalization.

    In the case that all schemas don't contain any pandas metadata, we will check the Arrow
    schemas directly for compatibility.

    Parameters
    ----------
    schemas: List[Schema]
        Schema information from multiple sources, e.g. multiple partitions. List may be empty.
    ignore_pandas: bool
        Ignore the schema information given by Pandas an always use the Arrow schema.

    Returns
    -------
    schema: SchemaWrapper
        The reference schema which was tested against

    Raises
    ------
    ValueError
        At least two schemas are incompatible.
    """
    reference, schemas_to_evaluate = _determine_schemas_to_compare(
        schemas, ignore_pandas
    )

    for current, null_columns in schemas_to_evaluate:
        # We have schemas so the reference schema should be non-none.
        assert reference is not None
        # Compare each schema to the reference but ignore the null_cols and the Pandas schema information.
        reference_to_compare = _strip_columns_from_schema(
            reference, null_columns
        ).remove_metadata()
        current_to_compare = _strip_columns_from_schema(
            current, null_columns
        ).remove_metadata()

        def _fmt_origin(origin):
            origin = sorted(origin)
            # dask cuts of exception messages at 1k chars:
            #   https://github.com/dask/distributed/blob/6e0c0a6b90b1d3c/distributed/core.py#L964
            # therefore, we cut the the maximum length
            max_len = 200
            inner_msg = ", ".join(origin)
            ellipsis = "..."
            if len(inner_msg) > max_len + len(ellipsis):
                inner_msg = inner_msg[:max_len] + ellipsis
            return f"{{{inner_msg}}}"

        if reference_to_compare != current_to_compare:
            schema_diff = _diff_schemas(reference, current)
            exception_message = f"""Schema violation

Origin schema: {_fmt_origin(current.origin)}
Origin reference: {_fmt_origin(reference.origin)}

Diff:
{schema_diff}

Reference schema:
{str(reference)}"""
            raise ValueError(exception_message)

    # add all origins to result AFTER error checking, otherwise the error message would be pretty misleading due to the
    # reference containing all origins.
    if reference is None:
        return None
    else:
        return reference.with_origin(
            reduce(
                set.union,
                (schema.origin for schema, _null_columns in schemas_to_evaluate),
                reference.origin,
            )
        )


def validate_shared_columns(schemas, ignore_pandas=False):
    """Validate that columns that are shared amongst schemas are compatible.

    Only DataFrame columns are taken into account, other fields (like index data) are ignored. The following data must
    be an exact match:

    - metadata (as stored in the ``"columns"`` list of the ``b'pandas'`` schema metadata)
    - pyarrow type (that means that e.g. ``int8`` and ``int64`` are NOT compatible)

    Columns that are only present in a subset of the provided schemas must only be compatible for that subset, i.e.
    non-existing columns are ignored. The order of the columns in the provided schemas is irrelevant.

    Type normalization should be handled by :meth:`make_meta`.

    In the case that all schemas don't contain any pandas metadata, we will check the Arrow
    schemas directly for compatibility. Then the metadata information will not be checked
    (as it is non-existent).

    Parameters
    ----------
    schemas: List[Schema]
        Schema information from multiple sources, e.g. multiple tables. List may be empty.
    ignore_pandas: bool
        Ignore the schema information given by Pandas an always use the Arrow schema.

    Raises
    ------
    ValueError
        Incompatible columns were found.
    """
    seen: dict[str, Any] = {}
    has_pandas = _pandas_in_schemas(schemas) and not ignore_pandas

    for schema in schemas:
        if has_pandas:
            metadata = schema.metadata
            if metadata is None or b"pandas" not in metadata:
                raise ValueError(
                    "Pandas and non-Pandas schemas are not comparable. "
                    "Use ignore_pandas=True if you only want to compare "
                    "on Arrow level."
                )
            pandas_metadata = load_json(metadata[b"pandas"].decode("utf8"))

            columns = []
            for cmd in pandas_metadata["columns"]:
                name = cmd.get("name")
                if name is None:
                    continue
                columns.append(cmd["field_name"])
        else:
            columns = schema.names

        for col in columns:
            field_idx = schema.get_field_index(col)
            field = schema[field_idx]
            obj = (field, col)
            if col in seen:
                ref = seen[col]
                if pa.types.is_null(ref[0].type) or pa.types.is_null(field.type):
                    continue
                if ref != obj:
                    raise ValueError(
                        f'Found incompatible entries for column "{col}"\n{ref}\n{obj}'
                    )
            else:
                seen[col] = obj


def _dict_to_binary(dct):
    return simplejson.dumps(dct, sort_keys=True).encode("utf8")


def empty_dataframe_from_schema(
    schema, columns=None, date_as_object=False, coerce_temporal_nanoseconds=True
):
    """Create an empty DataFrame from provided schema.

    Parameters
    ----------
    schema: Schema
        Schema information of the new empty DataFrame.
    columns: Union[None, List[str]]
        Optional list of columns that should be part of the resulting DataFrame. All columns in that list must also be
        part of the provided schema.
    date_as_object: bool
        Cast dates to objects.
    coerce_temporal_nanoseconds: bool
        Coerce date32, date64, duration and timestamp units to nanoseconds to retain behaviour of pandas 1.x.
        Only applicable to pandas version >= 2.0 and PyArrow version >= 13.0.0.

    Returns
    -------
    DataFrame
        Empty DataFrame with requested columns and types.
    """
    # HACK: Cast bytes to object in metadata until Pandas bug is fixed: https://github.com/pandas-dev/pandas/issues/50127
    schema = schema_metadata_bytes_to_object(schema.internal())

    _coerce = {"coerce_temporal_nanoseconds": coerce_temporal_nanoseconds}
    df = schema.empty_table().to_pandas(date_as_object=date_as_object, **_coerce)

    df.columns = df.columns.map(ensure_string_type)
    if columns is not None:
        df = df[columns]

    return df
