import pyarrow as pa
import pyarrow.compute as pc


def empty_table_from_schema(schema: pa.Schema, columns=None) -> pa.Table:
    """Create an empty pyarrow.Table from a schema, optionally selecting only certain columns."""
    if columns is not None:
        # Build a new schema using only the specified column names
        fields = [schema.field(name) for name in columns if name in schema.names]
        schema = pa.schema(fields)
    arrays = [pa.array([], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def cast_categoricals_to_dictionary(
    table: pa.Table, categoricals: list[str]
) -> pa.Table:
    """Cast specified columns in a table to dictionary (categorical) type if they aren't already."""
    for col in categoricals:
        col_index = table.schema.get_field_index(col)
        if col_index == -1:
            continue  # skip if the column doesn't exist
        column = table.column(col_index)
        if not pa.types.is_dictionary(column.type):
            # Create a dictionary type with an int32 index and the same value type as the original column
            dict_type = pa.dictionary(index_type=pa.int32(), value_type=column.type)
            column = column.cast(dict_type)
            table = table.set_column(col_index, col, column)
    return table


def align_categories(tables: list[pa.Table], categoricals: list[str]) -> list[pa.Table]:
    """
    Aligns categorical (dictionary encoded) columns across a list of pyarrow Tables.

    For each column in `categoricals` it computes the union of dictionary values
    across all tables. It then uses the dictionary of the largest table (by row count)
    as the baseline order and appends any additional values (sorted) to it.

    Each table’s column is recoded so that the indices refer to the union dictionary.

    Parameters
    ----------
    tables : List[pa.Table]
        A list of pyarrow Tables.
    categoricals : List[str]
        List of column names that should be treated as categorical and aligned.

    Returns
    -------
    List[pa.Table]
        The list of tables with aligned categorical columns.
    """
    if not categoricals:
        return tables

    # Process each categorical column
    for column in categoricals:
        all_types = [table[column].type for table in tables]

        union_values = set()
        baseline_categories = None
        baseline_num_rows = -1

        # First pass: for this column, compute the union of all categories
        # and choose the baseline ordering from the largest table.
        for table in tables:
            if column not in table.column_names:
                continue

            col = table[column]
            # Ensure the column is dictionary encoded.
            # if not pa.types.is_dictionary(col.type):
            #     col = pc.dictionary_encode(col)

            # Combine chunks to get a single Array (if needed)
            col_combined = (
                col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
            )
            # Extract the dictionary as a Python list.
            cats = col_combined.dictionary.to_pylist()
            union_values.update(cats)
            if table.num_rows > baseline_num_rows:
                baseline_num_rows = table.num_rows
                baseline_categories = cats

        if baseline_categories is None:
            # No table contained this column.
            continue

        # Build the new dictionary order: use the baseline order then add any additional values
        extra = union_values - set(baseline_categories)
        new_dictionary = baseline_categories + sorted(extra)
        # Build a lookup map for quick conversion: value -> new index
        union_map = {val: idx for idx, val in enumerate(new_dictionary)}

        # Second pass: recast the column in every table to use the new dictionary
        new_tables = []
        for table in tables:
            if column not in table.column_names:
                new_tables.append(table)
                continue

            col = table[column]
            if not pa.types.is_dictionary(col.type):
                col = pc.dictionary_encode(col)
            # Decode the column to its raw values (as a Python list)
            col_combined = (
                col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
            )
            decoded = col_combined.to_pylist()
            # Map each value to the new dictionary index (preserving nulls)
            new_indices = [
                union_map[val] if val is not None else None for val in decoded
            ]
            new_indices_array = pa.array(new_indices, type=pa.int32())
            # Create a new dictionary array with the new dictionary
            new_dict_array = pa.DictionaryArray.from_arrays(
                new_indices_array, pa.array(new_dictionary, type=col.type.value_type)
            )
            # Replace the column in the table
            col_index = table.schema.get_field_index(column)
            table = table.set_column(col_index, column, new_dict_array)
            new_tables.append(table)
        tables = new_tables  # update tables for next categorical column

    return tables
