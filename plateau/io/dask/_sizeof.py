from dask.sizeof import sizeof as dask_sizeof


def _dct_sizeof(obj):
    return dask_sizeof(obj.__dict__)


def register_sizeof_ktk_classes():
    from plateau.core.common_metadata import SchemaWrapper
    from plateau.core.dataset import DatasetMetadata
    from plateau.core.factory import DatasetFactory
    from plateau.core.index import ExplicitSecondaryIndex, PartitionIndex
    from plateau.core.partition import Partition
    from plateau.io_components.metapartition import MetaPartition

    dask_sizeof.register(DatasetMetadata, _dct_sizeof)
    dask_sizeof.register(DatasetFactory, _dct_sizeof)
    dask_sizeof.register(MetaPartition, _dct_sizeof)
    dask_sizeof.register(ExplicitSecondaryIndex, _dct_sizeof)
    dask_sizeof.register(PartitionIndex, _dct_sizeof)
    dask_sizeof.register(Partition, _dct_sizeof)
    dask_sizeof.register(SchemaWrapper, _dct_sizeof)
