from typing import Any

PartitionDictType = dict[str, dict[str, str]]


# TODO: purge this. This is just slowing us down by creating many python objects we don't actual
# Changing the partition class needs to be done with are since it's to_dict is used for the storage metadata spec
class Partition:
    def __init__(
        self,
        label: str,
        files: dict[str, str] | None = None,
        metadata: dict | None = None,
    ):
        """An object for the internal representation of the metadata of a
        partition.

        This class is for internal use only

        Parameters
        ----------
        label:
            A label identifying the partition, e.g. `partition_1` or `P=0/L=A`
        files:
            A dictionary containing the keys of the files contained in this partition
        metadata:
            Partition level, custom metadata
        """
        self.label = label
        self.files = files if files else {}

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Partition):
            return False
        if self.label != other.label:
            return False
        if self.files != other.files:
            return False
        return True

    @staticmethod
    def from_dict(label: str, dct: str | PartitionDictType):
        if isinstance(dct, str):
            raise ValueError(
                "Trying to load a partition from a string. Probably the dataset file uses the multifile "
                "feature. Please load the metadata object using the DatasetMetadata.load_from_buffer "
                "method instead to resolve references to external partitions."
            )
        return Partition(label, files=dct.get("files", {}))

    def to_dict(self, version: Any = None) -> PartitionDictType:
        return {"files": self.files}
