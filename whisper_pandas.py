#!/usr/bin/env python
"""WhisperDB Python Pandas Reader."""
import dataclasses
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

__all__ = [
    "WhisperFile",
    "WhisperMeta",
    "WhisperArchiveData",
    "WhisperArchiveMeta",
    "read_whisper_archive",
]


class WhisperBytes:
    """Size of Whisper file elements in bytes"""
    # See https://graphite.readthedocs.io/en/latest/whisper.html#database-format
    point = 12
    archive_info = 12
    meta = 16


@dataclasses.dataclass
class WhisperArchiveMeta:
    """Whisper archive metadata."""
    index: int
    offset: int
    seconds_per_point: int
    points: int
    retention: int

    @property
    def size(self):
        return WhisperBytes.point * self.points

    def print_info(self):
        print("archive:", self.index)
        print("offset:", self.offset)
        print("seconds_per_point:", self.seconds_per_point)
        print("points:", self.points)
        print("retention:", self.retention)
        print("size:", self.size)


@dataclasses.dataclass
class WhisperMeta:
    """Whisper file metadata."""
    path: str
    aggregation_method: str
    max_retention: int
    x_files_factor: float
    archives: List[WhisperArchiveMeta]

    @classmethod
    def read(cls, path) -> "WhisperMeta":
        # TODO: replacado by selber parsen
        import whisper
        info = whisper.info(path)
        archives = []
        for index, _ in enumerate(info["archives"]):
            archive = WhisperArchiveMeta(
                index=index,
                offset=_["offset"],
                seconds_per_point=_["secondsPerPoint"],
                points=_["points"],
                retention=_["retention"],
            )
            archives.append(archive)

        return WhisperMeta(
            path=str(path),
            aggregation_method=info["aggregationMethod"],
            max_retention=info["maxRetention"],
            x_files_factor=info["xFilesFactor"],
            archives=archives
        )

    @property
    def header_size(self) -> int:
        """Whisper file header size in bytes"""
        return WhisperBytes.meta + WhisperBytes.archive_info * len(self.archives)

    @property
    def file_size(self) -> int:
        """Whisper file total size in bytes"""
        return self.header_size + sum(archive.size for archive in self.archives)

    @property
    def file_size_actual(self) -> int:
        """Actual file size in bytes"""
        return Path(self.path).stat().st_size

    @property
    def file_size_mismatch(self) -> bool:
        """Does actual and expected file size according to header match?"""
        return self.file_size != self.file_size_actual

    def print_info(self):
        print("path:", self.path)
        print("aggregation_method:", self.aggregation_method)
        print("max_retention:", self.max_retention)
        print("x_files_factor:", self.x_files_factor)

        for archive in self.archives:
            print()
            archive.print_info()

        if self.file_size_mismatch:
            print("\n*** FILE IS CORRUPT! ***")
            print("actual size:", self.file_size_actual)
            print("expected size:", self.file_size)


@dataclasses.dataclass
class WhisperArchiveData:
    """Whisper archive data."""
    raw: np.ndarray


@dataclasses.dataclass
class WhisperFile:
    """Whisper file (the whole enchilada, meta + data)."""
    meta: WhisperMeta
    data: List[WhisperArchiveData]

# TODO: add test, then refactor this to WhisperFile.read
# TODO: then add ZIP support

def read_whisper_archive(path: str, archive_id: int, dtype: str = "float32") -> pd.Series:
    """Read Whisper archive into a pandas.Series.

    Parameters
    ----------
    path : str
        Filename
    archive_id : int
        Archive ID (highest time resolution is 0)
    dtype : {"float32", "float64"}
        Value float data type
    """
    meta = WhisperMeta.read(path)
    if archive_id < 0 or archive_id >= len(meta.archives):
        raise ValueError(f"Invalid archive_id = {archive_id}")

    info = meta.archives[archive_id]
    data = np.fromfile(
        path, dtype=np.dtype([("time", ">u4"), ("val", ">f8")]), count=info.points, offset=info.offset
    )

    # That's right, señor. We remove all points with `time==0`.
    # The spec doesn't say, but apparamento this is what it takes.
    data = data[data["time"] != 0]

    # The type cast for the values is needed to avoid this error later on
    # ValueError: Big-endian buffer not supported on little-endian compiler
    val = data["val"].astype(dtype)

    # Workaround for a performance bug on pandas versions < 1.3
    # https://github.com/pandas-dev/pandas/issues/42606
    # int32 max value can represent times up to year 2038
    index = data["time"].astype("int32")
    index = pd.to_datetime(index, unit="s", utc=True)

    return pd.Series(val, index).sort_index()


def cli():
    """Command line tool"""
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    meta = WhisperMeta.read(args.path)
    meta.print_info()

    print(read_whisper_archive(args.path, archive_id=0))
    print(read_whisper_archive(args.path, archive_id=1))
    print(read_whisper_archive(args.path, archive_id=2))


if __name__ == '__main__':
    cli()
