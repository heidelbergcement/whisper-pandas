#!/usr/bin/env python
"""WhisperDB Python Pandas Reader."""
import dataclasses
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import whisper

__all__ = [
    "WhisperFile",
    "WhisperFileMeta",
    "WhisperArchiveMeta",
]

# Whisper file element formats
# See https://graphite.readthedocs.io/en/latest/whisper.html#database-format

FMT_POINT = np.dtype([("time", ">u4"), ("val", ">f8")])
# TODO: change to format dtype and use it for parsing header
N_ARCHIVE = 12
N_META = 16


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
        return FMT_POINT.itemsize * self.points

    def print_info(self):
        print("archive:", self.index)
        print("offset:", self.offset)
        print("seconds_per_point:", self.seconds_per_point)
        print("points:", self.points)
        print("retention:", self.retention)
        print("size:", self.size)


@dataclasses.dataclass
class WhisperFileMeta:
    """Whisper file metadata."""

    path: str
    aggregation_method: str
    max_retention: int
    x_files_factor: float
    archives: List[WhisperArchiveMeta]

    @classmethod
    def read(cls, path) -> "WhisperFileMeta":
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

        return cls(
            path=str(path),
            aggregation_method=info["aggregationMethod"],
            max_retention=info["maxRetention"],
            x_files_factor=info["xFilesFactor"],
            archives=archives,
        )

    @property
    def header_size(self) -> int:
        """Whisper file header size in bytes"""
        return N_META + N_ARCHIVE * len(self.archives)

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
class WhisperFile:
    """Whisper file (the whole enchilada, meta + data)."""

    meta: WhisperFileMeta
    data: List[pd.Series]

    @classmethod
    def read(
        cls, path, archives: List[int] = None, dtype: str = "float32"
    ) -> "WhisperFile":
        """Read Whisper archive into a pandas.Series.

        Parameters
        ----------
        path : str
            Filename
        archives : list
            List of archive IDs to read.
            Highest time resolution is archive 0.
            Default: all
        dtype : {"float32", "float64"}
            Value float data type
        """
        meta = WhisperFileMeta.read(path)

        if archives is None:
            archives = list(range(len(meta.archives)))

        data = []
        for archive_id in archives:
            if archive_id in archives:
                series = read_whisper_archive(
                    path, info=meta.archives[archive_id], dtype=dtype
                )
            else:
                series = None

            data.append(series)

        return cls(meta=meta, data=data)

    def print_info(self):
        self.meta.print_info()


def read_whisper_archive(
    path: str, info: WhisperArchiveMeta, dtype: str = "float32"
) -> pd.Series:
    data = np.fromfile(path, dtype=FMT_POINT, count=info.points, offset=info.offset)

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


def main():
    """Command line tool"""
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    meta = WhisperFileMeta.read(args.path)
    meta.print_info()


if __name__ == "__main__":
    main()
