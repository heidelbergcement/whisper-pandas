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
    "WhisperFileMeta",
    "WhisperArchiveMeta",
]

# Whisper file element formats
# See https://graphite.readthedocs.io/en/latest/whisper.html#database-format
FMT_FILE_META = np.dtype(
    [
        ("aggregation_type", ">u4"),
        ("max_retention", ">u4"),
        ("x_files_factor", ">f4"),
        ("archive_count", ">u4"),
    ]
)
FMT_ARCHIVE_META = np.dtype(
    [("offset", ">u4"), ("seconds_per_point", ">u4"), ("points", ">u4")]
)
FMT_POINT = np.dtype([("time", ">u4"), ("val", ">f8")])

AGGREGATION_TYPE_TO_METHOD = {
    1: "average",
    2: "sum",
    3: "last",
    4: "max",
    5: "min",
    6: "avg_zero",
    7: "absmax",
    8: "absmin",
}


@dataclasses.dataclass
class WhisperArchiveMeta:
    """Whisper archive metadata."""

    index: int
    offset: int
    seconds_per_point: int
    points: int

    @classmethod
    def _from_fh(cls, fh, index: int):
        meta = np.fromfile(fh, dtype=FMT_ARCHIVE_META, count=1)[0]
        return cls(
            index=index,
            offset=int(meta["offset"]),
            seconds_per_point=int(meta["seconds_per_point"]),
            points=int(meta["points"]),
        )

    @property
    def retention(self) -> int:
        return self.seconds_per_point * self.points

    @property
    def size(self) -> int:
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

    @staticmethod
    def _meta_from_fh(fh):
        meta = np.fromfile(fh, dtype=FMT_FILE_META, count=1)[0]
        aggregation_method = AGGREGATION_TYPE_TO_METHOD[int(meta["aggregation_type"])]
        return {
            "aggregation_method": aggregation_method,
            "max_retention": int(meta["max_retention"]),
            "x_files_factor": float(meta["x_files_factor"]),
            "archive_count": int(meta["archive_count"]),
        }

    @classmethod
    def _from_fh(cls, fh, path) -> "WhisperFileMeta":
        file_meta = cls._meta_from_fh(fh)
        archives = []
        for idx in range(file_meta["archive_count"]):
            archive_meta = WhisperArchiveMeta._from_fh(fh, idx)
            archives.append(archive_meta)

        return cls(
            path=str(path),
            aggregation_method=file_meta["aggregation_method"],
            max_retention=file_meta["max_retention"],
            x_files_factor=file_meta["x_files_factor"],
            archives=archives,
        )

    @property
    def header_size(self) -> int:
        """Whisper file header size in bytes"""
        return FMT_FILE_META.itemsize + FMT_ARCHIVE_META.itemsize * len(self.archives)

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
        cls,
        path,
        archives: List[int] = None,
        dtype: str = "float32",
        meta_only: bool = False,
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
        meta_only : bool
            Only read metadata from file header
        """
        with Path(path).open("rb") as fh:
            meta = WhisperFileMeta._from_fh(fh, path=path)

        if meta_only:
            return cls(meta=meta, data=[])

        data = []
        for archive_id in range(len(meta.archives)):
            if archives is None or archive_id in archives:
                # TODO: pass fh here, avoid 2x file open
                series = read_whisper_archive(
                    path, info=meta.archives[archive_id], dtype=dtype
                )
            else:
                series = None

            data.append(series)

        return cls(meta=meta, data=data)

    def read_zip(self, path: str):
        raise NotImplementedError()

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


def read_whisper_archive_dataframe(
    path: str, archive_id: int, dtype: str = "float32"
) -> pd.DataFrame:
    info = WhisperFileMeta.read(path).archives[archive_id]
    data = np.fromfile(path, dtype=FMT_POINT, count=info.points, offset=info.offset)
    data = data[data["time"] != 0]
    value = data["val"].astype(dtype)
    time = data["time"].astype("uint32")
    df = pd.DataFrame({"timestamp": time, "value": value}).sort_values("timestamp")
    return df


def main():
    """Command line tool"""
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    wsp = WhisperFile.read(args.path)
    wsp.meta.print_info()


if __name__ == "__main__":
    main()
