#!/usr/bin/env python
"""WhisperDB Python Pandas Reader."""
from __future__ import annotations
import dataclasses
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

__all__ = [
    "WhisperFile",
    "WhisperFileMeta",
    "WhisperArchive",
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
    def from_buffer(cls, buffer, index: int) -> WhisperArchiveMeta:
        offset = FMT_FILE_META.itemsize + index * FMT_ARCHIVE_META.itemsize
        meta = np.frombuffer(buffer, dtype=FMT_ARCHIVE_META, count=1, offset=offset)[0]
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
    def _meta_from_buffer(buffer: bytes) -> dict:
        meta = np.frombuffer(buffer, dtype=FMT_FILE_META, count=1)[0]
        aggregation_method = AGGREGATION_TYPE_TO_METHOD[int(meta["aggregation_type"])]
        return {
            "aggregation_method": aggregation_method,
            "max_retention": int(meta["max_retention"]),
            "x_files_factor": float(meta["x_files_factor"]),
            "archive_count": int(meta["archive_count"]),
        }

    @classmethod
    def from_buffer(cls, buffer: bytes, path: str | Path) -> WhisperFileMeta:
        file_meta = cls._meta_from_buffer(buffer[0 : FMT_FILE_META.itemsize])
        archives = []
        for idx in range(file_meta["archive_count"]):
            archive_meta = WhisperArchiveMeta.from_buffer(buffer, idx)
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
        print("actual size:", self.file_size_actual)
        print("expected size:", self.file_size)
        print()
        print("aggregation_method:", self.aggregation_method)
        print("max_retention:", self.max_retention)
        print("x_files_factor:", self.x_files_factor)

        for archive in self.archives:
            print()
            archive.print_info()


@dataclasses.dataclass
class WhisperArchive:
    """Whisper file single archive."""

    meta: WhisperArchiveMeta
    bytes: bytes = dataclasses.field(repr=False)

    def as_numpy(self) -> np.ndarray:
        return np.frombuffer(
            self.bytes, dtype=FMT_POINT, count=self.meta.points, offset=self.meta.offset
        )

    def as_dataframe(self, dtype: str = "float32") -> pd.DataFrame:
        data = self.as_numpy()
        data = data[data["time"] != 0]
        value = data["val"].astype(dtype)
        time = data["time"].astype("uint32")
        df = pd.DataFrame({"timestamp": time, "value": value})
        return df.sort_values("timestamp")

    # TODO: merge with as_dataframe since they almost do the same?
    def as_series(self, dtype: str = "float32") -> pd.Series:
        data = self.as_numpy()

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

        s = pd.Series(val, index)
        return s.sort_index()


@dataclasses.dataclass
class WhisperFile:
    """Whisper file (metadata and all archives)."""

    meta: WhisperFileMeta
    bytes: bytes = dataclasses.field(repr=False)

    @classmethod
    def read(cls, path: str | Path, compression: str = "infer") -> WhisperFile:
        """Read Whisper file.

        Parameters
        ----------
        path : str | Path
            Filename
        compression : {"infer", "none", "gzip"}
            For on-the-fly decompression
        """
        path = Path(path)

        if compression == "infer":
            if path.suffix == ".gz":
                compression = "gzip"
            else:
                compression = "none"

        if compression == "none":
            buffer = path.read_bytes()
        elif compression == "gzip":
            import gzip

            buffer = path.read_bytes()
            buffer = gzip.decompress(buffer)
        else:
            raise ValueError(f"Invalid compression: {compression!r}")

        meta = WhisperFileMeta.from_buffer(buffer, path=path)

        return cls(meta=meta, bytes=buffer)

    @property
    def archives(self) -> List[WhisperArchive]:
        """Whisper file archive list."""
        return [
            WhisperArchive(meta=meta, bytes=self.bytes) for meta in self.meta.archives
        ]

    def print_info(self):
        self.meta.print_info()


def main():
    """Command line tool"""
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    wsp = WhisperFile.read(args.path)
    wsp.meta.print_info()


if __name__ == "__main__":
    main()
