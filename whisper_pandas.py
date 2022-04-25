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
DTYPE_FILE_META = np.dtype(
    [
        ("aggregation_type", ">u4"),
        ("max_retention", ">u4"),
        ("x_files_factor", ">f4"),
        ("archive_count", ">u4"),
    ]
)
DTYPE_ARCHIVE_META = np.dtype(
    [("offset", ">u4"), ("seconds_per_point", ">u4"), ("points", ">u4")]
)
DTYPE_POINT = np.dtype([("timestamp", ">u4"), ("value", ">f8")])

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
        offset = DTYPE_FILE_META.itemsize + index * DTYPE_ARCHIVE_META.itemsize
        meta = np.frombuffer(buffer, dtype=DTYPE_ARCHIVE_META, count=1, offset=offset)[
            0
        ]
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
        return DTYPE_POINT.itemsize * self.points

    def describe(self) -> pd.DataFrame:
        return pd.Series(
            {
                "archive": self.index,
                "seconds_per_point": self.seconds_per_point,
                "points": self.points,
                "retention": self.retention,
                "offset": self.offset,
                "size": self.size,
            }
        ).to_frame(name="value")


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
        meta = np.frombuffer(buffer, dtype=DTYPE_FILE_META, count=1)[0]
        aggregation_method = AGGREGATION_TYPE_TO_METHOD[int(meta["aggregation_type"])]
        return {
            "aggregation_method": aggregation_method,
            "max_retention": int(meta["max_retention"]),
            "x_files_factor": float(meta["x_files_factor"]),
            "archive_count": int(meta["archive_count"]),
        }

    @classmethod
    def from_buffer(cls, buffer: bytes, path: str | Path) -> WhisperFileMeta:
        file_meta = cls._meta_from_buffer(buffer[0 : DTYPE_FILE_META.itemsize])
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
        return DTYPE_FILE_META.itemsize + DTYPE_ARCHIVE_META.itemsize * len(
            self.archives
        )

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

    def describe_meta(self) -> pd.DataFrame:
        return pd.Series(
            {
                "path": self.path,
                "actual size": self.file_size_actual,
                "expected size": self.file_size,
                "aggregation_method": self.aggregation_method,
                "max_retention": self.max_retention,
                "x_files_factor": self.x_files_factor,
            }
        ).to_frame(name="value")

    def describe_archives(self) -> pd.DataFrame:
        """Archive summary information table."""
        infos = [_.describe()["value"] for _ in self.archives]
        df = pd.DataFrame(infos).set_index("archive")
        return df

    def print_info(self):
        print(self.describe_meta())
        print()
        print(self.describe_archives())


@dataclasses.dataclass
class WhisperArchive:
    """Whisper file single archive."""

    meta: WhisperArchiveMeta
    bytes: bytes = dataclasses.field(repr=False)

    def to_numpy(self) -> np.ndarray:
        return np.frombuffer(
            self.bytes,
            dtype=DTYPE_POINT,
            count=self.meta.points,
            offset=self.meta.offset,
        )

    def to_frame(
        self,
        dtype: str = "float64",
        to_datetime: bool = True,
        drop_time_zero: bool = True,
        time_sort: bool = True,
    ) -> pd.DataFrame:
        """Convert archive data to pandas.DataFrame.

        Parameters
        ----------
        dtype : str
            Data type for point values
        to_datetime : bool
            Convert from Unix int timestamps to pandas timestamps?
        drop_time_zero : bool
            Drop points where time is 0, i.e. that were never filled?
        time_sort : bool
            Sort points in chronological order?

        Returns
        -------
        df : pandas.DataFrame
            Dataframe with columns "timestamp" and "value" and the
            point position index in the Whisper archive as index.
        """
        data = self.to_numpy()

        if drop_time_zero:
            data = data[data["timestamp"] != 0]

        # The int32 typecast is a workaround for a performance bug
        # on pandas versions < 1.3 when using uint32
        # https://github.com/pandas-dev/pandas/issues/42606
        # int32 max value can represent times up to year 2038
        timestamp = data["timestamp"].astype("int32")
        if to_datetime:
            timestamp = pd.to_datetime(timestamp, unit="s", utc=True)

        # The type cast for the values is needed to avoid this error later on
        # ValueError: Big-endian buffer not supported on little-endian compiler
        value = data["value"].astype(dtype)

        df = pd.DataFrame({"timestamp": timestamp, "value": value})

        if time_sort:
            df = df.sort_values("timestamp")

        return df


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
