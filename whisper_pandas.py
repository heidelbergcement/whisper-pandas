#!/usr/bin/env python
"""WhisperDB Python Pandas Reader.

https://github.com/graphite-project/whisper
https://graphite.readthedocs.io/en/stable/whisper.html

This was started because the existing dump and fetch
implementation has some issues:
https://github.com/graphite-project/whisper/issues/305
"""
import dataclasses
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


@dataclasses.dataclass()
class WhisperArchiveMeta:
    pass


@dataclasses.dataclass()
class WhisperMeta:
    pass


def whisper_info(path: str) -> dict:
    """Re-implementation of whisper.info to be standalone."""
    import whisper
    return whisper.info(path)


def whisper_file_size(info: dict) -> int:
    """Compute expected file size in bytes for a given whisper.info"""
    size_meta = 16
    size_archive_info = 12
    size_point = 12
    size_header = size_meta + len(info["archives"]) * size_archive_info
    size_total = size_header + sum(_["points"] * size_point for _ in info["archives"])
    return size_total


def whisper_info_print(path: str) -> None:
    info = whisper_info(path)
    size_actual = Path(path).stat().st_size
    size_expected = whisper_file_size(info)

    print(f"\npath: {path}\n")

    if size_actual != size_expected:
        print("FILE IS CORRUPT!")
        print(f" actual size: {size_actual}")
        print(f" expected size: {size_expected}")

    print()
    print(f"aggregationMethod: {info['aggregationMethod']}")
    print(f"maxRetention: {info['maxRetention']}")
    print(f"xFilesFactor: {info['xFilesFactor']}")
    print(f"fileSize: {size_actual}")
    print()


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

    infos = whisper_info(path)
    if archive_id < 0 or archive_id >= len(infos["archives"]):
        raise ValueError(f"Invalid archive_id = {archive_id}")

    info = infos["archives"][archive_id]
    data = np.fromfile(
        path, dtype=np.dtype([("time", ">u4"), ("val", ">f8")]), count=info["points"], offset=info["offset"]
    )
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
    whisper_info_print(args.path)


if __name__ == '__main__':
    cli()
