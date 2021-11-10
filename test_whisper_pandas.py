import pytest
import pandas as pd
from numpy.testing import assert_allclose

from whisper_pandas import WhisperFile, WhisperFileMeta, WhisperArchiveMeta, print_meta


@pytest.fixture(scope="session")
def wsp_path() -> str:
    return "example.wsp"


@pytest.fixture(scope="session")
def wsp(wsp_path) -> WhisperFile:
    return WhisperFile.read(wsp_path)


@pytest.fixture(scope="session")
def meta(wsp) -> WhisperFileMeta:
    return wsp.meta


def test_meta(meta):
    """Test if meta data is read OK."""
    assert meta.path == "example.wsp"
    assert meta.aggregation_method == "average"
    assert meta.max_retention == 315363600
    assert_allclose(meta.x_files_factor, 0.5)

    assert meta.header_size == 52
    assert meta.file_size == 82785664
    assert meta.file_size_actual == 82785664
    assert meta.file_size_mismatch is False

    assert len(meta.archives) == 3
    assert meta.archives[0] == WhisperArchiveMeta(
        index=0, offset=52, seconds_per_point=10, points=1555200
    )
    assert meta.archives[1] == WhisperArchiveMeta(
        index=1,
        offset=18662452,
        seconds_per_point=60,
        points=5256000,
    )
    assert meta.archives[2] == WhisperArchiveMeta(
        index=2,
        offset=81734452,
        seconds_per_point=3600,
        points=87601,
    )


def test_data_archive_1(wsp):
    """Test if data is read and converted to pandas OK."""
    s = wsp.data[1]

    assert len(s) == 2331015
    assert s.dtype == "float32"

    assert s.index.is_monotonic
    assert s.index.is_unique
    assert str(s.index.tz) == "UTC"

    assert s.index[0] == pd.Timestamp("2017-02-10 07:07:00+0000")
    assert s.index[-1] == pd.Timestamp("2021-07-20 13:39:00+0000")
    assert_allclose(s.iloc[-1], 4.099854, atol=1e-5)


def test_data_archive_0(wsp):
    s = wsp.data[0]
    assert len(s) == 1555200

    assert s.index[0] == pd.Timestamp("2020-07-29 08:28:10+0000")
    assert s.index[-1] == pd.Timestamp("2021-07-20 13:39:30+0000")
    assert_allclose(s.iloc[-1], 4.081736, atol=1e-5)


def test_data_archive_2(wsp):
    s = wsp.data[2]
    assert len(s) == 38855

    assert s.index[0] == pd.Timestamp("2017-02-10 07:00:00+0000")
    assert s.index[-1] == pd.Timestamp("2021-07-20 13:00:00+0000")
    assert_allclose(s.iloc[-1], 4.099754, atol=1e-5)


def test_read_only_some_archives():
    wsp = WhisperFile.read("example.wsp", archives=[2, 1])
    assert len(wsp.data) == 3
    assert wsp.data[0] is None
    assert len(wsp.data[1]) == 2331015
    assert len(wsp.data[2]) == 38855


def test_print_meta(wsp_path):
    print_meta(wsp_path)
