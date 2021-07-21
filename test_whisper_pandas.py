import pytest
import pandas as pd
from numpy.testing import assert_allclose

from whisper_pandas import WhisperFile, WhisperFileMeta


@pytest.fixture(scope="session")
def wsp() -> WhisperFile:
    return WhisperFile.read("example.wsp")


@pytest.fixture(scope="session")
def meta(wsp) -> WhisperFileMeta:
    return wsp.meta


def test_meta(meta):
    """Test if meta data is read OK."""
    assert meta.path == "example.wsp"
    assert meta.aggregation_method == "average"
    assert meta.max_retention == 315363600
    assert_allclose(meta.x_files_factor, 0.5)

    assert meta.archives[0].seconds_per_point == 10
    assert meta.archives[1].seconds_per_point == 60
    assert meta.archives[2].seconds_per_point == 3600


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


def test_print_info(wsp):
    wsp.print_info()
