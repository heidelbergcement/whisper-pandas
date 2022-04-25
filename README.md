# WhisperDB Python Pandas Reader

## Install

```
pip install whisper-pandas
```

## Basic usage example

Use as Python package:
```
# Simple to read any Whisper file
>>> from whisper_pandas import WhisperFile
>>> wsp = WhisperFile.read("data/example.wsp")

# Simple to work with metadata as objects
>>> wsp.meta.archives
[WhisperArchiveMeta(index=0, offset=52, seconds_per_point=10, points=1555200, retention=15552000),
 WhisperArchiveMeta(index=1, offset=18662452, seconds_per_point=60, points=5256000, retention=315360000),
 WhisperArchiveMeta(index=2, offset=81734452, seconds_per_point=3600, points=87601, retention=315363600)]

# Simple to work with data as `pandas.Series`
>>> wsp.archives[1].as_series()
2017-02-10 07:07:00+00:00    0.000000
2017-02-10 07:08:00+00:00    0.000000
2017-02-10 07:09:00+00:00    0.000000
2017-02-10 07:10:00+00:00    0.000000
2017-02-10 07:11:00+00:00    0.000000
                               ...   
2021-07-20 13:35:00+00:00    4.099915
2021-07-20 13:36:00+00:00    4.104024
2021-07-20 13:37:00+00:00    4.099772
2021-07-20 13:38:00+00:00    4.101358
2021-07-20 13:39:00+00:00    4.099854
Length: 2331015, dtype: float32
```

Use as command line tool:
```
whisper-pandas data/example.wsp
```

## Tutorial

See [whisper_pandas.ipynb](whisper_pandas.ipynb)


## Description

WhisperDB is a fixed-size time series format (see [docs](https://graphite.readthedocs.io/en/stable/whisper.html)).

The official Python package is here: [whisper](https://github.com/graphite-project/whisper)

You should use it, except if you like [Pandas](https://pandas.pydata.org/) and only need
to read (not write) Whisper files, then you should use `whisper-pandas`.

Why?

* Mucho simpler to use
* Mucho less likely you will shoot yourself in the foot
* Mucho speedy

A given file is always read completely into memory, partial read of metadata or archives is not supported.

## Development

Contributions are welcome via Github pull requests.

You can run the tests via `pytest`.

The packaging follows the recommendation in
[Packaging Python Projects](https://packaging.python.org/tutorials/packaging-projects/).
To make a release increase the version number in `setup.cfg` and run the commands there.
