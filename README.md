# WhisperDB Python Pandas Reader

## Getting started

Install:
```
pip install whisper-pandas
```

Use as Python package:
```
>>> from whisper_pandas import WhisperFile
>>> wsp = WhisperFile.read("example.wsp")
>>> wsp.print_info()
TBD
>>> wsp.archive[1].data
TBD
```

Use as command line tool:
```
whisper-pandas example.wsp
```


## Description

WhisperDB is a fixed-size time series format (see [docs](https://graphite.readthedocs.io/en/stable/whisper.html)).

The official Python package is here: [whisper](https://github.com/graphite-project/whisper)

You should use it, except if you like [Pandas](https://pandas.pydata.org/) and only need
to read (not write) Whisper files, then you should use `whisper-pandas`.

Why?

* Mucho simpler to use
* Mucho less likely you will shoot yourself in the foot
* Mucho speedy

It's a from-scratch Whisper read implementation using Numpy & Pandas,
using objects for the metadata and some conveniences like quickly
showing what data is available in a given file or reading zipped files.

Partial reading is not implemented, a given file is always read completely
and the data directly converted to columnar `pandas.Series` format with
a datetime index.