# Example data files

Used for tests and docs.

## example.wsp

A real sensor data file with 3 archives.

## example_truncated.wsp

A truncated corrupt Whisper file for testing.
The result of truncating `example.wsp` (82.785.664 bytes) to 100.000 bytes.

```
cp example.wsp example_truncated.wsp
truncate -s 100000 example_truncated.wsp
```

## sample1.wsp

A small sample Whisper file for tests, created like this:

```
tbd
```