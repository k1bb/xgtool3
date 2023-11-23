# xgtool3

A Python library to read Gtool3 files as an xarray DataArray.

## Installation

```sh
pip install git+https://github.com/k1bb/xgtool3
```

## Requirements

- xgtool3 depends on numpy, pandas, dask, and xarray.
- The location of the directory containing Gtool3 axis files must be set as the environment variable `$GTAXDIR`.

## How to use

### Open a single file

```python
import xgtool3 as xgt3
path = 'your_data_dir/ATM/Ts'
file = xgt3.Gtool3(path)
da = file.open()
```

### Open multiple files separated by year

```python
import xgtool3 as xgt3
path = 'your_data_dir/y????/OCN/sst'
# You can specify the path using wildcards.
files = xgt3.MultiFileGtool3(path)
da = files.mfopen()
```
