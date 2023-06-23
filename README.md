# xgtool3
A Python library to read Gtool3 files as xarray.

## Installation
```sh
pip install git+https://github.com/k1bb/xgtool3
```

## How to use
### Open a single file
```python
import xgtool3
path = 'your_data_dir/ATM/Ts'
xgt3 = xgtool3.Gtool3(path)
da = gt3.open()
```

### Open multiple files separated by year
```python
import xgtool3
path = 'your_data_dir/y????/ATM/Ts'
xgt3 = xgtool3.MultiFileGtool3(path)
da = gt3.mfopen(max_workers=8)
```
