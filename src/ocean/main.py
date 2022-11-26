from fsspec.implementations.sftp import SFTPFileSystem
from fsspec.mapping import FSMap
import xarray as xr
fs = SFTPFileSystem('login.hpc.qmul.ac.uk', username='ahw795', password="4='!Mp3M%%@P")
map = FSMap(root='/data/scratch/ahw795/oceans/publication/eddy/high_res.zarr', fs=fs)

ds = xr.open_zarr(map)
print(ds)