import xarray as xr
from fsspec.implementations.sftp import SFTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.mapping import FSMap
from enum import Enum

class AccesMode(Enum):
    local = 1
    remote = 2

def load_credentials():
    credentials = dict()
    with open('credentials.txt') as file_credentials:
        credentials_keys= ['host', 'username', 'password']
        for key in credentials_keys:
            value = file_credentials.readline().rstrip()
            credentials[key] = value
    return credentials

def load_ds(path: str, mode: AccesMode):
    if mode == AccesMode.local:
        fs = LocalFileSystem()
    elif mode == AccesMode.remote:
        credentials = load_credentials()
        print(credentials)
        fs = SFTPFileSystem(**credentials)
    map_to_ds_store = FSMap(root=path, fs=fs)
    ds = xr.open_zarr(map_to_ds_store)
    return ds