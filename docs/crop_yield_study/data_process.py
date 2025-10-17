"""
Tools for processing of the crop yield data
"""
import logging
import warnings

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import cartopy
import cartopy.crs as ccrs
from torch.utils.data import TensorDataset

DEVICE = 'cuda:0'
CROP = 'maize'
YEAR = 2010
LOG = False


class CropYieldData:
    """
    Convenience class to facilitate access to crop yield data for a specific crop type and for a specific year.
    """
    def __init__(self, crop: str, year: int):
        self.crop, self.year = crop, year
        self.ds = Dataset(f'./cropYieldData/{crop}/yield_{year}.nc4')

    @property
    def data(self):
        data = self.ds['var'][:].data
        data[self.ds['var'][:].mask] = np.nan
        return data

    @property
    def mask(self):
        mask = self.ds['var'][:].mask
        mask[self.data == np.nan] = True
        mask[self.data == 0] = True
        return mask

    @property
    def lon(self):
        return self.ds['lon'][:].data

    @property
    def lat(self):
        return self.ds['lat'][:].data

    def flat_to_grid(self, flat_data: np.ndarray):
        out = np.zeros(self.data.size)
        out[~self.mask.flatten()] = flat_data
        out[self.mask.flatten()] = np.nan
        return out.reshape(self.data.shape)

    def plot(self, data=None, ax=None, **kwargs):
        if data is None:
            data = self.data
        if ax is None:
            fig = plt.figure(figsize=(12, 9))
            ax = plt.axes(projection=ccrs.Robinson())
        img_extent = (self.lon[0], self.lon[-1], self.lat[0], self.lat[-1])
        im = ax.imshow(data, origin='lower', extent=img_extent, transform=ccrs.PlateCarree(), **kwargs)
        ax.coastlines()
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.OCEAN)
        ax.set_title('Global yield')
        # cax = fig.add_axes([ax.get_position().x1+1,ax.get_position().y0,0.02,ax.get_position().height])
        cax = ax.inset_axes((0.975, 0, .025, 1))
        plt.colorbar(im, cax=cax)  # Similar to fig.colorbar(im, cax = cax)
        if ax is None:
            fig.tight_layout()
        return fig





def build_features_targets(years: list[int],
                           crop: str,
                           remove_nan: bool = True,
                           return_lat_lon: bool = False,
                           add_year: bool = True,
                           log: bool = LOG):
    """
    Builds a set of features and targets as torch tensors from the dataset.

    Parameters
    ----------
    years : list[int]
    List of years to include in the dataset.

    remove_nan : bool
    Whether to remove NaN values.

    return_lat_lon: bool
    Whether to return lat/lon coordinates.

    add_year: bool
    Whether to add year in the features

    Returns
    -------
    features : numpy ndarray
        shape (n, 2) or (n, 3) depending on add_year value

    targets : numpy ndarray
        shape (n, 1)
    """
    features, targets = [], []

    for year in years:
        data_year = Dataset(f'./cropYieldData/{crop}/yield_{year}.nc4')
        yields = data_year['var'][:].data
        missing = data_year['var'][:].mask
        # consider zero yields as outliers
        missing[yields == 0] = True
        lat = data_year['lat'][:].data
        lon = data_year['lon'][:].data
        lat, lon = np.meshgrid(lat, lon, indexing='ij')
        if remove_nan:
            if add_year:
                year_array = year * np.ones(np.sum(~missing))
                data_flat = (np.stack((lat[~missing].flatten(), lon[~missing].flatten(), year_array), axis=-1),
                             yields[~missing].flatten())
            else:
                data_flat = (np.stack((lat[~missing].flatten(), lon[~missing].flatten()), axis=-1),
                             yields[~missing].flatten())
        else:
            yields[missing] = np.nan
            year_array = year * np.ones(yields.shape[0] * yields.shape[1])
            if add_year:
                data_flat = (np.stack((lat.flatten(), lon.flatten(), year_array), axis=-1), yields.flatten())
            else:
                data_flat = (np.stack((lat.flatten(), lon.flatten()), axis=-1), yields.flatten())
        features.append(data_flat[0])
        if np.any(data_flat[1] == 0):
            warnings.warn(f"Zero yield in the data for year {year}")
        logging.info(np.nanmin(data_flat[1]))
        if log:
            targets.append(np.log(data_flat[1]))
        else:
            targets.append(data_flat[1])

    features = np.concatenate(features)
    targets = np.concatenate(targets).reshape((-1, 1))
    if return_lat_lon:
        return features, targets, lat, lon
    return features, targets


# load and process the data
def dataset_from_arrays(features: np.ndarray, targets: np.ndarray, transform=None):
    features = torch.from_numpy(features).float()
    targets = torch.from_numpy(targets).float()
    if transform is None:
        t_features = torch.mean(features, dim=0), torch.std(features, dim=0)
        t_targets = torch.mean(targets, dim=0), torch.std(targets, dim=0)
        transform = (t_features, t_targets)
    t_features, t_targets = transform
    features = (features - t_features[0]) / t_features[1]
    targets = (targets - t_targets[0]) / t_targets[1]
    return TensorDataset(features, targets), transform


def dataset_from_arrays2(features: np.ndarray, targets: np.ndarray, transform=None):
    features = torch.from_numpy(features).float()
    targets = torch.from_numpy(targets).float()
    targets = np.log(targets + 0.1)
    if transform is None:
        t_features = torch.mean(features, dim=0), torch.std(features, dim=0)
        t_targets = torch.mean(targets, dim=0), torch.std(targets, dim=0)
        transform = (t_features, t_targets)
    t_features, t_targets = transform
    features = (features - t_features[0]) / t_features[1]
    targets = (targets - t_targets[0]) / t_targets[1]
    return TensorDataset(features, targets), transform