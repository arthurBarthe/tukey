"""
In this file we allow to combine multiple run outputs into a single dataset
"""
import torch
import xarray as xr
from torch.utils.data import Dataset, ConcatDataset
from typing import List


class DatasetFromRun(Dataset):
    """
    This class allows to define a dataset for a single run.
    """
    def __init__(self, xr_dataset_run: xr.Dataset, input_varnames: list, output_varnames: list):
        self.raw_data = xr_dataset_run
        self.input_varnames = input_varnames
        self.output_varnames = output_varnames

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        # TODO for now we only use one level
        data_idx = self.raw_data.isel(time=idx, lev=0)
        input_idx = tuple([torch.from_numpy(data_idx[varname].to_numpy()) for varname in self.input_varnames])
        output_idx = tuple([torch.from_numpy(data_idx[varname].to_numpy()) for varname in self.output_varnames])
        input_idx = torch.stack(input_idx, dim=0)
        output_idx = torch.stack(output_idx, dim=0)
        return input_idx, output_idx


class DatasetFromMultipleRuns(ConcatDataset):
    """
    This class allows to define a dataset based on multiple runs.
    """
    def __init__(self, xr_dataset_runs: xr.Dataset, run_indices: List[int],
                 input_varnames: list, output_varnames: list):
        datasets = [DatasetFromRun(xr_dataset_runs.isel(run=i), input_varnames, output_varnames) for i in run_indices]
        super().__init__(datasets)