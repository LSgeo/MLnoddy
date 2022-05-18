from pathlib import Path

import numpy as np
import verde as vd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def parse_geophysics(pth):
    """Return forward model values from Noddy geophysics .mag.gz and .grv.gz"""
    return np.loadtxt(pth, skiprows=8)

class NoddyDataset(Dataset):
    """Create a Dataset to access magnetic, gravity, and surface geology
    from the Noddyverse (https://doi.org/10.5194/essd-14-381-2022)

    The data are returned concatenated, with the geology model and
    unique identifier accessible.

    Parameters:
        model_dir: Path to numpy event history folder
        survey/augment: Do survey / augment. See method.
        load_geology: Optionally load the g12 voxel model, surface layer
        kwargs: Parameter dictionary for survey / augmentation
    """

    def __init__(
        self,
        model_dir,
    ):
        super().__init__()

        self.m_dir = Path(model_dir)
        self.m_names = sorted(
            set([p.name[:-7] for p in sorted(self.m_dir.iterdir())])
        )  # List of unique model names in model_dir - (named after a timestamp)
        self.len = len(self.m_names)
        self.label = self.m_dir.stem.split("_")  # Event 1,2 are STRAT, TILT


    def _process(self, index):
        """Convert parsed numpy arrays to tensors and augment"""
        f = self.m_dir / self.m_names[index]
        self.gt_mag = torch.from_numpy(parse_geophysics((f).with_suffix(".mag.gz")))
        self.gt_grv = torch.from_numpy(parse_geophysics((f).with_suffix(".grv.gz")))
        self.data = {
            "gt": torch.stack((self.gt_mag, self.gt_grv), dim=0),
            **self.data,
        }

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        self.data = {"label": self.label}
        self._process(index)

        return self.data


noddy_model_dir = Path(r"data/DYKE_FOLD_FAULT")
dset = NoddyDataset(noddy_model_dir)

import matplotlib.pyplot as plt

for i in range(20):
    fig, (inp, out) = plt.subplots(1, 2)

    inp.imshow(dset[i]["gt"][0])
    out.imshow(dset[i]["gt"][1])
