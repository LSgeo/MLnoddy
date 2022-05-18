from pathlib import Path

import numpy as np
import verde as vd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def parse_geology(pth, layer):
    """Return geology voxel model int labels from .g12.gz"""
    mod = np.loadtxt(pth, dtype=int).reshape(200, 200, 200)
    return np.transpose(mod, (0, 2, 1))[layer, :, :]


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
        load_geology=False,
        **kwargs,
    ):
        super().__init__()

        self.m_dir = Path(model_dir)
        self.m_names = sorted(
            set([p.name[:-7] for p in sorted(self.m_dir.iterdir())])
        )  # List of unique model names in model_dir - (named after a timestamp)
        self.load_geology = load_geology
        self.kwargs = kwargs
        self.len = len(self.m_names)
        self.label = self.m_dir.stem.split("_")  # Event 1,2 are STRAT, TILT

    def _augment(self, *tensors):
        if self.augs.get("hflip") and torch.rand(1) < 0.5:
            (TF.hflip(t) for t in tensors)
        if self.augs.get("vflip") and torch.rand(1) < 0.5:
            (TF.vflip(t) for t in tensors)
        if self.augs.get("rotate") and torch.rand(1) < 0.5:
            (TF.rotate(t, 90) for t in tensors)
        if self.augs.get("noise"):
            # add x % random noise to datasets.
            pass
            # (add_noise(t, noise) for t in tensors)

    def _process(self, index):
        """Convert parsed numpy arrays to tensors and augment"""
        f = self.m_dir / self.m_names[index]
        self.gt_mag = torch.from_numpy(parse_geophysics((f).with_suffix(".mag.gz")))
        self.gt_grv = torch.from_numpy(parse_geophysics((f).with_suffix(".grv.gz")))
        if self.load_geology:
            # This is mildly expensive - Could pass layer to np.loadtxt skips?
            self.data = {
                "geo": parse_geology((f).with_suffix(".g12.gz"), layer=0),
                **self.data,
            }

        if "augment" in self.kwargs:
            self.augs = {
                "hflip": True,
                "vflip": True,
                "rotate": False,
                "noise": False,
                **self.kwargs["augs"],
            }

            raise NotImplementedError("Need to troubleshoot Tensor shapes")
            self._augment([t for t in self.data[""]])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        self.data = {"label": self.label}
        self._process(index)

        return self.data


noddy_model_dir = Path(r"data/DYKE_FOLD_FAULT")
dset = NoddyDataset(noddy_model_dir, load_geology=True)


import matplotlib.pyplot as plt

for i in range(5):
    sample = dset[i]
    fig, (inp, out, geo) = plt.subplots(1, 3)
    plt.suptitle(sample["label"])
    inp.imshow(sample["gt"][0])
    out.imshow(sample["gt"][1])
    geo.imshow(sample["geo"])
    plt.show()

