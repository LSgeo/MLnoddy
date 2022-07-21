import time
import logging
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

labels = {
    "STRAT": 0,
    "FOLD": 1,
    "FAULT": 2,
    "UNCONFORMITY": 3,
    "DYKE": 4,
    "PLUG": 5,
    "SHEAR-ZONE": 6,
    "TILT": 7,
}
inverse_labels = {v: k for k, v in labels.items()}


def parse_geology(pth, layer):
    """Return geology voxel model int labels from .g12.gz"""
    model = np.loadtxt(pth, dtype=int).reshape(200, 200, 200)
    return np.ascontiguousarray(np.transpose(model, (0, 2, 1))[layer, :, :])


def parse_geophysics(pth: Path, mag=False, grv=False):
    """Return forward model values from Noddy geophysics .mag.gz and .grv.gz"""

    files = (pth.with_suffix(".mag.gz"), pth.with_suffix(".grv.gz"))
    if not grv:
        files = [files[0]]
    if not mag:
        files = [files[1]]
    for pth in files:
        yield np.ascontiguousarray(np.loadtxt(pth, skiprows=8, dtype=np.float32))
    # yield pd.read_csv(pth,sep="\t",skiprows=8,header=None,usecols=range(200),dtype=np.float32,na_filter=False,).values.astype(np.float32)


def encode_label(pth):
    """Return integer encoding for event history in Noddyverse"""
    return torch.tensor([labels[e] for e in pth.split("_")], dtype=torch.uint8)


class Norm:
    def __init__(self, clip=5_000):
        # TODO use our previously designed norm method
        # TODO derive stats for normalising GRAVITY data
        # OR rEad tHOsE PaPeRS
        self.clip = clip
        self.max = clip
        self.min = -clip
        assert self.min < self.max

    def min_max_clip(self, grid):
        """Clip to specified range and min-max normalise to range [-1, 1]"""
        grid[grid < self.min] = self.min
        grid[grid > self.max] = self.max
        return ((grid - self.min) / (self.max - self.min) * 2) - 1

    def inverse_mmc(self, grid):
        f"""Inverse of min_max_clip, limited to +-{self.clip}"""
        return ((grid + 1) / 2) * (self.max - self.min) + self.min

    def sample_min_max(self, grid):
        """Simple min-max normalisation unique to presented sample"""
        return ((grid - grid.min()) / (grid.max() - grid.min()) * 2) - 1


class NoddyDataset(Dataset):
    """Create a Dataset to access magnetic, gravity, and surface geology
    from the Noddyverse (https://doi.org/10.5194/essd-14-381-2022)

    The data are returned concatenated, with the geology model and
    unique identifier accessible.

    Args:
        model_dir: Path to Noddy root folder
        load_magnetics: Load magnetic data
        load_gravity: Load gravity data
        load_geology: Optionally load the g12 voxel model, surface layer
        encode_label: Encode the event history as a tensor
        m_names_precompute: Precomputed list of model names - useful for workers
        kwargs: Parameter dictionary for survey / augmentation
    """

    def __init__(
        self,
        model_dir=None,
        load_magnetics=True,
        load_gravity=False,
        load_geology=False,
        encode_label=False,
        m_names_precompute=None,
        **kwargs,
    ):
        super().__init__()

        self.norm = Norm(clip=5000).min_max_clip
        self.m_dir = Path(model_dir)
        
        if m_names_precompute is None:
            t0 = time.perf_counter()
            # A reasonable speedup can be had by computing this once and sharing
            # to all dataloader workers.
            logging.getLogger(__name__).debug(f"Computing model names")
            his_files = self.m_dir.glob("**/*.his*")
            m_names_precompute = [(p.parent.name, p.name[:-7]) for p in his_files]
            m_names_precompute = np.array(m_names_precompute).astype(np.string_)

        # List of unique folder/names in model_dir - (named after a timestamp)
        # See https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.m_names = m_names_precompute
       
        self.load_magnetics = load_magnetics
        self.load_gravity = load_gravity
        self.load_geology = load_geology
        self.encode_label = encode_label
        self.len = len(self.m_names)
        if not self.len:
            raise FileNotFoundError(f"No files found in {self.m_dir.absolute()}")

    def _process(self, index):
        """Convert parsed numpy arrays to tensors and augment"""
        self.data = {}

        parent, name = self.m_names[index]
        self.parent = str(parent, encoding="utf-8")
        f = (
            self.m_dir
            / self.parent
            / "models_by_code"
            / "models"
            / self.parent
            / str(name, encoding="utf-8")
        )

        if self.encode_label:
            self.data["label"] = encode_label(self.parent)

        _data = [
            torch.from_numpy(self.norm(g)).unsqueeze(0)
            for g in parse_geophysics(f, self.load_magnetics, self.load_gravity)
        ]

        if self.load_magnetics and self.load_gravity:
            self.data["gt_grid"] = torch.stack(_data, dim=0)
        else:
            self.data["gt_grid"] = _data

        if self.load_geology:
            # This is mildly expensive - Could pass layer to np.loadtxt skips?
            # It would be good to slice this too.
            # TODO confirm 0 is top (surface) layer ("ground truth" geology map)
            self.data["geo"] = torch.from_numpy(
                parse_geology((f).with_suffix(".g12.gz"), layer=0)
            )

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        self._process(index)
        return self.data
