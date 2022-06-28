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
    return np.transpose(model, (0, 2, 1))[layer, :, :]


def parse_geophysics(pth: Path, mag=False, grv=False):
    """Return forward model values from Noddy geophysics .mag.gz and .grv.gz"""

    files = (pth.with_suffix(".mag.gz"), pth.with_suffix(".grv.gz"))
    if not grv:
        files = [files[0]]
    if not mag:
        files = [files[1]]
    for pth in files:
        yield np.loadtxt(pth, skiprows=8, dtype=np.float32)
    # return pd.read_csv(pth,sep="\t",skiprows=8,header=None,usecols=range(200),dtype=np.float32,na_filter=False,).values.astype(np.float32)


def encode_label(pth):
    """Return integer encoding for event history in Noddyverse"""
    return torch.tensor([labels[e] for e in pth.split("_")], dtype=torch.uint8)


class NoddyDataset(Dataset):
    """Create a Dataset to access magnetic, gravity, and surface geology
    from the Noddyverse (https://doi.org/10.5194/essd-14-381-2022)

    The data are returned concatenated, with the geology model and
    unique identifier accessible.

    Parameters:
        model_dir: Path to Noddy root folder
        survey/augment: Do survey / augment. See method.
        load_geology: Optionally load the g12 voxel model, surface layer
        kwargs: Parameter dictionary for survey / augmentation
    """

    def __init__(
        self,
        model_dir=None,
        load_magnetics=True,
        load_gravity=False,
        load_geology=False,
        encode_label=False,
        augment=False,
        **kwargs,
    ):
        super().__init__()

        self.m_dir = Path(model_dir)
        his_files = self.m_dir.glob("**/*.his*")
        self.m_names = np.array(
            [(p.parent.name, p.name[:-7]) for p in his_files]
        ).astype(np.string_)
        # List of unique folder/names in model_dir - (named after a timestamp)
        # See https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.load_magnetics = load_magnetics
        self.load_gravity = load_gravity
        self.load_geology = load_geology
        self.encode_label = encode_label
        self.len = len(self.m_names)
        if not self.len:
            raise FileNotFoundError(f"No files found in {self.m_dir.absolute()}")
        if augment:
            self.augs = {
                "hflip": augment.get("hflip", False),
                "vflip": augment.get("vflip", False),
                "rotate": augment.get("rotate", False),
                "noise": augment.get("noise", False),
            }
        else:
            self.augs = None

    def _augment(self, *tensors):
        if self.augs.get("hflip") and torch.rand(1) < 0.5:
            (TF.hflip(t) for t in tensors)
        if self.augs.get("vflip") and torch.rand(1) < 0.5:
            (TF.vflip(t) for t in tensors)
        if self.augs.get("rotate") and torch.rand(1) < 0.5:
            (TF.rotate(t, 90) for t in tensors)
        if self.augs.get("noise"):
            raise NotImplementedError
            # (add_noise(t, noise) for t in tensors)

    def _norm(self, grid, norm_type="min_max_clip"):
        # TODO use our previously designed norm method
        # TODO derive stats for normalising GRAVITY data
        # OR rEad tHOsE PaPeRS
        if norm_type == "sample_min_max":
            return ((grid - grid.min()) / (grid.max() - grid.min()) * 2) - 1
        elif norm_type == "min_max_clip":
            clip = 5_000
            grid[grid < -clip] = -clip
            grid[grid > clip] = clip
            grid = ((grid - -clip) / (clip - -clip) * 2) - 1
            return grid
        else:
            raise NotImplementedError("Unsupported normalisation method")

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
            torch.from_numpy(self._norm(g))
            for g in parse_geophysics(f, self.load_magnetics, self.load_gravity)
        ]

        if self.load_magnetics and self.load_gravity:
            self.data["gt_grid"] = torch.stack(_data, dim=0)
        else:
            self.data["gt_grid"] = _data

        if self.load_geology:
            # This is mildly expensive - Could pass layer to np.loadtxt skips?
            self.data["geo"] = torch.from_numpy(
                parse_geology((f).with_suffix(".g12.gz"), layer=0)
            )

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        self._process(index)

        if self.augs:
            raise NotImplementedError("Need to troubleshoot Tensor shapes")
            self._augment([t for t in self.data[""]])

        return self.data
