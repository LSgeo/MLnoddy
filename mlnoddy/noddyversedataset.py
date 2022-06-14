from pathlib import Path

import numpy as np
import verde as vd
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


def subsample(parameters: dict, scale=1, *rasters):
    input_cell_size = 20
    """Run a mock-survey on a geophysical raster.
    Designed for use with Noddy forward models, as part of a Pytorch dataset.

    Args:
        parameters:
            line_spacing: in meters, spacing between parallel lines
            sample_spacing: in meters, spacing between points along line
            heading: "NS" for columns as lines, "EW" for rows as lines
        scale: multiplier for low resolution (vs high resolution)
        input_cell_size: ground truth cell size, 20 m for Noddy models
        *rasters: input Tensor forward model

    The Noddyverse dataset is a suite of 1 Million 200x200x200 petrophysical
    voxels, at a designated size of 20 m per pixel. Forward models in the
    Noddyverse (https://doi.org/10.5194/essd-14-381-2022) are generated
    as per below:
        Geophysical forward models were calculated using a Fourier domain
        formulation using reflective padding to minimise (but not remove)
        boundary effects. The forward gravity and magnetic field calculations
        assume a flat top surface with a 100 m sensor elevation above this
        surface and the Earth's magnetic field with vertical inclination,
        zero declination and an intensity of 50000nT.

    Note that every single cell of the forward model has a calculated forward
    model, i.e. they are 200x200, with no interpolation (for a 20 m cell size)

    We simulate an airborne survey, by selecting rows (flight lines) of pixels
    at every n m. We can (but not by default) also subsample along rows (ss).

    """
    cs = input_cell_size
    ss = int(parameters.get("sample_spacing") / cs)
    ls = int(parameters.get("line_spacing") * scale / cs)

    if parameters.get("heading").upper() in ["EW", "E", "W"]:
        ls, ss = ss, ls  # swap convention to emulate survey direction

    x, y = np.meshgrid(np.arange(200), np.arange(200), indexing="xy")
    x = cs * x[::ls, ::ss]
    y = cs * y[::ls, ::ss]
    zs = [raster[::ls, ::ss] for raster in rasters]

    return x, y, zs


def grid(x, y, zs, ls: int = 20, cs_fac: int = 4):
    in_cs: int = 20
    """Grid a subsampled noddy forward model.

    params:
        x, y: x, y coordinates
        z: geophysical response value
        line_spacing: sample line spacing, to calculate target cell_size
        cs_fac: line spacing to cell size factor, typically 4 or 5
        name: data_variable name
        input_cell_size: Input model cell size, 20m for Noddyverse

    See docstring for subsample() for further notes.
    """
    for rz in zs:
        gridder = vd.ScipyGridder("cubic").fit((x, y), rz)

        yield gridder.grid(
            region=[0, in_cs * 200, 0, in_cs * 200],
            spacing=ls / cs_fac,
            dims=["x", "y"],
            data_names="forward",
        ).get("forward").values.astype(np.float32)


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

    def _process(self, index):
        """Convert parsed numpy arrays to tensors and augment"""
        parent, name = self.m_names[index]
        self.parent = str(parent, encoding="utf-8")
        f = self.m_dir / self.parent / "models_by_code" / "models" / self.parent / str(name, encoding="utf-8")

        self.data = {"label": encode_label(self.parent)}

        _data = [
            torch.from_numpy(g)
            for g in parse_geophysics(f, self.load_magnetics, self.load_gravity)
        ]
        self.data = {"gt": torch.stack(_data, dim=0), **self.data}

        if self.load_geology:
            # This is mildly expensive - Could pass layer to np.loadtxt skips?
            self.data = {
                "geo": torch.from_numpy(
                    parse_geology((f).with_suffix(".g12.gz"), layer=0)
                ),
                **self.data,
            }

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        self._process(index)

        if self.augs:
            raise NotImplementedError("Need to troubleshoot Tensor shapes")
            self._augment([t for t in self.data[""]])

        return self.data


class HRLRNoddyDataset(NoddyDataset):
    """Load a Noddy dataset with high- and low-resolution grids
    If Heading is not specified, it will be randomly selected from NS or EW
    """

    def __init__(self, **kwargs):
        self.scale = kwargs.get("scale", 2)
        self.random_heading = not bool(kwargs.get("heading"))
        self.sp = {
            "line_spacing": kwargs.get("line_spacing", 20),
            "sample_spacing": kwargs.get("sample_spacing", 20),
            "heading": kwargs.get("heading", None),  # Default will be random
        }
        super().__init__(**kwargs)

    def _process(self, index):
        super()._process(index)

        hls = self.sp["line_spacing"]
        lls = hls * self.scale
        if self.random_heading:
            if torch.rand(1) < 0.5:
                self.sp["heading"] = "NS"
            else:
                self.sp["heading"] = "EW"

        hr_x, hr_y, _hr_zs = subsample(self.sp, 1, *self.data["gt"])
        _hr_grids = [
            torch.from_numpy(g) for g in grid(hr_x, hr_y, _hr_zs, ls=hls, cs_fac=4)
        ]
        self.data = {"hr": torch.stack(_hr_grids, dim=0), **self.data}

        lr_x, lr_y, _lr_zs = subsample(self.sp, self.scale, *self.data["gt"])
        _lr_grids = [
            torch.from_numpy(g) for g in grid(lr_x, lr_y, _lr_zs, ls=lls, cs_fac=4)
        ]
        self.data = {"lr": torch.stack(_lr_grids, dim=0), **self.data}
