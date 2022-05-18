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
    return np.loadtxt(pth, skiprows=8, dtype=np.float32)


def subsample(line_spacing, sample_spacing, cell_size, heading, *rasters):
    """Run a mock-survey on a geophysical raster.
    Designed for use with Noddy forward models, as part of a Pytorch dataset.

    Params:
        raster: input Tensor forward model
        line_spacing: in meters, spacing between parallel lines
        sampling_rate: in meters, spacing between cells along line
        cs: ground truth cell size, 20 m for Noddy models
        heading: "NS" for columns as lines, "EW" for rows as lines

    The Noddyverse dataset is a suite of 1 Million 200x200x200 petrophysical
    voxels, at a designated size of 20 m per pixel. Forward models in the
    Noddyverse (https://doi.org/10.5194/essd-14-381-2022) are generated
    as per below:
        | Geophysical forward models were calculated using a Fourier domain
        | formulation using reflective padding to minimise (but not remove)
        | boundary effects. The forward gravity and magnetic field calculations
        | assume a flat top surface with a 100 m sensor elevation above this
        | surface and the Earth's magnetic field with vertical inclination,
        | zero declination and an intensity of 50000nT.

    Note that every single cell of the forward model has a calulated forward
    model, i.e. they are 200x200, with no interpolation (for a 20 m cell size)

    We simulate an airborne survey, by selecting rows (flight lines) of pixels
    at every n m. We can (but not by default) also subsample along rows.

    """
    ls = int(line_spacing / cell_size)
    sr = int(sample_spacing / cell_size)
    if heading.upper() in ["EW", "E", "W"]:
        ls, sr = sr, ls  # swap convention to emulate survey direction

    x, y = np.meshgrid(np.arange(200), np.arange(200), indexing="xy")
    x = cell_size * x[::ls, ::sr]
    y = cell_size * y[::ls, ::sr]
    zs = [raster[::ls, ::sr] for raster in rasters]

    return x, y, zs


def grid(x, y, z, cs=20, name="foward"):
    """Grid a subsampled noddy forward model.

    params:
        x,y: x, y coordinates
        z: geophysical response value
        cs: cell size for interpolating grid

    See docstring for subsample() for further notes.
    """

    grd = vd.ScipyGridder("cubic").fit((x, y), z)
    grid_raster = grd.grid(
        region=[0, cs * 199, 0, cs * 199],
        spacing=cs,
        dims=["x", "y"],
        data_names=name,
    )

    return grid_raster.get(name).values.astype(np.float32)


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
        self.data = {
            "gt": torch.stack((self.gt_mag, self.gt_grv), dim=0),
            **self.data,
        }

        if "survey" in self.kwargs:
            sp = {"ls": 20, "ss": 20, "cs": 20, "h": "NS", **self.kwargs["survey"]}
            x, y, (self.hr_mag, self.hr_grv) = subsample(
                *sp.values(), self.gt_mag, self.gt_grv
            )
            self.hr_mag = torch.from_numpy(grid(x, y, self.hr_mag))
            self.hr_grv = torch.from_numpy(grid(x, y, self.hr_grv))
            self.data = {
                "hr": torch.stack((self.hr_mag, self.hr_grv), dim=0),
                **self.data,
            }

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


import matplotlib.pyplot as plt
noddy_model_dir = Path(r"data/DYKE_FOLD_FAULT")
dset = NoddyDataset(noddy_model_dir, load_geology=True)

dset = NoddyDataset(
    noddy_model_dir,
    load_geology=True,
    survey={"ls": 400, "ss": 20},
    # augment={},
)

for i in range(5):
    sample = dset[i]

    fig, axes = plt.subplots(3, 3, constrained_layout=True)
    # [ax.set_axis_off() for ax in axes.ravel()]
    [ax.set_xlim(0, 200) for ax in axes.ravel()]
    [ax.set_ylim(0, 200) for ax in axes.ravel()]
    ([mag, grv, geo], [mgd, ggd, ax1_off], [mdf, gdf, ax2_off]) = axes

    ax1_off.set_axis_off()
    ax2_off.set_axis_off()
    plt.suptitle(", ".join(sample["label"]))
    mag.imshow(sample["gt"][0])
    grv.imshow(sample["gt"][1])
    geo.imshow(sample["geo"])

    mgd.imshow(sample["hr"][0])
    ggd.imshow(sample["hr"][1])

    mdf.imshow(sample["gt"][0] - sample["hr"][0])
    gdf.imshow(sample["gt"][1] - sample["hr"][1])
    plt.show()

