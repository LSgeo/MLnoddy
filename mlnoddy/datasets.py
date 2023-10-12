import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset  # , IterableDataset, get_worker_info

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


@lru_cache()
def load_noddy_csv(csv_path):
    """Return list of [(Event_triplet_string, model_filename)] present
    in csv_path. This can be used to generate a list of file names for models.
    """
    if not csv_path:  # Return empty list for compatability
        return []

    lpath = Path(csv_path)
    paths = lpath.read_text().split(",")[6::5]
    events = lpath.read_text().split(",")[10::5]
    return [
        (e.replace(" ", "_").split()[0], p.split("/")[2]) for e, p in zip(events, paths)
    ]


@lru_cache()
def load_noddy_allow_list(alllist, blocklist, file_cache=".noddy_allowlist.npy"):
    """Generate a list of models in alllist that are not in blocklist"""
    if blocklist is not None and Path(file_cache).exists():
        allowlist = np.load(Path(file_cache), mmap_mode="r")
        print(f"Using cached file {Path(file_cache).absolute()} with {len(allowlist)} elements")
    else:
        alllist = load_noddy_csv(alllist)
        blocklist = load_noddy_csv(blocklist)
        allowlist = [e_n for e_n in alllist if e_n not in blocklist]
        try:
            np.save(file_cache, allowlist)
        except OSError:  # May mask other OS errors
            pass  # Case multiple datasets are concurrent in RAM
    return allowlist


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
    for f in files:
        d = np.ascontiguousarray(np.loadtxt(f, skiprows=8, dtype=np.float32))
        if d.min() == d.max():  # If at least one noddyverse model is all 0 nT
            d[0, 0] = 1  # norm and grid breaks, so add a tiny nT value
        yield d
    # yield pd.read_csv(pth,sep="\t",skiprows=8,header=None,usecols=range(200),dtype=np.float32,na_filter=False,).values.astype(np.float32)


@lru_cache()
def encode_label(pth):
    """Return integer encoding for event history in Noddyverse"""
    return torch.tensor([labels[e] for e in pth.split("_")], dtype=torch.uint8)


class Norm:
    """Handle normalisation and unnormalisation of data with set min/max"""

    def __init__(self, clip_min=-10000, clip_max=10000, out_vals=(0, 1)):
        # TODO use our previously designed norm method
        # OR rEad tHOsE PaPeRS
        """Defaults are suitable for noddyverse TMI"""

        self.min = clip_min
        self.max = clip_max
        self.out_range = out_vals[1] - out_vals[0]
        self.out_min = out_vals[0]
        self.out_max = out_vals[1]
        if self.min >= self.max:
            raise ValueError(f"Min ({self.min}) must be less than Max ({self.max})")

    def per_sample_norm(self, grid):
        """Per-tile normalisation to range [-1, 1]
        We only set min and max on the HR and use these to normalise both HR/LR
        inverse_mmc() is suitable to unnormalise.
        """
        if self.min is None or self.max is None:
            self.min = grid.min().item()
            self.max = grid.max().item()
        return (
            (grid - self.min) / (self.max - self.min) * self.out_range
        ) + self.out_min

    def min_max_clip(self, grid):
        """Clip to specified range and min-max normalise to range [0, 1]"""
        grid[grid < self.min] = self.min
        grid[grid > self.max] = self.max
        return (
            self.out_range * ((grid - self.min) / (self.max - self.min))
        ) + self.out_min

    def inverse_mmc(self, grid):
        """Inverse of min_max_clip, limited to +-self.clip"""
        return (
            (grid - self.out_min) * ((self.max - self.min) / self.out_range)
        ) + self.min

    def sample_min_max(self, grid):
        """Simple min-max normalisation unique to presented sample"""
        return (
            self.out_range * (grid - grid.min()) / (grid.max() - grid.min())
        ) + self.out_min


class NoddyDataset(Dataset):
    """Dataset to access magnetic, gravity, and surface geology
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
        use_dset_slice=[0, -1],
        norm=None,
        **kwargs,
    ):
        super().__init__()
        if norm is not None:
            self.norm = Norm(
                clip_min=norm[0], clip_max=norm[1], out_vals=(0, 1)
            ).min_max_clip
            self.unorm = Norm(
                clip_min=norm[0], clip_max=norm[1], out_vals=(0, 1)
            ).inverse_mmc
        else:
            self.norm = Norm(out_vals=(0, 1)).per_sample_norm
            self.unorm = Norm(out_vals=(0, 1)).inverse_mmc
        self.m_dir = Path(model_dir)

        model_list = self._generate_model_lists(
            noddylist=kwargs["noddylist"],
            blocklist=kwargs["blocklist"],
            events=kwargs.get("events"),
        )
        start_id, stop_id = use_dset_slice
        self.m_names = model_list[start_id:stop_id]
        logging.getLogger(__name__).info(
            f"Using dataset slice of [{start_id}, {stop_id}]"
        )

        self.load_magnetics = load_magnetics
        self.load_gravity = load_gravity
        self.load_geology = load_geology
        self.encode_label = encode_label
        self.len = len(self.m_names)
        if not self.len:
            if not self.m_dir.exists():
                raise FileNotFoundError(f"{self.m_dir.absolute()} does not exist")
            else:
                raise FileNotFoundError(f"No files found in {self.m_dir.absolute()}")

    def _generate_model_lists(self, noddylist=None, blocklist=None, events=[]):
        """Collection of functions to process dataset lists"""
        # See https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        model_list = load_noddy_allow_list(noddylist, blocklist)

        if events is not None:
            event_filter = [any(e in h[0] for e in events) for h in model_list]

        model_list = np.array(model_list).astype(np.string_)

        if events is not None:  # bool selection only on arr
            model_list = model_list[event_filter]

        return model_list

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
            torch.from_numpy(g).unsqueeze(0)
            for g in parse_geophysics(f, self.load_magnetics, self.load_gravity)
        ]

        # self.data["gt_path"] = torch.tensor(name.astype(np.string_))

        if self.load_geology:
            # This is mildly expensive - Could pass layer to np.loadtxt skips?
            # It would be good to slice this too.
            # TODO confirm 0 is top (surface) layer ("ground truth" geology map)
            self.data["geo"] = torch.from_numpy(
                parse_geology((f).with_suffix(".g12.gz"), layer=0)
            )

        if self.load_magnetics and self.load_gravity:
            self.data["gt_grid"] = torch.stack(_data, dim=0)
        else:
            self.data["gt_grid"] = _data[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        self._process(index)
        return self.data


# class NoddyIterableDataset(IterableDataset):
#     """derp"""

#     def __init__(self, start:int, end:int, model_dir:Path, model_names):
#         super().__init__()
#         self.start = start
#         self.end = end
#         self.file_list = model_names

#     def __iter__(self):
#         worker_info = get_worker_info()
#         if worker_info is None:
#             iter_start = self.start
#             iter_end = self.end
#         else:
#             worker_id = worker_info.id
#             per_worker = int(np.ceil((self.end - self.start) / float(worker_info.num_workers)))
#             iter_start = self.start + worker_id * per_worker
#             iter_end = min(iter_start + per_worker, self.end)
#             return iter(range(iter_start, iter_end))


def e_size(s) -> float:
    """Calculate inch for pyplot from elsevier figure widths
    https://beta.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing
    """
    if isinstance(s, (int, float)):
        mm = s
    elif isinstance(s, str):
        if s.lower() in ["minimal"]:
            mm = 30
        elif s == "1" or s.lower() in ["single"]:
            mm = 90
        elif s == "1.5":
            mm = 140
        elif s == "2" or s.lower() in ["double", "full"]:
            mm = 190
        else:
            raise ValueError("Unsupported target size")
    else:
        raise ValueError(f"{s=}, {mm=}")
    return mm / 25.4
