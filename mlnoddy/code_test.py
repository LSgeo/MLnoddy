from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
import colorcet as cc
import datasets as nv

cfg = SimpleNamespace()
cfg.encode_label = True
cfg.load_magnetics = True
cfg.load_gravity = True
cfg.load_geology = True
cfg.batch_size = 1
cfg.shuffle = False
cfg.pin_memory = False
cfg.num_workers = 0
cfg.indices = [95, 143, 19]  # Different source data
cfg.dataset_path = Path("../noddyverse_data")  # I use Dyke_Plug_Unconformity folder
cfg.norm = [-5000, 5000]

model_names = [(p.parent.name, p.name[:-7]) for p in cfg.dataset_path.glob("**/*.his*")]
model_names = np.array(model_names).astype(np.string_)

train_dataset = nv.NoddyDataset(
    model_dir=cfg.dataset_path,
    encode_label=cfg.encode_label,
    load_magnetics=cfg.load_magnetics,
    load_gravity=cfg.load_gravity,
    load_geology=cfg.load_geology,
    norm=cfg.norm,
)
train_dataset = Subset(train_dataset, cfg.indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=cfg.shuffle,
    pin_memory=cfg.pin_memory,
    num_workers=cfg.num_workers,
    persistent_workers=bool(cfg.num_workers),
)

for i, batch in enumerate(train_loader):
    for j in range(cfg.batch_size):

        fig, axes = plt.subplots(1, 3, constrained_layout=True, figsize=(9, 3))
        label = [nv.inverse_labels[e] for e in batch["label"].numpy()[0]]
        unorm = train_dataset.dataset.unorm

        plt.suptitle("_".join(label))
        ([mag, geo, grv]) = axes

        ops = dict(cmap=cc.cm.CET_L1, origin="lower")

        n = 0
        if cfg.load_magnetics:
            mag.set_title("Magnetics Forward Model")
            tmi = mag.imshow(unorm(batch["gt_grid"][0][n].squeeze()), **ops)
            plt.colorbar(tmi, ax=mag, location="bottom", label="nT")
            n = 1

        if cfg.load_gravity:
            grv.set_title("Gravity Forward Model")
            g = grv.imshow(unorm(batch["gt_grid"][0][n].squeeze()), **ops)
            # Note I'm lazily handling the norm/unnorm of the gravity - it's tied to mag still
            plt.colorbar(g, ax=grv, location="bottom", label="mGal?")

        if cfg.load_geology:
            geo.set_title("Surface Geology")
            ge = geo.imshow(batch["geo"].squeeze(), origin="lower")
            plt.colorbar(ge, ax=geo, location="bottom", label="Class")
        else:
            geo.set_axis_off()

        plt.savefig(f"index_{cfg.indices[i]}.png", facecolor="white", transparent=False)
        # plt.close()
