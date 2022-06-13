import datetime
from types import SimpleNamespace

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import noddyversedataset as nv

cfg = SimpleNamespace()
cfg.scale = 2
cfg.hr_linespacing = 12 * 20
cfg.load_magnetics = True
cfg.load_gravity = False
cfg.load_geology = False
cfg.batch_size = 4
cfg.shuffle = True
cfg.pin_memory = True
cfg.num_workers = 0

if __name__ == "__main__":
    train_dataset = nv.HRLRNoddyDataset(
        model_dir="data/models/",
        load_magnetics=cfg.load_magnetics,
        load_gravity=cfg.load_gravity,
        load_geology=cfg.load_geology,
        scale=cfg.scale,
        line_spacing=cfg.hr_linespacing,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
    )

    for batch in train_loader:
        for i in range(len(batch)):
            fig, axes = plt.subplots(3, 3, constrained_layout=True, figsize=(8, 10))
            label = [nv.inverse_labels[e] for e in batch["label"].numpy()[i]]

            plt.suptitle("_".join(label))
            # [ax.set_axis_off() for ax in axes.ravel()]
            # [ax.set_xlim(0, 200) for ax in axes.ravel()]
            # [ax.set_ylim(0, 200) for ax in axes.ravel()]
            ([mag, grv, geo], [mgd, ggd, ax1_off], [mdf, gdf, ax2_off]) = axes

            ax1_off.set_axis_off()
            ax2_off.set_axis_off()
            n = 0
            # plt.suptitle(", ".join((batch["label"])))
            if cfg.load_magnetics:
                mag.set_title("Magnetics Ground-truth")
                mag.imshow(batch["gt"][i][0])
                mgd.set_title("Magnetics HR")
                mgd.imshow(batch["hr"][i][0])
                mdf.set_title("Magnetics LR")
                mdf.imshow(batch["lr"][i][0])
                n = 1

            if cfg.load_gravity:
                grv.set_title("Gravity Ground-truth")
                grv.imshow(batch["gt"][i][n])
                ggd.set_title("Gravity HR")
                ggd.imshow(batch["hr"][i][n])
                gdf.set_title("Gravity LR")
                gdf.imshow(batch["lr"][i][n])

            if cfg.load_geology:
                geo.set_title("Surface Geology")
                geo.imshow(batch["geo"][i])
            else:
                geo.set_axis_off()

            plt.show()
