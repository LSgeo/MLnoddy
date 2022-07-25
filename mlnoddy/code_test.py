if __name__ == "__main__":

    from pathlib import Path
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    import numpy as np
    from torch.utils.data import DataLoader

    import datasets as nv

    cfg = SimpleNamespace()
    cfg.encode_label = False
    cfg.load_magnetics = True
    cfg.load_gravity = False
    cfg.load_geology = False
    cfg.batch_size = None  # Disable automatic batching for iterable dataset
    cfg.shuffle = True
    cfg.pin_memory = False
    cfg.num_workers = 0

    train_dataset = nv.NoddyDataset(
        model_dir=r"D:\luke\Noddy_data\noddyverse_train_data",
        encode_label=cfg.encode_label,
        load_magnetics=cfg.load_magnetics,
        load_gravity=cfg.load_gravity,
        load_geology=cfg.load_geology,
    )

    model_names = model_names = [
        (p.parent.name, p.name[:-7]) for p in Path(cfg.train_path).glob("**/*.his*")
    ]
    model_names = np.array(model_names).astype(np.string_)

    # train_dataset = nv.NoddyIterableDataset(start=0, end=16, model_dir=r"D:\luke\Noddy_data\noddyverse_train_data", model_names=model_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
        persistent_workers=bool(cfg.num_workers),
    )

    for batch in train_loader:
        for i in range(cfg.batch_size):

            fig, axes = plt.subplots(3, 3, constrained_layout=True, figsize=(8, 10))
            label = [nv.inverse_labels[e] for e in batch["label"][i].numpy()]

            plt.suptitle("_".join(label))
            # [ax.set_axis_off() for ax in axes.ravel()]
            # [ax.set_xlim(0, 200) for ax in axes.ravel()]
            # [ax.set_ylim(0, 200) for ax in axes.ravel()]
            ([mag, grv, geo], [mgd, ggd, ax1_off], [mdf, gdf, ax2_off]) = axes

            ax1_off.set_axis_off()
            ax2_off.set_axis_off()
            n = 0

            if cfg.load_magnetics:
                mag.set_title("Magnetics Ground-truth")
                tmi = mag.imshow(batch["gt_grid"][i][0])
                plt.colorbar(tmi, cax=mgd)
                n = 1

            if cfg.load_gravity:
                grv.set_title("Gravity Ground-truth")
                g = grv.imshow(batch["gt_grid"][i][n])
                plt.colorbar(g, cax=ggd)

            if cfg.load_geology:
                geo.set_title("Surface Geology")
                geo.imshow(batch["geo"][i])
            else:
                geo.set_axis_off()

            plt.savefig("test.png", facecolor="white", transparent=False)
            plt.close()
