from . import datasets

import imageio
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def merge_z_slices(dir_path):
    """Merge multiple Noddy forward models into a single 3D array
    For example, create a 200x200x200 volume of synthetic measurements.
    Args:
        dir_path (str): Path to directory to merge all .mag (and .grv)
        spacing (int): Spacing at which slices were modelled
    """

    dir_path = Path(dir_path)
    mag_files = sorted(dir_path.glob("*.mag"))
    grv_files = sorted(dir_path.glob("*.grv"))

    # get x y extent from header
    _, xlen, ylen, _ = mag_files[0].read_text().splitlines()[3].split()
    zlen = len(mag_files)

    return np.stack(
        [np.genfromtxt(f, dtype=np.float32, skip_header=8) for f in mag_files], axis=-1
    )
