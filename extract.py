#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path
from functions import preprocess

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path("D:\local_Schmitz\data")
train_path = Path(Path.cwd(), "data", "train")
hstack_paths = list(local_path.glob("*.tif"))

# Parameters
nImg = 10
downscale_factor = 2
np.random.seed(42)

#%% Extract -------------------------------------------------------------------

for hstack_path in hstack_paths:
    stack = io.imread(hstack_path)[..., 2]
    idxs = np.random.choice(range(stack.shape[0]), size=nImg, replace=False)
    print(idxs)
    for idx in idxs:
        img = stack[idx, ...]
        img = preprocess(img, downscale_factor=downscale_factor)
        img_name = hstack_path.name.replace(".tif", f"_t{idx}.tif")
        io.imsave(
            Path(train_path, img_name),
            img.astype("float32"), check_contrast=False,
            )        