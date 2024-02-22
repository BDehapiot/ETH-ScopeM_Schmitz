#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm

# Functions
from functions import preprocess, get_patches, merge_patches

#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path("D:\local_Schmitz\data")
model_path = Path(Path.cwd(), "model_weights.h5")
stack_name = "organoid_immune_19.tif"
# stack_name = "organoid_immune_27.tif"
# stack_name = "organoid_immune_82.tif"
# stack_name = "organoid_immune_90.tif"
# stack_name = "organoid_immune_189.tif"

# Frames
frame = "all"

# Parameters
downscale_factor = 2
size = 1024 // downscale_factor
overlap = size // 8

#%% Pre-processing ------------------------------------------------------------

# Open data
path = Path(local_path, stack_name)
print("Open data       :", end='')
t0 = time.time()
stack = io.imread(path)[..., 2]
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# Preprocessing
print("Preprocessing   :", end='')
t0 = time.time()
stack = preprocess(stack, downscale_factor, mask=False)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# Extract patches
print("Extract patches :", end='')
t0 = time.time()
patches = np.stack(get_patches(stack, size, overlap))
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

#%% Predict -------------------------------------------------------------------

# Define & compile model
model = sm.Unet(
    'resnet34', 
    input_shape=(None, None, 1), 
    classes=1, 
    activation='sigmoid', 
    encoder_weights=None,
    )
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', 
    metrics=['mse']
    )

# Load weights
model.load_weights(model_path) 

# Predict
predict = model.predict(patches).squeeze()

# Merge patches
print("Merge patches   :", end='')
t0 = time.time()
predict = merge_patches(predict, stack.shape, size, overlap)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# Display
viewer = napari.Viewer()
viewer.add_image(np.stack(stack)) 
viewer.add_image(np.stack(predict)) 

# # Save
# io.imsave(
#     Path(local_path, avi_name.replace(".avi", "_rescale.tif")),
#     arr.astype("float32"), check_contrast=False
#     )
# io.imsave(
#     Path(local_path, avi_name.replace(".avi", "_predict.tif")),
#     predict.astype("float32"), check_contrast=False
#     )