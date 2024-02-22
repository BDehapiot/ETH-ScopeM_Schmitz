#%% Imports -------------------------------------------------------------------

import numpy as np
from joblib import Parallel, delayed
from skimage.transform import downscale_local_mean

#%% Functions -----------------------------------------------------------------

def preprocess(arr, rescale_factor, mask=False):
       
    def _preprocess(img):
        
        if mask:     
            img = img.astype(bool).astype(float)
            img = (downscale_local_mean(img, rescale_factor) > 0.5).astype(float)
            
        else:
            img = downscale_local_mean(img, rescale_factor)
            pMax = np.percentile(img, 99.9)
            img[img > pMax] = pMax
            img = (img / pMax).astype(float)
            
        return img
    
    if arr.ndim == 2:
        arr = _preprocess(arr)
        
    if arr.ndim == 3:
        outputs = Parallel(n_jobs=-1)(
            delayed(_preprocess)(img)
            for img in arr
            )
        arr = np.stack(outputs)
    
    return arr

# -----------------------------------------------------------------------------

def get_patches(arr, size, overlap):
    
    # Get dimensions
    if arr.ndim == 2: nT = 1; nY, nX = arr.shape 
    if arr.ndim == 3: nT, nY, nX = arr.shape
    
    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1, yPad2 = yPad // 2, (yPad + 1) // 2
    xPad1, xPad2 = xPad // 2, (xPad + 1) // 2
    
    # Pad array
    if arr.ndim == 2:
        arr_pad = np.pad(
            arr, ((yPad1, yPad2), (xPad1, xPad2)), mode='reflect') 
    if arr.ndim == 3:
        arr_pad = np.pad(
            arr, ((0, 0), (yPad1, yPad2), (xPad1, xPad2)), mode='reflect')         
    
    # Extract patches
    patches = []
    if arr.ndim == 2:
        for y0 in y0s:
            for x0 in x0s:
                patches.append(arr_pad[y0:y0 + size, x0:x0 + size])
    if arr.ndim == 3:
        for t in range(nT):
            for y0 in y0s:
                for x0 in x0s:
                    patches.append(arr_pad[t, y0:y0 + size, x0:x0 + size])
            
    return patches

# -----------------------------------------------------------------------------

def merge_patches(patches, shape, size, overlap):
    
    # Get dimensions 
    if len(shape) == 2: nT = 1; nY, nX = shape
    if len(shape) == 3: nT, nY, nX = shape
    nPatch = len(patches) // nT

    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1 = yPad // 2
    xPad1 = xPad // 2

    # Merge patches
    def _merge_patches(patches):
        count = 0
        arr = np.full((2, nY + yPad, nX + xPad), np.nan)
        for i, y0 in enumerate(y0s):
            for j, x0 in enumerate(x0s):
                if i % 2 == j % 2:
                    arr[0, y0:y0 + size, x0:x0 + size] = patches[count]
                else:
                    arr[1, y0:y0 + size, x0:x0 + size] = patches[count]
                count += 1 
        arr = np.nanmean(arr, axis=0)
        arr = arr[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
        return arr
        
    if len(shape) == 2:
        arr = _merge_patches(patches)

    if len(shape) == 3:
        patches = np.stack(patches).reshape(nT, nPatch, size, size)
        arr = Parallel(n_jobs=-1)(
            delayed(_merge_patches)(patches[t,...])
            for t in range(nT)
            )
        arr = np.stack(arr)
        
    return arr