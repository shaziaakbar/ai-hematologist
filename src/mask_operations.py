from scipy import ndimage
import numpy as np
from skimage.measure import regionprops


def keep_largest_blob(mask):
    """ Function to remove all big largest blob in a boolean mask

    Args:
        mask (numpy): array containing mask contents

    Returns:
        Numpy array containing largest blob in mask.
    """
    labeled_array, num_features = ndimage.label(mask)
    regions = regionprops(labeled_array)

    # Find the region with the largest area (except background)
    sorted_regions = sorted(regions, key=lambda region: region.area, reverse=True)
    if len(sorted_regions) < 2:
        return mask

    largest_region_mask = labeled_array == sorted_regions[1].label
    return largest_region_mask.astype('bool')


def smooth_mask(mask, iterations=2):
    """ Function to smooth out binary mask using dilation and erosion

    Args:
        mask (numpy): array containing mask contents
        iterations (int): amount of smoothing to perform

    Returns:
        Smoothed mask.
    """
    for i in range(iterations):
        mask = ndimage.binary_erosion(mask, iterations=1)
        mask = ndimage.binary_dilation(mask, iterations=1)
    return mask.astype('bool')
