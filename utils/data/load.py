import numpy as np
import os

from SimpleITK import GetArrayFromImage, ReadImage


def load_itk(filename):
    """
    This function reads a '.mhd' file using SimpleITK
    and return the image array, origin and spacing of the image.
    """
    # Reads the image using SimpleITK
    itk_image = ReadImage(filename)

    # Convert the image to a numpy array first and then
    # shuffle the dimensions to get axis in the order z,y,x
    ct_scan = GetArrayFromImage(itk_image)

    # Read the origin of the ct_scan, will be used to convert
    # the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itk_image.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itk_image.GetSpacing())))

    return ct_scan, origin, spacing


def load_scan(path):
    slices = [ReadImage(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.GetMetaData('0020|1041')))
    scan = [GetArrayFromImage(slice)[0] for slice in slices]

    return np.array(scan)
