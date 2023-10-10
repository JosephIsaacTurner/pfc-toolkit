"""
Useful utilities for working with the Precomputed Connectome

"""

import os
import csv
import numpy as np
from tqdm import tqdm
from glob import glob
from nilearn import image
from nilearn._utils import check_niimg
from pfctoolkit import s3_tools

def load_roi(roi_path, s3_storage=None):
    """Load ROIs from path or CSV.

    Parameters
    ----------
    roi_path : str
        Path to CSV containing paths to NIfTI images OR Path to directory containing
        NIfTI images OR Path to NIfTI image.
    s3_storage : pfctoolkit.s3_tools.S3Storage, default None
            S3Storage object, to be used if roi_path is a path to a file in the cloud.


    Returns
    -------
    roi_paths : list of str
        List of paths to NIfTI image ROIs.

    """
    if s3_storage:
        roi_paths = s3_storage.list_s3_files(roi_path)
        if len(roi_paths) == 0:
            raise FileNotFoundError(f"No NIfTI images found in s3 path = {roi_path}!")
    elif os.path.isdir(roi_path):
        roi_paths = glob(os.path.join(os.path.abspath(roi_path), "*.nii*"))
        if len(roi_paths) == 0:
            raise FileNotFoundError("No NIfTI images found!")
    else:
        roi_extension = os.path.basename(roi_path).split(".")[1:]
        if "csv" in roi_extension:
            with open(roi_path, newline="") as f:
                reader = csv.reader(f)
                data = list(reader)
            roi_paths = [path for line in data for path in line]
        elif "nii" in roi_extension:
            roi_paths = [roi_path]
        else:
            raise ValueError(
                "Input File is not a NIfTI or a CSV containing paths to NIfTIs"
            )
    print(f"Found {len(roi_paths)} ROIs...")
    return roi_paths


def get_chunks(rois, config, s3_storage=None):
    """Get list of dicts containing Chunks to load and their associated ROIs

    Parameters
    ----------
    rois : list of str
        List of paths to ROIs to find chunks for.
    config : Config
        Configuration object.
    s3_storage : pfctoolkit.s3_tools.S3Storage, default None
            S3Storage object, to be used if chunk_idx is a file in the cloud.

    Returns
    -------
    dict of dicts
        Dict of dicts containing Chunk path and list of associated ROI paths.

    """
    chunk_dict = {}
    if s3_storage:
        chunk_map = s3_storage.get_file_from_cloud(config.get("chunk_idx"))
    else:
        chunk_map = image.load_img(config.get("chunk_idx"))
    for roi in tqdm(rois):
        if s3_storage:
            roi_image = s3_storage.get_file_from_cloud(roi)
        else:
            roi_image = image.load_img(roi)
        bin_roi_image = image.math_img("img != 0", img=roi_image)
        roi_chunks = image.math_img(
            "img * mask", img=bin_roi_image, mask=chunk_map
        ).get_fdata()
        chunks = np.unique(roi_chunks).astype(int)
        chunks = chunks[chunks != 0]
        for chunk in chunks:
            if chunk in chunk_dict:
                chunk_dict[chunk].append(roi)
            else:
                chunk_dict[chunk] = [roi]
    return chunk_dict


class NiftiMasker:
    """A Faster NiftiMasker.

    Attributes
    ----------
    mask_img : nibabel.nifti1.Nifti1Image
        Nifti binary mask.
    mask_idx : numpy.ndarray
        1D numpy.ndarray containing indexes from flattened mask image.
    mask_shape : tuple of ints
        Shape of mask_idx.
    mask_size : int
        Number of voxels in entire space, including outside the brain.

    """

    def __init__(self, mask_img=None, s3_storage=None):
        """

        Parameters
        ----------
        mask_img : Niimg-like object
            If string, consider it as a path to NIfTI image and call `nibabel.load()` on
            it. The '~' symbol is expanded to the user home folder. If it is an object,
            check if affine attribute is present, raise `TypeError` otherwise.
        s3_storage : pfctoolkit.s3_tools.S3Storage, default None
            S3Storage object, to be used if mask_img is a path to a file in the cloud.

        """
        if s3_storage and isinstance(mask_img, str):
            mask_img = s3_storage.get_file_from_cloud(mask_img)

        if isinstance(mask_img, str) and not os.path.exists(mask_img):
            raise ValueError(f"Local file not found at path: {mask_img}.")

        self.mask_img = check_niimg(mask_img)
        # self.mask_img = check_niimg(mask_img)
        self.mask_data = self.mask_img.get_fdata().astype(np.float32)
        (self.mask_idx,) = np.where((self.mask_data != 0).flatten())
        self.mask_shape = self.mask_data.shape
        self.mask_size = np.prod(self.mask_shape)

    def transform(self, niimg=None, weight=False, s3_storage=None):
        """Masks 3D Nifti file into 1D array. Retypes to float32.

        Parameters
        ----------
        niimg : Niimg-like object
            If string, consider it as a path to NIfTI image and call `nibabel.load()` on
            it. The '~' symbol is expanded to the user home folder. If it is an object,
            check if affine attribute is present, raise `TypeError` otherwise.
        weight : bool, default False
            If True, transform the niimg with weighting. If False, transform the niimg
            without weighting.
        s3_storage : pfctoolkit.s3_tools.S3Storage, default None
            S3Storage object, to be used if niimg is a path to a file in the cloud.

        Returns
        -------
        region_signals : numpy.ndarray
            Masked Nifti file.

        """
        if s3_storage and isinstance(niimg, str):
            niimg = s3_storage.get_file_from_cloud(niimg)
        niimg = np.nan_to_num(check_niimg(niimg).get_fdata()).astype(np.float32)
        if weight:
            img = np.multiply(self.mask_data, niimg)
        else:
            img = niimg
        return np.take(img.flatten(), self.mask_idx)

    def inverse_transform(self, flat_niimg=None):
        """Unmasks 1D array into 3D Nifti file. Retypes to float32.

        Parameters
        ----------
        flat_niimg : numpy.ndarray
            1D array to unmask.

        Returns
        -------
        niimg : nibabel.nifti1.Nifti1Image
            Unmasked Nifti.

        """
        new_img = np.zeros(self.mask_size, dtype=np.float32)
        new_img[self.mask_idx] = flat_niimg.astype(np.float32)
        return image.new_img_like(self.mask_img, new_img.reshape(self.mask_shape))

    def mask(self, niimg=None):
        """Masks 3D Nifti file into Masked 3D Nifti file.

        Parameters
        ----------
        niimg : Niimg-like object
            If string, consider it as a path to NIfTI image and call `nibabel.load()`
            on it. The '~' symbol is expanded to the user home folder. If it is an
            object, check if affine attribute is present, raise `TypeError` otherwise.

        Returns
        -------
        masked_niimg : nibabel.nifti1.Nifti1Image
            Masked Nifti file.

        """
        return self.inverse_transform(self.transform(niimg))
