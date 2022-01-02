"""
Tools to generate a Precomputed Functional Connectome.

"""
import os
import time
import numpy as np
from glob import glob
from tqdm import tqdm
from numba import jit
from pathlib import Path
from nilearn import image
from pfctoolkit import tools
from natsort import natsorted

@jit(nopython=True)
def extract_chunk_signals(connectome_mat, roi_mat):
    """Extract Chunk TC Signal from a connectome subject.

    Parameters
    ----------
    connectome_mat : ndarray
        Connectome subject TC matrix.
    roi_mat : ndarray
        Chunk ROI binary mask

    Returns
    -------
    roi_masked_tc : ndarray
        TC signals extracted from connectome TC according to ROI binary mask
    
    """
    roi_masked_tc = connectome_mat[:, roi_mat > 0]
    return roi_masked_tc

@jit(nopython=True)
def dot(a, b):
    return np.dot(a,b)

@jit(nopython=True)
def divide(a, b):
    return np.divide(a, b)

@jit(nopython=True)
def arctanh(a):
    return np.arctanh(a)

@jit(nopython=True)
def tanh(a):
    return np.tanh(a)

@jit(nopython=True)
def subtract(a,b):
    return np.subtract(a,b)

@jit(nopython=True)
def multiply(a,b):
    return np.multiply(a,b)

@jit(nopython=True)
def sqrt(a):
    return np.sqrt(a)

def make_fz_maps(connectome_files, roi_mat):
    """Make Fz Maps for a chunk ROI and a connectome subject.

    Parameters
    ----------
    connectome_files : (str, str)
        Tuple of paths for connectome subject TC matrix and TC norm vector.
    roi_mat : ndarray
        Brain-masked binary chunk ROI with shape (<size of brain>,)

    """
    connectome_mat = np.load(connectome_files[0]).astype(np.float32)
    connectome_norms_mat = np.load(connectome_files[1]).astype(np.float32)
    chunk_tc = extract_chunk_signals(connectome_mat, roi_mat)
    corr_num = dot(connectome_mat.T, chunk_tc)
    corr_denom = dot(connectome_norms_mat.reshape(-1,1),
                        np.linalg.norm(chunk_tc, axis=0).reshape(1,-1))
    np.seterr(invalid='ignore')
    corr = divide(corr_num, corr_denom)
    corr[np.isnan(corr)] = 0
    fz = arctanh(corr)
    # Fix infinite values in the case of single voxel autocorrelations
    finite_max = np.amax(np.ma.masked_invalid(fz), 1).data
    while(np.isinf(np.sum(fz))):
        fz[range(fz.shape[0]), np.argmax(fz, axis=1)] = finite_max
    return fz

@jit(nopython=True)
def welford_update_map(count, mean, M2, newMap):
    """Update a Welford map with data from a new map. See
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    Parameters
    ----------
    count : int
        Welford counter.
    mean : ndarray, shape (<brain size>, <ROI size>)
        Mean aggregate.
    M2 : ndarray, shape (<brain size>, <ROI size>)
        M2 aggregate.
    newMap : ndarray, shape (<brain size>, <ROI size>)
        New data to incorporate into a Welford map.

    Returns
    -------
    updatedAggregateMap : ndarray
        Updated Welford map.

    """
    count += 1
    delta = np.subtract(newMap, mean)
    mean += np.divide(delta, count)
    delta2 = np.subtract(newMap, mean)
    M2 += np.multiply(delta, delta2)
    return count, mean, M2

@jit(nopython=True)
def welford_finalize_map(count, mean, M2):
    """Convert a Welford map into maps of arrays containing the statistics 
    [mean, variance, sampleVariancce].

    Parameters
    ----------
    count : int
        Welford counter.
    mean : ndarray
        Welford Mean array.
    M2 : ndarray
        Welford M2 array.

    Returns
    -------
    ndarray, ndarray
        Array of Welford means and sample variances.

    """
    sampleVariance = np.divide(M2, count-1)
    return mean, sampleVariance

def make_stat_maps(count, mean, M2, output_dir, chunk_idx):
    """Generate statistical maps from welford maps and output to file.

    Parameters
    ----------
    count : int
        Number of connectome subjects.
    mean : ndarray
        Welford Mean array.
    M2 : ndarray
        Welford M2 array.
    output_dir : str
        Path to output directory.
    chunk_idx : int
        Chunk index number being processed.

    """
    mean, sampleVariance = welford_finalize_map(count, mean, M2)
    map_types = ['AvgR_Fz', 'AvgR', 'T']
    output_dir = os.path.abspath(output_dir)
    output_dirs = {}
    for map_type in map_types:
        Path(os.path.join(output_dir, map_type)).mkdir(parents=True,
                                                       exist_ok=True)
        output_dirs[map_type] = os.path.join(output_dir,
                                             map_type,
                                             f"{chunk_idx}_{map_type}.npy")
    print("Output Chunk files to:")
    # Save AvgR_Fz chunk
    np.save(output_dirs['AvgR_Fz'], mean)
    print(f"AvgR_Fz: {output_dirs['AvgR_Fz']}")
    # Save AvgR chunk
    np.save(output_dirs['AvgR'], tanh(mean))
    print(f"AvgR: {output_dirs['AvgR']}")
    # Save T Chunk
    ttest_denom = sqrt(divide(sampleVariance, count))
    np.save(output_dirs['T'], divide(mean, ttest_denom))
    print(f"T: {output_dirs['T']}")

def precomputed_connectome_chunk(mask,
                                 chunk_idx_mask,
                                 chunk_idx,
                                 connectome_dir,
                                 output_dir):
    """Generate a precomputed connectome FC map chunk (AvgR/Fz/T)

    Parameters
    ----------
    mask : str
        Path to binary mask.
    chunk_idx_mask : str
        Path to mask containing voxel-wise chunk labels.
    chunk_idx : int
        Index of chunk to process.
    connectome_dir : str
        Path to individual subject connectome files.
    output_dir : str
        Path to output directory.

    """
    start = time.time()
    # Check that mask and chunk_idx_mask are in same space
    same_size = np.sum(image.get_data(mask)==(image.get_data(chunk_idx_mask)>0))
    assert same_size==np.sum(image.get_data(mask)), f"Binary mask and chunk idx"
    " mask do not match! Binary mask has size {np.sum(image.get_data(mask))}"
    " and chunk idx mask has size {np.sum(image.get_data(chunk_idx_mask)>0)}"
    masker = tools.NiftiMasker(mask)

    # Get list of connectome files
    connectome_files_norms = natsorted(glob(os.path.join(connectome_dir,
                                                         "*_norms.npy")))
    connectome_files = [(glob(f.split('_norms')[0] + ".npy")[0], f) for f in connectome_files_norms]
    if (len(connectome_files) == 0):
        raise ValueError("No connectome files found")
    else:
        print(f"Found {len(connectome_files)} connectome subjects!")

    # Get binary chunk roi in brain-space
    chunk_roi = masker.transform(image.math_img(f"img == {chunk_idx}",
                                                img = chunk_idx_mask))

    # Initialize Welford Maps
    count = 0
    mean = np.zeros((292019, 3209), dtype=np.float32)
    M2 = np.zeros((292019, 3209), dtype=np.float32)
    # For each connectome subject
    for connectome_file in tqdm(connectome_files):
        # Calculate Fz maps from a connectome subject
        fz_welford = make_fz_maps(connectome_file, chunk_roi)
        # Update Welford Maps
        count, mean, M2 = welford_update_map(count, mean, M2, fz_welford)
    # Finalize Welford Maps and output to file
    print("Outputting Chunk to disk...")
    make_stat_maps(count, mean, M2, output_dir, chunk_idx)
    end = time.time()
    print(f"Elapsed time: {end-start} seconds")