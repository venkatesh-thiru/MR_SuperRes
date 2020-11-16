import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from nilearn import image
from tqdm import tqdm
import scipy
from scipy.ndimage import zoom
from fsl.data.image import Image
from fsl.utils.image import resample
from pathlib import Path
import torchio as tio
import glob
from sklearn.model_selection import train_test_split


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(len(slices), len(slices[0]))
    fig.set_size_inches(15,7.5)
    for row in range(len(slices)):
        for column in range(len(slices[0])):
            axes[row][column].imshow(slices[row][column].T, cmap="gray", origin="lower")


def plot_images(inp_np,res_np):
    slice_0 = inp_np[int(inp_np.shape[0]/2), :, :]
    slice_1 = inp_np[:, int(inp_np.shape[1]/2), :]
    slice_2 = inp_np[:, :, int(inp_np.shape[2]/2)]
    nslice_0 = res_np[int(res_np.shape[0]/2), :, :]
    nslice_1 = res_np[:, int(res_np.shape[1]/2), :]
    nslice_2 = res_np[:, :, int(res_np.shape[2]/2)]
    show_slices([[slice_0, slice_1, slice_2],
                 [nslice_0, nslice_1, nslice_2]])

def make_chunks(image,target_size = 64):
    B,H,S = image.shape
    chunks = []
    for x in range(0,B,target_size):
        for y in range(0,H,target_size):
            for z in range(0,S,target_size):
                chunks.append(image[x:x+target_size,y:y+target_size,z:z+target_size])
    for i in range(2,len(chunks),3):
        patch = (chunks[i-1][:,:,32:])
        chunks[i] = np.concatenate([patch,chunks[i]],axis = 2)
    return chunks

def unmake_chunks(chunks):
    slices = []
    widths = []
    for i in range(2,len(chunks),3):
        slices.append(np.concatenate([chunks[i-2],chunks[i-1],chunks[i][:,:,32:]],axis = 2))
    for i in range(3,len(slices),4):
        widths.append(np.concatenate([slices[i-3],slices[i-2],slices[i-1],slices[i]],axis = 1))
    image = np.concatenate(widths,axis = 0)
    return image

def FFT_compression(myimg,scale_factor,ifzoom=True):
    resample_img = resample.resampleToPixdims(myimg, scale_factor)
    new_affine = resample_img[1]
    if ifzoom:
        zoom_factor = np.array(myimg.shape) / np.array(resample_img[0].shape)
        zoomed = zoom(resample_img[0], zoom_factor, mode='nearest')
        return zoomed,new_affine
    else:
        return resample_img[0],new_affine

def write_data(compressed,scan,new_affine,scale_factor):
    target_dir = "IXI-T1/Compressed_{}x{}x{}".format(scale_factor[0],scale_factor[1],scale_factor[2])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    compressed_img = nib.Nifti1Image(compressed.astype(np.int16),new_affine)
    compressed_img.to_filename(os.path.join(target_dir, scan))


def prepare_datasets(scale_factor):
    data_dir = "IXI-T1/Actual_Images"
    scans = os.listdir(data_dir)
    for scan in tqdm(scans):
        try:
            myimg = Image(os.path.join(data_dir, scan))
            compressed,new_affine = FFT_compression(myimg,scale_factor)
            compressed = compressed.astype(np.int16)
            write_data(compressed,scan,new_affine,scale_factor)
        except:
            print("{} invalid input".format(scan))
    print("write finished")


def train_test_val_split():
    ground_truths = Path("IXI-T1/Actual_Images")
    ground_paths = sorted(ground_truths.glob('*.nii.gz'))
    compressed_dirs = [sorted(Path((os.path.join("IXI-T1",comp))).glob('*.nii.gz')) for comp in os.listdir("IXI-T1") if "Compressed" in comp]
    # compressed_images = Path("IXI-T1/Compressed_3x3x1.2")
    # compressed_paths = sorted(compressed_images.glob('*.nii.gz'))
    training_subjects = []
    test_subjects = []
    validation_subjects = []
    for compressed_paths in compressed_dirs:
        subjects = []
        for gt,comp in zip(ground_paths,compressed_paths):
            subject = tio.Subject(
                    ground_truth = tio.ScalarImage(gt),
                    compressed = tio.ScalarImage(comp),
                    )
            subjects.append(subject)
        train_split,test_split = train_test_split(subjects,test_size=0.3)
        test_split,validation_split = train_test_split(test_split,test_size=0.2)
        training_subjects += train_split
        validation_subjects += validation_split
        test_subjects += test_split
    return training_subjects,test_subjects,validation_subjects