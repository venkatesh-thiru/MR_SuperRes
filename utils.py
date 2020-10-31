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

def write_data(compressed,scan,new_affine):
    target_dir = "IXI-T1/Compressed_3x3x1.2"
    compressed_img = nib.Nifti1Image(compressed.astype(np.int16),new_affine)
    compressed_img.to_filename(os.path.join(target_dir, scan) + ".nii.gz")


def prepare_datasets():
    data_dir = "IXI-T1/Actual_Images"
    scans = os.listdir(data_dir)
    for scan in tqdm(scans):
        try:
            myimg = Image(os.path.join(data_dir, scan))
            compressed,new_affine = FFT_compression(myimg,[3,3,1.2])
            compressed = compressed.astype(np.int16)
            write_data(compressed,scan,new_affine)
        except:
            print("{} invalid input".format(scan))
    print("write finished")
