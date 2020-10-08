import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from nilearn import image
from tqdm import tqdm
import scipy
from scipy.ndimage import zoom

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

def FFT_compression(inp_np):
    resample = scipy.signal.resample(inp_np, 64, axis=0)
    resample = scipy.signal.resample(resample, 64, axis=1)
    zoom_factor = np.array(inp_np.shape) / resample.shape
    zoomed = zoom(resample, zoom_factor, mode='nearest')
    return zoomed

def write_data(chunks,scan,type):
    datadir = os.path.join("IXI-T1",type)
    scan = scan.split('.')[0]
    for i in range(0,len(chunks)):
        file_name = scan + "_" + str(i)
        file_name = os.path.join(datadir,file_name)
        img = nib.Nifti1Image(chunks[i],np.eye(4))
        img.to_filename(os.path.join(str(file_name)+".nii.gz"))

def preprocess_data(img):
    affine = img.affine
    img = image.resample_img(img, target_affine=affine, target_shape=[256, 256, 160], interpolation="nearest")
    return img

def prepare_datasets():
    data_dir = "IXI-T1/Actual_Images"
    scans = os.listdir(data_dir)
    for scan in tqdm(scans):
        try:
            fname = os.path.join(data_dir, scan)
            inp = nib.load(fname)
            old_affine = inp.affine
            inp = image.resample_img(inp, target_affine=old_affine, target_shape=[256, 256, 160], interpolation="nearest")
            inp_np = inp.get_fdata()
            inp_np = inp_np.astype(np.int16)
            chunks = make_chunks(image = inp_np)
            write_data(chunks,scan,type = "ground_truth")
            compressed = FFT_compression(inp_np)
            compressed = compressed.astype(np.int16)
            chunks = make_chunks(compressed)
            write_data(chunks,scan,type="compressed")
        except:
            print("{} invalid input".format(scan))
    print("write finished")