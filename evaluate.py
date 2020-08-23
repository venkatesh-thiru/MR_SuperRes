import torch
from skimage.metrics import structural_similarity as ssim
from UNetModel import Unet
from MRIBrainData import BrainDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm




Test_dataset  = BrainDataset ("IXI-T1",Validation= True)
TestLoader = DataLoader(Test_dataset,batch_size=1)

Training_dataset = BrainDataset("IXI-T1",Validation=False)
TrainLoader = DataLoader(Training_dataset,batch_size=1)

checkpoint = torch.load("Models/baseUnet3D_OriginalSize_ADAMOptim_20Epochs_BS2_GlorotWeights")
unet3D = Unet(in_channel=1,out_channel=1,filters=2)
unet3D.load_state_dict(checkpoint["model_state_dict"])
unet3D.to("cuda")


unet3D.eval()
ssim_overall = 0

for idx,(original,reduced) in enumerate(tqdm(TestLoader)):
    original = original.to("cuda")
    reduced = reduced.to("cuda")
    pred = unet3D(reduced)
    original, reduced, pred = torch.squeeze(original), torch.squeeze(reduced), torch.squeeze(pred)
    original,reduced = original.detach().cpu().numpy().T,reduced.detach().cpu().numpy().T
    index = ssim(original,reduced,data_range=(np.max(original)-np.min(original)))
    print(index)
    ssim_overall += index


print("Over all ssim = {}".format(ssim_overall/len(Test_dataset)))