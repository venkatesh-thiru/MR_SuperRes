from torch.utils.data import Dataset,DataLoader
import torch
from UNetModel import Unet
from MRIBrainData import BrainDataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib
from utils import make_chunks,preprocess_data,FFT_compression,unmake_chunks

'''
Parameters to avoid memory overflow error
'''
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
restore = False
'''
Checks for GPU and decides the device for processing
'''
device = "cuda" if torch.cuda.is_available() else "cpu"

'''
Initialize model
'''
def init_weights(m):
    if type(m) == nn.Conv3d:
        torch.nn.init.xavier_uniform_(m.weight)
    # m.bias.data.fill_(0.01)

unet3D = Unet(in_channel=1,out_channel=1,filters=8)
unet3D.apply(init_weights)
unet3D = unet3D.to(device)
# unet3D = torch.load("Models/chunked_test_training")
unet3D = unet3D.to(device)

'''
General Hyper parameters
'''
learning_rate = 0.001
Batch_Size = 20
opt = optim.Adam(unet3D.parameters(),lr=learning_rate)
loss_fn = nn.MSELoss()
Epochs = 10

'''
Initializes a Tensorboard summary writer to keep track of the training process
'''

training_name = "baseUnet3D_chunkedSize_ADAMOptim_{}Epochs_BS{}_GlorotWeights_MSELoss_2709".format(Epochs,Batch_Size)
writer = SummaryWriter(os.path.join("runs",training_name))

'''
Training data, calls the BrainDataset(Custom Dataset Class)
'''
training_dataset = BrainDataset("IXI-T1",type = "train")
validation_dataset = BrainDataset("IXI-T1",type = "validation")
TrainLoader = DataLoader(training_dataset,batch_size=Batch_Size)
ValidationLoader = DataLoader(validation_dataset,batch_size=Batch_Size)
'''
Testing data, calls the BrainDataset(Custom Dataset Class)
'''
Test_dataset  = BrainDataset ("IXI-T1",type="test")
TestLoader = DataLoader(Test_dataset,batch_size=1)


'''
Generated image from the writeImage method is the arranged and converted into a matplotlib figure and written on tensorboard
'''
def show(img,epoch,step):
    fig,ax = plt.subplots(1,3,figsize = (15,5))
    fig.suptitle("Epoch {}".format(epoch+1))
    ax[0].imshow(img[0],interpolation='nearest',origin="lower",cmap="gray")
    ax[0].set_title("Original")
    ax[0].set_axis_off()
    ax[1].imshow(img[1], interpolation='nearest', origin="lower", cmap="gray")
    ax[1].set_title("Reduced")
    ax[1].set_axis_off()
    ax[2].imshow(img[2], interpolation='nearest', origin="lower", cmap="gray")
    ax[2].set_title("Predicted")
    ax[2].set_axis_off()
    writer.add_figure("comparison",fig,step)


'''
changes the model to the evaluation mode and feed forwards Test data into the model, Generates a resultant image
'''

def writeImage(epoch,step):
    unet3D.eval()
    print("writing image.............")
    datadir = "IXI-T1/Actual_Images"
    scans = os.listdir(datadir)
    img = nib.load(os.path.join(datadir,scans[-1]))
    img = preprocess_data(img)
    original = img.get_fdata()
    reduced = FFT_compression(original)
    reduced_chunks = make_chunks(reduced)
    predicted = []
    for chunk in reduced_chunks:
        chunk = torch.from_numpy(np.expand_dims(chunk,0).astype("float32")).to(device)
        with torch.no_grad():
            predicted_chunk = unet3D(torch.unsqueeze(chunk,0))
        predicted.append(torch.squeeze(predicted_chunk).detach().cpu().numpy())
    pred = unmake_chunks(predicted)
    slice_original = (original[:, :, int(original.shape[2] / 2)])
    slice_reduced  = (reduced[:,: ,int(reduced.shape[2]/2)])
    slice_pred = (pred[:,:,int(pred.shape[2]/2)])
    slices_list = [slice_original.T,slice_reduced.T,slice_pred.T]
    show(slices_list,epoch,step)
    unet3D.train()

def validate(step):
    unet3D.eval()
    loss = 0
    torch.cuda.empty_cache()
    for idx,(original,reduced) in enumerate(tqdm(ValidationLoader)):
        original= original.to(device)
        reduced = reduced.to(device)
        with torch.no_grad():
            logit = unet3D(reduced)
        curr_loss = loss_fn(logit, original)
        loss += curr_loss.item()
    unet3D.train()
    return loss/len(validation_dataset)

'''
The Training loop begins here
'''

step = 0
print("Beginning Training............")
Batch_count = len(TrainLoader)

'''
Restoring checkpoint since the training has been interupted
'''
if restore:
    checkpoint = torch.load("Models/baseUnet3D_OriginalSize_ADAMOptim_10Epochs_BS1_GlorotWeights_L1Loss")
    unet3D.load_state_dict(checkpoint["model_state_dict"])


for epoch in range(Epochs):
    torch.cuda.empty_cache()
    overall_loss = 0
    for idx,(original,reduced) in enumerate(tqdm(TrainLoader)):
        step += 1
        original = original.to(device)
        reduced = reduced.to(device)
        logit = unet3D(reduced)
        loss = loss_fn(logit,original)
        opt.zero_grad()
        loss.backward()
        opt.step()
        overall_loss += loss.item()
        writer.add_scalar("training_loss",loss.item(),step)
        if not step % 200:
            validation_loss = validate(step)
            writer.add_scalar("validation loss", validation_loss, step)
    validation_loss = validate(step)
    writer.add_scalar("validation loss",validation_loss, step)
    writeImage(epoch, step)
    print("EPOCH {} training Loss ===> {}|| validation loss ===>{}".format(epoch+1,overall_loss/Batch_count,validation_loss))
    torch.save({
        'epoch': epoch,
        'model_state_dict': unet3D.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': overall_loss/Batch_count,
    }, os.path.join("Models",training_name))

'''
saves the model after training
'''
torch.save({
    'epoch': epoch,
    'model_state_dict': unet3D.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
}, os.path.join("Models", training_name))

