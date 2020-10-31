from pathlib import Path
import torchio as tio
from torchio import AFFINE,DATA
from tqdm import tqdm
import glob
import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
from UNetModel import Unet
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchio.transforms import Compose,ZNormalization,RescaleIntensity
import statistics
import random
import DenseNetModel
import pytorch_ssim
import multiprocessing




torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
device = "cuda" if torch.cuda.is_available() else "cpu"

def init_weights(m):
    if type(m) == nn.Conv3d:
        torch.nn.init.xavier_uniform_(m.weight)


# unet3D = Unet(in_channel=1,out_channel=1,filters=8)
# unet3D.apply(init_weights)
# unet3D = unet3D.to(device)
model = DenseNetModel.DenseNet(num_init_features=4,growth_rate=6,block_config=(6,6,6))
model.apply(init_weights)
model = model.to(device)


learning_rate = 0.001
opt = optim.Adam(model.parameters(),lr=learning_rate)
loss_fn = pytorch_ssim.SSIM3D(window_size=11)

Epochs = 50
training_batch_size = 12
validation_batch_size = 6
patch_size = 32
samples_per_volume = 40
max_queue_length = 80


training_name = "denseNet3D_torchIO_patch_{}_samples_{}_ADAMOptim_{}Epochs_BS{}_GlorotWeights_SSIM_2810".format(patch_size,samples_per_volume,Epochs,training_batch_size)
train_writer = SummaryWriter(os.path.join("runs",training_name+"_training"))
validation_writer = SummaryWriter(os.path.join("runs",training_name+"_validation"))


ground_truths = Path("IXI-T1/Actual_Images")
compressed_images = Path("IXI-T1/Compressed_3x3x1.2")
ground_paths = sorted(ground_truths.glob('*.nii.gz'))
compressed_paths = sorted(compressed_images.glob('*.nii.gz'))
subjects = []

for gt,comp in zip(ground_paths,compressed_paths):
    subject = tio.Subject(
                ground_truth = tio.ScalarImage(gt),
                compressed = tio.ScalarImage(comp),
                )
    subjects.append(subject)


training_subjects = subjects[:int(len(subjects)*0.85)]
validation_subjects = subjects[int(len(subjects)*0.85)+1:int(len(subjects)*0.9)]
test_subjects = subjects[int(len(subjects)*0.9)+1:]


training_transform = Compose([RescaleIntensity((0,1))])
validation_transform = Compose([RescaleIntensity((0,1))])
test_transform = Compose([RescaleIntensity((0,1))])


training_dataset = tio.SubjectsDataset(training_subjects,transform=training_transform)
validation_dataset = tio.SubjectsDataset(validation_subjects,transform=validation_transform)
test_dataset = tio.SubjectsDataset(test_subjects,transform=test_transform)

'''Patching'''

patches_training_set = tio.Queue(
    subjects_dataset=training_dataset,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=tio.sampler.UniformSampler(patch_size),
    # shuffle_subjects=True,
    # shuffle_patches=True,
)

patches_validation_set = tio.Queue(
    subjects_dataset=validation_dataset,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume*2,
    sampler=tio.sampler.UniformSampler(patch_size),
    # shuffle_subjects=False,
    # shuffle_patches=False,
)

training_loader = torch.utils.data.DataLoader(
    patches_training_set, batch_size=training_batch_size)
validation_loader = torch.utils.data.DataLoader(
    patches_validation_set, batch_size=validation_batch_size)


def write_image(slice_list,epoch):
    print("writing image.......")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Epoch {}".format(epoch))
    ax[0].imshow(slice_list[0], interpolation='nearest', origin="lower", cmap="gray")
    ax[0].set_title("Original")
    ax[0].set_axis_off()
    ax[1].imshow(slice_list[1], interpolation='nearest', origin="lower", cmap="gray")
    ax[1].set_title("Reduced")
    ax[1].set_axis_off()
    ax[2].imshow(slice_list[2], interpolation='nearest', origin="lower", cmap="gray")
    ax[2].set_title("Predicted")
    ax[2].set_axis_off()
    train_writer.add_figure("comparison", fig, epoch)

def test_network(epoch):
    sample = random.choice(test_dataset)
    input_tensor = sample.compressed.data[0]
    patch_size = 64,64,64
    patch_overlap = 4,4,4
    grid_sampler = tio.inference.GridSampler(sample,patch_size,patch_overlap)
    patch_loader = torch.utils.data.DataLoader(grid_sampler,int(validation_batch_size/4))
    aggregator = tio.inference.GridAggregator(grid_sampler,overlap_mode="average")
    with torch.no_grad():
        for batch in patch_loader:
            inputs = batch["compressed"][DATA].to(device)
            logits = model(inputs)
            location = batch[tio.LOCATION]
            aggregator.add_batch(logits,location)
    result = aggregator.get_output_tensor()
    original, compressed = torch.squeeze(sample.ground_truth["data"]), torch.squeeze(sample.compressed["data"])
    result = torch.squeeze(result)
    original,compressed,result = original.detach().cpu().numpy(),compressed.detach().cpu().numpy(),result.detach().cpu().numpy()
    slice_original = (original[:, :, int(original.shape[2] / 2)])
    slice_compressed = (compressed[:, :, int(compressed.shape[2] / 2)])
    slice_result = (result[:, :, int(result.shape[2] / 2)])
    slice_list = [slice_original.T,slice_compressed.T,slice_result.T]
    write_image(slice_list,epoch)


def validation_loop():
    overall_validation_loss = []
    for batch in tqdm(validation_loader):
        batch_actual = batch["ground_truth"][DATA].to(device)
        batch_compressed = batch["compressed"][DATA].to(device)
        with torch.no_grad():
            logit = model(batch_compressed)
        loss = loss_fn(logit, batch_actual)
        overall_validation_loss.append(loss.item())

    validation_loss = statistics.mean(overall_validation_loss)
    return validation_loss



steps = 0
old_validation_loss = 0
for epoch in range(Epochs):
    overall_training_loss = []
    for batch in tqdm(training_loader):
        steps += 1
        batch_actual = batch["ground_truth"][DATA].to(device)
        batch_compressed = batch["compressed"][DATA].to(device)
        logit = model(batch_compressed)
        loss = -loss_fn(logit,batch_actual)
        opt.zero_grad()
        loss.backward()
        opt.step()
        overall_training_loss.append(-loss.item())
        # if not steps % 50:
        #     validation_loss = validation_loop()
        #     training_loss = statistics.mean(overall_training_loss)
        #     train_writer.add_scalar("training_loss", training_loss, steps)
        #     validation_writer.add_scalar("validation_loss", validation_loss, steps)
    validation_loss = validation_loop()
    test_network(epoch)
    training_loss = statistics.mean(overall_training_loss)
    print("EPOCH {} : training_loss ===> {};validation_loss ===> {}".format(epoch,training_loss,validation_loss))
    if (old_validation_loss == 0) or (old_validation_loss<validation_loss):
        torch.save(model,os.path.join("Models",training_name+".pth"))
        old_validation_loss= validation_loss
        print("model_saved")
