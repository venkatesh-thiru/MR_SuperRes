from torch.utils.data import  DataLoader,Dataset
import numpy as np
import os
from nilearn import image
import torch



class BrainDataset(Dataset):
    def __init__(self,directory,transform = None,Validation = False):
        self.directory = directory
        self.transform = transform
        self.scans = os.listdir(directory)
        if Validation:
            self.scans = self.scans[int(len(self.scans)*0.9)+1 : ]
        else:
            self.scans = self.scans[ : int(len(self.scans)*0.9)]

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        scan_file = os.path.join(self.directory,self.scans[item])
        inp = image.load_img(scan_file,dtype=("float32"))
        training_data = []
        try:
            old_affine = inp.affine
            new_affine = old_affine * 2
            inp = image.resample_img(inp, target_affine=new_affine, target_shape=[128, 128, 128],
                                     interpolation="continuous")
            update1 = inp.affine * 2
            red = image.resample_img(inp, target_affine=update1, interpolation='nearest')
            update2 = red.affine / 2
            red = image.resample_img(red, target_affine=update2, target_shape=[128, 128, 128], interpolation="nearest")

            inp_np = inp.get_fdata()
            inp_np = np.expand_dims((inp_np/np.max(inp_np)),0).astype("float32")
            res_np = red.get_fdata()
            res_np = np.expand_dims(res_np/np.max(res_np),0).astype("float32")
            training_data.append(inp_np)
            training_data.append(res_np)
            if self.transform:
                training_data = self.transform(training_data)

            return training_data
        except:
            print("invalid data")




dataset = BrainDataset("IXI-T1")
loader = DataLoader(dataset)

