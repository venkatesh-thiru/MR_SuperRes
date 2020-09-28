from torch.utils.data import  DataLoader,Dataset
import numpy as np
import os
from nilearn import image
import torch
from tqdm import tqdm


class BrainDataset(Dataset):
    def __init__(self,directory,transform = None,type="train"):
        self.directory = directory
        self.transform = transform
        self.ground_truth_dir = os.path.join(directory,"ground_truth")
        self.compressed_dir = os.path.join(directory,"compressed")
        self.ground_truth = os.listdir(self.ground_truth_dir)
        self.compressed = os.listdir(self.compressed_dir)
        if type == "test":
            self.ground_truth = self.ground_truth[int(len(self.ground_truth)*0.9)+1 : ]
            self.compressed = self.compressed[int(len(self.compressed) * 0.9) + 1:]
        elif type == "train":
            self.ground_truth = self.ground_truth[ : int(len(self.ground_truth)*0.85)]
            self.compressed = self.compressed[: int(len(self.compressed) * 0.85)]
        else:
            self.ground_truth = self.ground_truth[int(len(self.ground_truth)*0.85)+1:int(len(self.ground_truth)*0.9)+1]
            self.compressed = self.compressed[int(len(self.compressed)*0.85)+1:int(len(self.compressed)*0.9)+1]

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        ground_truth_fname = os.path.join(self.ground_truth_dir,self.ground_truth[item])
        compressed_fname = os.path.join(self.compressed_dir, self.ground_truth[item])

        training_data = []
        try:
            ground_truth_img = image.load_img(ground_truth_fname, dtype=("float32"))
            compressed_img = image.load_img(compressed_fname, dtype=("float32"))
            gt_np = ground_truth_img.get_fdata()
            gt_np = np.expand_dims((gt_np / np.max(gt_np) if np.max(gt_np)!= 0 else gt_np), 0).astype("float32")
            comp_np = compressed_img.get_fdata()
            comp_np = np.expand_dims((comp_np / np.max(comp_np) if np.max(comp_np)!= 0 else comp_np), 0).astype("float32")
            training_data.append(gt_np)
            training_data.append(comp_np)
            if self.transform:
                training_data = self.transform(training_data)

            return training_data
        except:
            print("invalid data")




if __name__ =="__main__":
    dataset = BrainDataset("IXI-T1")
    loader = DataLoader(dataset,batch_size=1)
    for i in tqdm(range(0,len(dataset))):
        gt,comp = dataset[i]
        max_gt,max_comp = np.max(gt),np.max(comp)
        if np.isnan(max_gt) or np.isnan(max_comp):
            print(i)
