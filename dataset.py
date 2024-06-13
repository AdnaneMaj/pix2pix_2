from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class PairedTransform:
    def __init__(self,base_transform):
        self.base_transform = base_transform

    def __call__(self, seg_img, rgb_img):
        if random.random() > 0.5:
            seg_img = TF.hflip(seg_img)
            rgb_img = TF.hflip(rgb_img)

        if random.random() > 0.5:
            seg_img = TF.vflip(seg_img)
            rgb_img = TF.vflip(rgb_img)

        angle = random.randint(-45,45)
        seg_img = TF.rotate(seg_img,angle)
        rgb_img = TF.rotate(rgb_img,angle)

        if self.base_transform:
            seg_img = self.base_transform(seg_img)
            rgb_img = self.base_transform(rgb_img)

        return seg_img,rgb_img


base_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
])

paired_transform = PairedTransform(base_transform)
        
default_path = "/kaggle/input/drive-22/Drive_data/"
class Costum(Dataset):
    def __init__(self,root_dir=default_path,paired_transform=paired_transform,val=False):
        self.root_dir = root_dir
        self.val = val
        self.paired_transform = paired_transform
        self.list_files_seg = os.listdir(root_dir+"seg/seg/") if not val else os.listdir(root_dir+"seg_val/seg/")
        self.list_files_rgb = os.listdir(root_dir+"rgb/images/") if not val else os.listdir(root_dir+"rgb_val/images/")

    def __len__(self):
        return len(self.list_files_seg)*100
    
    def __getitem__(self,idx):
        idx = idx%len(self.list_files_seg)
        input_img = self.list_files_seg[idx]
        target_img = self.list_files_rgb[idx]
        input_img = Image.open(self.root_dir+"seg/seg/"+input_img).convert("L") if not self.val else Image.open(self.root_dir+"seg_val/seg/"+input_img).convert("L")
        target_img = Image.open(self.root_dir+"rgb/images/"+target_img).convert("RGB") if not self.val else Image.open(self.root_dir+"rgb_val/images/"+target_img).convert("RGB")

        if self.paired_transform:
            input_img,target_img = self.paired_transform(input_img,target_img)
        return input_img,target_img
    
if __name__=="__main__":
    dataset = Costum()
    print(len(dataset))
    input_img,target_img = dataset[0]
    print(input_img.shape,target_img.shape)
