import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.custom_typing import SmallNORBData, SmallNORBKey
from utils.smallnorb_dataprovider import *

class SmallNORBDataset(Dataset):
    def __init__(self, data_folder, split="train"):
        self.data_folder = data_folder
        self.split = split
        self.initialize()

    def initialize(self):
        self.loaded_data = load_dataset(self.split) #dict[('train'/'test', 'cat'/'dat'/'info')]
        self.imgs_pair_attributes = generate_imgs_pair_attributes(
            combinations=100_000,
            split=self.split
        )
        self.smallnorb_data_lookup = create_smallnorb_seeker(self.loaded_data, self.split)

    def __getitem__(self, index):
        [left_attributes, right_attributes] = self.imgs_pair_attributes[index]

        left_img_index = self.smallnorb_data_lookup[SmallNORBKey(*left_attributes)]
        right_img_index = self.smallnorb_data_lookup[SmallNORBKey(*right_attributes)]      

        image_left = (
            self.loaded_data[(self.split, "dat")][left_img_index][0]
        ).unsqueeze(0) / 255.0
        image_right = (
            self.loaded_data[(self.split, "dat")][right_img_index][0]
        ).unsqueeze(0) / 255.0
        
        left_cat = torch.tensor(left_attributes[0], dtype=torch.int32)
        right_cat = torch.tensor(right_attributes[0], dtype=torch.int32)
        
        # Separate info into different components
        
        #left_img_info_instance = torch.tensor(left_attributes[1], dtype=torch.int32)
        left_elevation = torch.tensor(left_attributes[2], dtype=torch.int32)
        left_lighting = torch.tensor(left_attributes[4], dtype=torch.int32)

        #right_img_info_instance = torch.tensor(right_attributes[1], dtype=torch.int32)
        right_elevation = torch.tensor(right_attributes[2], dtype=torch.int32)
        right_lightning = torch.tensor(right_attributes[4], dtype=torch.int32)

        return SmallNORBData(
            left_img = image_left,
            right_img = image_right,
            left_cat = left_cat,
            right_cat = right_cat,
            elevation = left_elevation,
            lightning =  right_lightning
        )

    def __len__(self):
        return len(self.imgs_pair_attributes)

if __name__ == "__main__":
    dataset = SmallNORBDataset(data_folder="path/to/your/data", split="train")
