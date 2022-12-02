import os
import json
import torch.utils.data as data
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class Fetal_frame(data.Dataset):
    def __init__(self, root, ann_path, transform=None, target_transform=None):

        self.data_path = root
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform
        # id & label: https://github.com/google-research/big_transfer/issues/7
        # total: 21843; only 21841 class have images: map 21841->9205; 21842->15027
        self.database = pd.read_csv(self.ann_path)

    def _load_image(self, path):
        try:
            im = Image.open(path)
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        idb = self.database.iloc[index]
        #video, index, Class, measures, abdomen_ps, frames_n, min, max, measures_normalized
        # images
        # video, index, Class, AC, abdomen_ps, frames_n
        images = self._load_image(self.data_path  + idb[1] + '.png')
        if self.transform is not None:
            images = self.transform(images)

        # target
        # Max measure scaled: head = 214.14944514917548
        # Max measure scaled: abdomen = 215.22020313394543
        # Max measure scaled: femur = 72.1250937626219
        video = idb[0]
        index = idb[1]
        Class = idb[2]
        measure = idb[3]
        ps = idb[4]
        frames_n = idb[5]
        measure_scaled = measure/ps
        max_measure = 214.14944514917548 if Class == 2 else 215.22020313394543 if Class == 4 else 72.1250937626219
        measure_normalized = measure_scaled/max_measure
        if self.target_transform is not None:
            target = self.target_transform(target)
        #save_image(images[0], '/data/kpusteln/examples' + str(index) + '.png')
        return images, index, Class, video, measure, ps, frames_n, measure_normalized

    def __len__(self):
        return len(self.database)



class Fetal_vid(data.Dataset):
    def __init__(self, videos_path, root, ann_path, transform=None, target_transform=None):
        
        
        self.videos = pd.read_csv(videos_path)
        self.data_path = root
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform
        self.database = pd.read_csv(self.ann_path)
        
    def _load_image(self, path):
        try:
            im = Image.open(path)
            im.convert('RGB')
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, frame_positions, labels)
        """
        images = list() # list of images
        vid = self.videos.iloc[index][0]
        vid_len = self.database.query('video == @vid')['frames_n'].iloc[0]
        ps = self.database.query('video == @vid')['abdomen_ps'].iloc[0]
        measure = self.database.query('video == @vid')['AC'].iloc[0]
        measure_scaled = measure/ps
        # Max measure scaled: head = 214.14944514917548
        max_measure = 214.14944514917548
        measure_normalized = measure_scaled/max_measure
        #print(self.database.query(vid[0]))
        #index, #class, #video, #frames_n, abdomen_ps, AC
        # images
        for frame in range(vid_len):
            #print(frame)
            image = self._load_image(self.data_path  + vid + f'_{frame+1}' + '.png')
            # transform image              
            if self.transform is not None:
                image = self.transform(image)
            images.append(image) # append image to list
            idx = f'{vid}_' + f'{frame+1}'

            
            

        # target
        frames_position = [i+1 for i in range(vid_len)]
        #save_image(images[0], '/data/kpusteln/examples' + str(index) + '.png')
        images = torch.stack(images)
        images = images.permute(1, 0, 2, 3)
        frames_position = torch.tensor(frames_position)
        Class = 1
        video = 1
        indexes = 1
        return images, indexes, Class, video, measure, ps, frames_position, measure_normalized

    def __len__(self):
        return len(self.videos)