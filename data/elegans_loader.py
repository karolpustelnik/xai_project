import os
import json
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
import torchvision.transforms as transforms
from torchvision.utils import save_image

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class Elegans(data.Dataset):
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

        # images
        images = self._load_image(self.data_path  + idb[0]).convert('RGB')
        if self.transform is not None:
            images = self.transform(images)

        # target
        target = 1
        score = idb[1]
        if self.target_transform is not None:
            target = self.target_transform(target)
        #save_image(images[0], '/data/kpusteln/examples' + str(index) + '.png')
        return images, target, score

    def __len__(self):
        return len(self.database)
