# this script calculates mean and std of the dataset

import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms





class Melanoma_loader(data.Dataset):
    def __init__(self, root, ann_path, transform=None, target_transform=None):

        self.data_path = root
        self.ann_path = ann_path
        self.transform = transform
        self.target_transform = target_transform
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
        filename = idb[0]
        Class = int(idb[5])
        images = self._load_image(self.data_path + '/' + str(filename) + '.jpg')
        if self.transform is not None:
            images = self.transform(images)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        return images, 1, Class, 1, 1, 1, 1, 1

    def __len__(self):
        return len(self.database)
    

t = transforms.Compose([transforms.ToTensor(),
                        transforms.Resize((224,224)),
                        ]) 





train_data = Melanoma_loader(root = '/home/kpusteln/melanoma/train/train', 
                        ann_path = '/home/kpusteln/melanoma/train_set.csv', 
                        transform=t)




batch_size = 1
loader = torch.utils.data.DataLoader(
      train_data,
      batch_size = batch_size,
      shuffle=False,
      num_workers = 40
  )


mean = 0.
meansq = 0.
print('Calculating mean and std...')
for i, data in enumerate(loader):
    data = data[0]
    #data = data.cuda()
    mean += data.sum()
    meansq += (data**2).sum()
    if i % 10 == 0:
        print(f'Done: {int((i/(len(loader.dataset)/batch_size))*100)}%')
        with open("status_mean_std_cpu.txt", "w") as text_file:
            text_file.write(f'Done: {int((i/(len(loader.dataset)/batch_size))*100)}%')
with open("status_mean_std_cpu.txt", "w") as text_file:
            text_file.write(f'Done: {int((i/(len(loader.dataset)/batch_size))*100)}%')

mean = mean/(len(train_data)*batch_size*224*224)
meansq = meansq/(len(train_data)*batch_size*224*224)
std = torch.sqrt(meansq - mean**2)
mean = mean.cpu().detach().numpy()
std = std.cpu().detach().numpy()
print('Mean: ', mean)
print('Std: ', std)
np.save('mean_std_melanoma', [mean, std])

