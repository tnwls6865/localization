import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class Cub2011(Dataset):
    def __init__(self, root='../datas/', train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ', names=['img_id', 'filepath'])
    
        self.image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'])
        self.train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])
        self.bbox = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'bounding_boxes.txt'), sep=' ', names=['img_id', 'x', 'y', 'w', 'h'])

        data = self.images.merge(self.image_class_labels, on='img_id')
        data = data.merge(self.train_test_split, on='img_id')
        self.all_data = data.merge(self.bbox, on='img_id')

        if self.train:
            self.data = self.all_data[self.all_data.is_training_img == 1]
        else:
            self.data = self.all_data[self.all_data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data['filepath'].iloc[idx]
        b_x = self.data['x'].iloc[idx]
        b_y = self.data['y'].iloc[idx]
        b_w = self.data['w'].iloc[idx]
        b_h= self.data['h'].iloc[idx]
        target = self.data['target'].iloc[idx]
        
        image = Image.open(os.path.join(self.root, 'CUB_200_2011/images/', file_path)).convert('RGB')
        img_w = image.size[0]
        img_h = image.size[1]
        
        if self.train == True:
            bbox_x = int((224/img_w)*b_x)
            bbox_y = int((224/img_h)*b_y)
            bbox_w = int((224/img_w)*b_w)
            bbox_h = int((224/img_h)*b_h)
            
        target = target - 1  # Targets start at 1 by default, so shift to 0
        bbox = [bbox_x, bbox_y, bbox_w, bbox_h]
        bbox = torch.LongTensor(bbox)

        if self.transform is not None:
            image = self.transform(image)
        
        target = torch.tensor(target)

        return image, target, bbox