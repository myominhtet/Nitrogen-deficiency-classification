TEST = "C://Users//Myo Min Htet//Downloads//Compressed//NitrogenDeficiencyImage//Test"
PATH = 'C:/Users/Myo Min Htet/OneDrive/Desktop/New folder/youngsavage/output/'
OUTPUT_DIR = "C:/Users/Myo Min Htet/OneDrive/Desktop/New folder/youngsavage/output/"
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transform
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import cv2
from tqdm import tqdm
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inform(dir):

    folders = os.listdir(dir)
    count = 0
    total = 0
    for filenames in folders:
        folders = os.path.join(dir + "//" + filenames)  
        count += 1
        images = os.listdir(folders)
        print(f"{filenames} has {len(images)} samples")
        total += len(images)
   
    print(f"There are {count} LCC")
    print(f"There are {total} images")
inform(TEST)
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str('seed')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print('SEEDING is Done')
seed_everything()

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        super(ImageDataset, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = os.listdir(root_dir)
        
        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index]*len(files)))
    
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])
        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations['image']
        return image, label

test_transform = A.Compose(
                [
                    A.Resize(96, 96),
                    A.Normalize(
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        max_pixel_value=255,
                        ),ToTensorV2()
                ])

test_data = ImageDataset(TEST, transform=test_transform)

#model
class Model(nn.Module):
    def __init__(self, in_channels = 3, numb_classes = 4):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = 8,
            kernel_size = (3,3),
            stride = (1,1),
            padding = (1,1),
        )
        self.pool1 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(
            in_channels = 8, 
            out_channels = 16,
            kernel_size = (3,3),
            stride = (1,1),
            padding = (1,1),
        )
        self.pool2 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        self.conv3 = nn.Conv2d(
            in_channels = 16, 
            out_channels = 32,
            kernel_size = (3,3),
            stride = (1,1),
            padding = (1,1),
        )
        
        self.pool3 = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        self.fc1 = nn.Linear(32*12*12, numb_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
model = Model(in_channels = 3, numb_classes=4)
test_loader = DataLoader(test_data, batch_size = 400, shuffle=False, num_workers=0)
probs = []
def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
for images, label in tqdm(test_loader):
    avg_preds=[]
    auc = 0.0
    images.to(device)
    label = label.to(device)
    states = [torch.load(PATH+f'model_fold{fold}_best.path') for fold in range(5)]
    for state in states:
        model.load_state_dict(state['model'])
        model.eval()
        with torch.no_grad():
            y_preds = model(images)
            predictions  = y_preds.argmax(1)
            oof_df1 = pd.DataFrame(y_preds.cpu().detach().numpy(), columns=['stage0', 'stage1', 'stage2', 'stage3'])
            oof_df1.loc[:, 'predictions'] = predictions.cpu().detach().numpy()
            oof_df1.loc[:, 'targets'] = label.cpu().detach().numpy()
            oof_df1.to_csv(OUTPUT_DIR + 'oof1_df.csv', index=False)
auc += get_score(label.cpu().detach().numpy(), y_preds.argmax(1).cpu().detach().numpy())
tqdm(test_loader).set_postfix(AUC = auc/len(test_loader))   
        #avg_preds.append(y_preds.softmax(1).cpu().detach().numpy())
    
    #avg_preds = np.mean(avg_preds, axis=0)
    #probs.append(avg_preds)
#tqdm(test_loader).set_postfix(AUC = auc/len(test_loader))
#probs = np.concatenate(probs)
