import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
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

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
input_dir = "C://Users//Myo Min Htet//Downloads//Compressed//NitrogenDeficiencyImage"
os.listdir(input_dir)
TRAIN = "C://Users//Myo Min Htet//Downloads//Compressed//NitrogenDeficiencyImage//Training"
TEST = "C://Users//Myo Min Htet//Downloads//Compressed//NitrogenDeficiencyImage//Test"

OUTPUT_DIR = 'C://Users//Myo Min Htet//OneDrive//Desktop//New folder//youngsavage//output//'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def inform(dir):

    if dir == TRAIN:
        print("Training Information")
    else:
        print("Test Information")
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
    
inform(TRAIN)
inform(TEST)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str('seed')
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

train_transform = A.Compose(
                [
                    A.Resize(96, 96),
                    A.Rotate(limit = 40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.Normalize(
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        max_pixel_value=255,
                        ),ToTensorV2(),
                ])
test_transform = A.Compose(
                [
                    A.Resize(96, 96),
                    A.Normalize(
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        max_pixel_value=255,
                        ),ToTensorV2()
                ])

train_data = ImageDataset(root_dir = TRAIN, transform = train_transform)
for i in range(4):
    image, label = train_data[i]
    plt.subplots()
    plt.imshow(image.permute(2,1,0))
    plt.title(f'label: {label}')
    plt.show
    
test_data = ImageDataset(root_dir = TEST, transform = test_transform)
for i in range(4):
    image, label = test_data[i]
    plt.subplots()
    plt.imshow(image.permute(2,1,0))
    plt.title(f'label: {label}')
    plt.show()
    
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

train_loader = DataLoader(train_data, batch_size = 64, num_workers=4, shuffle=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = 0.001)
writer =SummaryWriter(f'runs/tryingout_tensorboard')

def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

#Define the Kfold cross validator
kfold = KFold(n_splits=5, shuffle=True)

for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_data)):
    print(f"fold {fold}")
    #sample elements randomly from a given list of ids
    train_subsampler = SubsetRandomSampler(train_ids)
    valid_subsampler = SubsetRandomSampler(valid_ids)

    train_loader = DataLoader(train_data, batch_size=64, sampler=train_subsampler)
    valid_loader = DataLoader(train_data, batch_size=64, sampler=valid_subsampler)
    
    for epoch in range(epochs):
        model.to(device)
        model.train()
        losses = []
        accuracies = []
        train_loss = 0.0
        train_accuracy = 0.0
        step = 0

        loop1 = tqdm(train_loader)
        for image, label in loop1:
            image = image.to(device)
            label = label.to(device)

            preds = model(image)
            loss = criterion(preds, label)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_accuracy += get_score(label.cpu().detach().numpy(), preds.argmax(1).cpu().detach().numpy())
            accuracies.append(train_accuracy/len(train_loader))

            loop1.set_description(f'EPOCH: {epoch+1}/{epochs} TRAIN')
            loop1.set_postfix(TRAIN_AUC = train_accuracy/len(train_loader), TRAIN_LOSS = train_loss/len(train_loader))
            TRAIN_AUC = train_accuracy/len(train_loader)
            writer.add_scalar('Training Loss', loss, global_step=step)
            writer.add_scalar('training Accuracy', TRAIN_AUC, global_step=step)
            step += 1

        with torch.no_grad():
            valid_loss = 0.0
            valid_accuracy = 0.0
            num_corrects = 0.0
            count = 0

            model.eval()
            loop2 = tqdm(valid_loader)
            for x, y in loop2:
                x = x.to(device)
                y = y.to(device)
                output = model(x)

                loss = criterion(output, y)
                valid_loss += loss.item()

                predictions = output.argmax(1)

                score = get_score(y.cpu().detach().numpy(), output.argmax(1).cpu().detach().numpy())

                best_score = 0.0
                if score > best_score:
                    best_score = score
                    torch.save({'model': model.state_dict()},OUTPUT_DIR + f'model_fold{fold}_best.path')
                check_point = torch.load(OUTPUT_DIR + f'model_fold{fold}_best.path')

                if fold == 4:
                    oof_df = pd.DataFrame(output.cpu().detach().numpy(), columns=['stage0', 'stage1', 'stage2', 'stage3'])
                    oof_df.loc[:, 'predictions'] = predictions.cpu().detach().numpy()
                    oof_df.loc[:, 'targets'] = y.cpu().detach().numpy()
                    oof_df.loc[:, 'fold'] = fold
                    oof_df.to_csv(OUTPUT_DIR + 'oof_df.csv', index=False)
                valid_accuracy += get_score(y.cpu().detach().numpy(), output.argmax(1).cpu().detach().numpy())
                loop2.set_description(f'EPOCH: {epoch+1}/{epochs} VALID')
                loop2.set_postfix(VAL_AUC = valid_accuracy/len(valid_loader), VAL_LOSS = valid_loss/len(valid_loader))

                VAL_AUC = valid_accuracy/len(valid_loader)
                writer.add_scalar('Validation Loss', loss, global_step=step)
                writer.add_scalar('Validation Accuracy', VAL_AUC, global_step=count)
                count += 1
    
                    