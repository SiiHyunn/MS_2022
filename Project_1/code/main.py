import copy

import pandas as pd
import os
from tqdm import tqdm
import sys


from customdataset import CustomDataset
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy
import warnings

warnings.filterwarnings("ignore")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## 0. Augmentation
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Resize(224, 224),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomShadow(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transfrom = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


    ## 1. Loading Classification Dataset
    train_dataset = CustomDataset("/Users/sihyun/Downloads/dataset/train", transform=train_transform)
    val_dataset = CustomDataset("/Users/sihyun/Downloads/dataset/val", transform=val_transfrom)
    test_dataset = CustomDataset("/Users/sihyun/Downloads/dataset/test", transform=val_transfrom)

    ## def visualize_augmentations()
    def visualize_augmentations(dataset, idx=0, samples=20, cols=5):
        dataset = copy.deepcopy(dataset)
        dataset.transform = A.Compose([
            t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))
        ])
        rows = samples // cols
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6))
        for i in range(samples):
            image, _ = dataset[idx]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()

    ## 2. Data Loader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    ## 3. model loader
    
    # mobilnet_v2
    resnet18 = models.resnet18(pretrained=True)
    # net = models.mobilenet.mobilenet_v2(pretrained=True)
    # resnet18 = optim.Adam(resnet18.parameters(), lr=0.001)
    # resnet18.to(device)

    num_classes = 10
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, num_classes)

    evice = torch.device('cuda:0')
    resnet18.to(device)  

    ## 4. hy_parameter 지정
    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.AdamW(resnet18.parameters(), lr=0.001)
    epochs = 5

    # 5. 변수 선언
    best_val_acc = 0.0
    train_step = len(train_loader)
    valid_step = len(val_loader)
    save_path = "best.pt"
    dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                                 columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

    for epoch in range(epochs):
        running_loss = 0
        val_acc = 0
        train_acc = 0
        val_running_loss = 0

        # train code
        resnet18.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour='red')
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = resnet18(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs, 1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = f"train epoch [{epoch+1}/{epochs}], loss >> {loss.data:.3f}"

        # valid code
        resnet18.eval()
        with torch.no_grad():
            valid_bar = tqdm(val_loader, file=sys.stdout, colour="green")
            for data in valid_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                val_outputs = resnet18(images)
                val_loss = loss_function(val_outputs, labels)
                val_running_loss = val_loss.item()
                val_acc += (torch.argmax(val_outputs, dim=1) == labels).sum().item()

        val_accuracy = val_acc / len(val_dataset)
        train_accuracy = train_acc / len(train_dataset)

        # 그래프 그리기 위하여 csv로 저장
        dfForAccuracy.loc[epoch, 'epoch'] = epoch + 1
        dfForAccuracy.loc[epoch, 'train_loss'] = round((running_loss / train_step), 3)
        dfForAccuracy.loc[epoch, 'val_loss'] = round((val_running_loss / valid_step), 3)
        dfForAccuracy.loc[epoch, 'train_acc'] = round(train_accuracy, 3)
        dfForAccuracy.loc[epoch, 'val_acc'] = round(val_accuracy, 3)

        print(f"epoch [{epoch+1}/{epochs}]"
              f" train loss : {(running_loss / train_step):.3f} val_loss : {(val_running_loss / valid_step):.3f} "
              f"train_acc : {train_accuracy:.3f} val_acc : {val_accuracy:.3f}")

        # best.pt 저장
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(resnet18.state_dict(), save_path)

        # csv 저장
        if epoch == epochs - 1:
            dfForAccuracy.to_csv("./modelAccuracy.csv", index = False)

if __name__ == "__main__":
    main()