import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms as T
from tqdm import tqdm

from dataset import FoodDataset
from model import vanillaCNN, vanillaCNN2, VGG19

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['CNN1', 'CNN2', 'VGG'], required=True, help='model architecture to train')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='the number of train epochs')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs('./save', exist_ok=True)
    os.makedirs(f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}', exist_ok=True)
    
    transforms = T.Compose([
        T.Resize((227,227), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomVerticalFlip(0.5),
        T.RandomHorizontalFlip(0.5),
    ])

    train_dataset = FoodDataset("./data", "train", transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dataset = FoodDataset("./data", "val", transforms=transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    if args.model == 'CNN1':
        model = vanillaCNN()
    elif args.model == 'CNN2':
        model = vanillaCNN2()
    elif args.model == 'VGG': 
        model = VGG19()
    else:
        raise ValueError("model not supported")
        
    ##########################   fill here   ###########################
        
    # TODO : Training Loop을 작성해주세요
    # 1. logger, optimizer, criterion(loss function)을 정의합니다.
    # train loader는 training에 val loader는 epoch 성능 측정에 사용됩니다.
    # torch.save()를 이용해 epoch마다 model이 저장되도록 해 주세요
            
    ######################################################################

    # Loss, optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epoch):
        model.train()   # train mode
        #for i, (images,labels) in enumerate(tqdm(train_loader)):
        for batch, data in enumerate(tqdm(train_loader)):
            images = data['input']  # shape : (N, 3, 227, 227)
            labels = data['target'] # shape : (N, )

            #images = images.to(device)
            #labels = labels.to(device)

            # Forward : call model.forward method
            outputs = model(images)

            # Calculate Loss
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()     # Calculate gadients
            optimizer.step()    # Update model weights


        # Validation
        model.eval()    # evaluation mode
        total = 0
        correct = 0

        with torch.no_grad():
            for data in val_loader:
              images = data['input']  # shape : (N, 3, 227, 227)
              labels = data['target'] # shape : (N, )
              
              total += len(images)
              #images = images.to(device)
              #labels = labels.to(device)

              # Forward : prediction
              outputs = model(images)
              _, pred = torch.max(outputs, dim=1)

              correct += (pred==labels).sum().item()

        print(f"After training epoch {epoch}, Validation accuracy : {correct/total}")
