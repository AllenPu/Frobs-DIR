import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from scipy.stats import gmean
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
#from tensorboard_logger import Logger


from resnet import *
from utils import *
from agedb import *

import os
os.environ["KMP_WARNINGS"] = "FALSE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device is ", device)

parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--dataset', type=str, default='agedb', choices=['agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='/home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data', help='data directory')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--warm_up_epoch', type=int, default=30)
parser.add_argument('--epoch', type=int, default=100)



def build_model(args):
    return Regression(name='resnet18')



def load_datasets(args):
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']

    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=224, split='train')
    val_dataset = AgeDB(data_dir=args.data_dir, df=df_val, img_size=224, split='val')
    test_dataset = AgeDB(data_dir=args.data_dir, df=df_test, img_size=224, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, train_labels


def warm_up_one_epoch(model, train_loader, opt):
    for idx, (x, y) in enumerate(train_loader):
        x,y = x.to(device), y.to(device)
        y_pred, _ = model(x)
        loss = torch.nn.functional.mse_loss(y, y_pred)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model



def train_one_epoch(model, train_loader, opt):
    ##################################
    #proto = cal_prototype(model, train_loader)
    ##################################
    model.train()
    for idx, (x,y, w) in enumerate(train_loader):
        loss = 0
        x,y,w = x.to(device), y.to(device), w.to(device)
        y_pred, _ = model(x)
        #loss_mse = torch.nn.functional.mse_loss(y, y_pred, reduction='none')
        loss_mse = torch.mean(torch.abs(y -  y_pred))
        # LDS
        #loss_mse = torch.mean(loss_mse * w.expand_as(loss_mse))
        loss += loss_mse
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model
   

##################################
# return the prototype of each label
def cal_prototype(model, train_loader):
    model.eval()
    with torch.no_grad():
        label_feat, proto = {}, []
        for idx, (x,y, _) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            _, z_pred = model(x)
            for l in y.unique(sorted=True):
                index = y==l
                index = index.squeeze(-1)
                rows = z_pred[index]
                keys = int(l.item())
                label_feat[keys] = label_feat.get(keys, []) + list(rows.unbind(0))
                #print('shape ', label_feat[keys])
        sorted_label_feat = {key: label_feat[key] for key in sorted(label_feat.keys())}
        proto = [torch.stack(sorted_label_feat[e], dim=0) for e in sorted_label_feat.keys()]
        proto = [torch.mean(p, 0) for p in proto]
        labels = [k for k in sorted_label_feat.keys()]
    return proto, labels



if __name__ == '__main__':
    args = parser.parse_args()
    model = build_model(args).to(device)
    train_loader, val_loader, test_laoder, train_labels = load_datasets(args)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    #for e in range(args.warm_up_epoch):
    #    model = warm_up_one_epoch(model, train_loader, opt)
    for e in tqdm(range(args.epoch)):
        model = train_one_epoch(model, train_loader, opt)
    #torch.save(model, './MAE.pth')
# this can be written for SDE-EDG
    #
    #
    # to do : calcualte the distance between the majority and minority