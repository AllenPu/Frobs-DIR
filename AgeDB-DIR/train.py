import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from scipy.stats import gmean

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboard_logger import Logger


from resnet import *
from utils import *
from agedb import *

import os
os.environ["KMP_WARNINGS"] = "FALSE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='agedb', choices=['agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./data', help='data directory')



def build_model(args):
    return resnet18()



def load_datasets(args):
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']

    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=224, split='train')
    val_dataset = AgeDB(data_dir=args.data_dir, df=df_val, img_size=224, split='val')
    test_dataset = AgeDB(data_dir=args.data_dir, df=df_test, img_size=224, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
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
    model.train()
    for idx, (x,y) in enumerate(train_loader):
        x,y = x.to(device), y.to(device)
        y_pred, _ = model(x)
        loss = torch.nn.functional.mse_loss(y, y_pred)
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
        for idx, (x,y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            _, z_pred = model(x)
            for l in y.unique(sort=True):
                rows = z_pred[y == l]
                keys = int(l.item())
                label_feat[keys] = label_feat.get(keys, []) + list(rows.unbind(0))
        proto = [torch.mean(label_feat[e], 0) for e in label_feat.keys()]
    return proto



if __name__ == "main":
    args, unknown = parser.parse_known_args()
    model = build_model(args)
