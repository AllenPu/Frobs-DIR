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
parser.add_argument('--resume', action='store_true', help='whether use the ptrtrained model')
parser.add_argument('--model_name', type=str, default='./MSE.pth' )



def build_model(args):
    if args.resume:
        model = torch.load(args.model_name)
    else:
        model = Regression(name='resnet18')
    return model



def load_datasets(args):
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']

    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=224, split='train')
    #
    many_shot, med_shot, few_shot = train_dataset.cal_shots()
    diff_shots = [many_shot, med_shot, few_shot]
    #
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
    return train_loader, val_loader, test_loader, train_labels, diff_shots


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
   


# return the linear coifficient 





#####################################
def post_hoc_train_one_epoch(model, train_loader, maj_shot, opt):
    # first calculate the prototypes
    #proto = cal_prototype(model, train_loader)
    frob_norm = cal_per_label_Frob(model, train_loader)
    # first train the 1-d linear
    # orgnaize the (F, Y) pairs
    maj_pairs = [], maj_pair_index = []
    #
    for idx, (x, y, _) in enumerate(train_loader):
        x,y = x.to(device), y.to(device)
        y_index, y_maj = match_A_in_B(y, maj_shot)
        # 
        y_maj_uniq = torch.unique(y_maj)
        #
        y_pred, z_pred = model(x)
        #
        y_list = [e.item() for e in y]
        maj_pair_index.extend(y_list)
        maj_pair_index = set(maj_pair_index)
        sub_proto = [(key, frob_norm[key.item()]) for key in y_maj_uniq if key not in maj_pair_index]
        maj_pair_index = [*maj_pair_index]
        

    return 0





if __name__ == '__main__':
    args = parser.parse_args()
    model = build_model(args).to(device)
    train_loader, val_loader, test_laoder, train_labels, diff_shots = load_datasets(args)
    many_shot, med_shot, few_shot = diff_shots
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    #for e in range(args.warm_up_epoch):
    #    model = warm_up_one_epoch(model, train_loader, opt)
    for e in tqdm(range(args.epoch)):
        model = train_one_epoch(model, train_loader, opt)
    ###############################
    
    #torch.save(model, './MAE.pth')
# this can be written for SDE-EDG
    #
    #
    # to do : calcualte the distance between the majority and minority