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
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
#from tensorboard_logger import Logger
from test import test

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
def post_hoc_train_one_epoch(model_regression, model_linear, train_loader, val_loader, maj_shot):
    # first calculate the prototypes
    #proto = cal_prototype(model, train_loader)
    frob_norm = cal_per_label_Frob(model_regression, train_loader)
    # first train the 1-d linear
    # orgnaize the (F, Y) pairs
    # this dictionary is constructed by (majority, its true frobs) + (med/few, its pred frobs)
    frob_norm_pred = {}
    maj_pairs_l, maj_pairs_f, leftover_l = [], [], []
    # orgnaize the majority in pairs (label, frobs)
    for label in np.unique(train_labels):
        if label in maj_shot:
            frobs = frob_norm[label]
            maj_pairs_l.append(label)
            maj_pairs_f.append(frobs)
            #
        else:
            # leftover (label, frob_norm) expect majority
            leftover_l.append(label)
    # fill the dictionary
    for l, f in zip(maj_pairs_l, maj_pairs_f):
        frob_norm_pred[l] = f
    #
    l_data = torch.Tensor(maj_pairs_l).float().unsqueeze(1) 
    f_data = torch.Tensor(maj_pairs_f).float()
    # dataset to train linear with only majority shot (label, frobs)
    linear_dataset = TensorDataset(l_data, f_data)
    linear_dataloader = DataLoader(linear_dataset, batch_size=4, shuffle=True)
    # to obtain the frobs prediction to regularize the few and median shot
    #
    model_linear.train()
    # train the linear to map (labels, frobs_norm)
    for idx, (l,  f) in enumerate(linear_dataloader):
        l, f = l.to(device), f.to(device)
        f_pred = model_linear(l)
        loss = nn.functional.mse_loss(f_pred, f)
        opt_linear.zero_grad()
        loss.backward()
        opt_linear.step()
    #
    model_linear.eval()
    #
    leftover_l = torch.Tensor(leftover_l).float().unsqueeze(1) 
    #
    with torch.no_grad():
        # the x here is the label
        leftover_f  = model_linear(leftover_l.to(device))
        #leftover_f = f_pred.cpu().view(-1).tolist()
    #
    # we treat the predicted value over the linear as the ground truth of the minority and median
    # therefore we construct the {label , frobs_pred} pairs given the predicted frobs on the few and med
    #
    for l, f in zip(leftover_l, leftover_f):
        frob_norm_pred[l.item()] = f.item()
    #
    model_regression.train()
    for idx, (x, y, _) in enumerate(val_loader):
        frob_loss = 0
        x, y = x.to(device), y.to(device)
        y_pred, z_pred = model_regression(x)
        z_pred_f_norm = torch.norm(z_pred, p='fro', dim=1)
        for y_ in torch.unqiue(y):
            idxs = (y == y_).nonzero(as_tuple=True)[0].unsqueeze(-1)
            pred_frob = torch.mean(z_pred_f_norm[idxs].float())
            gt_frob = frob_norm_pred[y_]
            frob_loss += nn.functional.mse_loss(pred_frob, gt_frob)
        mse_loss = nn.functional.mse_loss(y_pred, y)
        loss = mse_loss + frob_loss
        #
        opt_regression.zero_grad()
        loss.backward()
        opt_regression.step()
    #
    return model_regression, model_linear
    '''
    leftover_l = torch.Tensor(leftover_l).float().unsqueeze(1) 
    leftover_f = torch.Tensor(leftover_f).unsqueeze(-1)
    leftover_dataset = TensorDataset(leftover_l, leftover_f)
    # we concat the leftover (minority and median shots) with majority to formulate the new dataset
    sft_dataset = ConcatDataset([linear_dataset, leftover_dataset])
    sft_dataloader = DataLoader(sft_dataset, batch_size=4, shuffle=True)
    # the  x  here is the label and the y here  is the frobs norm
    #
    # we can fine tune the regression model noew
    #
    '''





if __name__ == '__main__':
    #
    args = parser.parse_args()
    #
    model_regression = build_model(args).to(device)
    model_linear = Linears().to(device)
    #
    train_loader, val_loader, test_loadder, train_labels, diff_shots = load_datasets(args)
    maj_shot, med_shot, few_shot = diff_shots
    #
    opt_regression = optim.Adam(model_regression.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_linear = optim.Adam(model_linear.parameters(), lr=1e-3, weight_decay=1e-4)
    #for e in range(args.warm_up_epoch):
    #    model = warm_up_one_epoch(model, train_loader, opt)
    for e in tqdm(range(args.epoch)):
        model_regression = train_one_epoch(model_regression, train_loader)
    ###############################
    model_regression, model_linear = post_hoc_train_one_epoch(model_regression, model_linear, train_loader, val_loader, maj_shot)
    # test
    mse_avg, l1_avg, loss_gmean = test(model_regression,test_loadder, train_labels, args)


    #torch.save(model, './MAE.pth')
    # this can be written for SDE-EDG
    #
    #
    # to do : calcualte the distance between the majority and minority