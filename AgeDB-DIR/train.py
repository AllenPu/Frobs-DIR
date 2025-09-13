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
from utils import cal_per_label_Frob, cal_per_label_mae
from post_hoc_train import post_hoc_train_one_epoch
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
parser.add_argument('--model_name', type=str, default='MSE' )
parser.add_argument('--sft_epoch', type=int, default=1, help='how much epoch used to fine tune the pre-trained model for whole post-hoc')
parser.add_argument('--linear_epoch', type=int, default=10, help='epoch to train the linear mapping')
parser.add_argument('--regression_epoch', type=int, default=10, help='SFT epoch in each post-hoc training')




def build_model(args):
    if args.resume:
        # CR : /home/rpu2/scratch/code/last/pth
        model_name = args.model_name + '.pth'
        model_path = os.path.join('./trained_models/', model_name)
        model = torch.load(model_path)
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




if __name__ == '__main__':
    #
    args = parser.parse_args()
    #
    model_regression = build_model(args).to(device)
    model_linear = Linears().to(device)
    #
    train_loader, val_loader, test_loader, train_labels, diff_shots = load_datasets(args)
    maj_shot, med_shot, few_shot = diff_shots
    #
    opt_regression = optim.Adam(model_regression.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_linear = optim.Adam(model_linear.parameters(), lr=1e-3, weight_decay=1e-4)
    #for e in range(args.warm_up_epoch):
    #    model = warm_up_one_epoch(model, train_loader, opt)
    if not args.resume:
        for e in tqdm(range(args.epoch)):
            model_regression = train_one_epoch(model_regression, train_loader, opt_regression)
    print('==================Before SFT===================')
    #######
    #
    # We add this to show the train and test MAE
    #
    per_label_MAE_train = cal_per_label_mae(model_regression, train_loader)
    print('===============train key MAE============='+'\n')
    k_train = [k for k in per_label_MAE_train.keys()]
    print(f'k_train is {k_train}')
    v_train = [per_label_MAE_train[k] for k in per_label_MAE_train.keys()]
    print(f'v_train is {v_train}')
    print('===============train MAE============='+'\n')
    per_label_MAE_test = cal_per_label_mae(model_regression, test_loader)
    print('===============test key MAE============='+'\n')
    k_test = [k for k in per_label_MAE_test.keys()]
    print(f'k_test is {k_test}')
    v_test = [per_label_MAE_test[k] for k in per_label_MAE_test.keys()]
    print(f'v_test is {v_test}')
    print('===============test MAE============='+'\n')
    #
    per_label_Frobs_train = cal_per_label_Frob(model_regression, train_loader)
    per_label_Frobs_test = cal_per_label_Frob(model_regression, test_loader)
    k_frobs_train = [k for k in per_label_Frobs_train.keys()]
    k_frobs_test = [k for k in per_label_Frobs_test.keys()]
    v_frobs_train = [per_label_Frobs_train[k] for k in per_label_Frobs_train.keys()]
    v_frobs_test = [per_label_Frobs_train[k] for k in per_label_Frobs_test.keys()]
    print('===============train frobs key============='+'\n')
    print(f'k_frobs_train is {k_frobs_train}')
    print('===============train frobs============='+'\n')
    print(f'v_frobs_train is {v_frobs_train}')
    print('===============test frobs key============='+'\n')
    print(f'k_frobs_test is {k_frobs_test}')
    print('===============test frobs============='+'\n')
    print(f'v_frobs_test is {v_frobs_test}')

    ####
    df = pd.DataFrame({
        "train MAE labels" : k_train,
        "train MAE" : v_train,
        "train Frobs" : v_frobs_train,
        "test MSE labels" : k_test,
        "test MAE" : v_test,
        "test Frobs" : v_frobs_test
    })

    assert 1 == 2


    #######
    mse_avg, l1_avg, loss_gmean = test(model_regression,test_loader, train_labels, args)
    #
    models = [model_regression,model_linear]
    opts = [opt_regression, opt_linear]
    loaders = [train_loader, val_loader]
    #
    # regression epoch is for SFT, linear epoch is for train the linear mapping
    regression_epoch, linear_epoch = args.regression_epoch, args.linear_epoch
    epochs = [regression_epoch, linear_epoch]
    ###############################
    for e in range(args.sft_epoch):
        model_regression, model_linear = post_hoc_train_one_epoch(models, loaders, opts, train_labels, maj_shot, epochs)
    # test
    print('===================After SFT======================')
    mse_avg, l1_avg, loss_gmean = test(model_regression,test_loader, train_labels, args)


    #torch.save(model, './MAE.pth')
    # this can be written for SDE-EDG
    #
    #
    # to do : calcualte the distance between the majority and minority