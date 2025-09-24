import argparse
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

from model import *
from utils import cal_per_label_Frob, cal_per_label_mae, cal_per_label_frobs_mae
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
    # we can load any model with .pth
    #
    if args.resume:
        #prefix = ''
        model = resnet18(fds=False, bucket_num=100, bucket_start=3,
                     start_update=0, start_smooth=1,
                     kernel='gaussian', ks=9, sigma=1, momentum=0.9,
                     return_features=True)
        # ranksim
        checkpoint = torch.load('/home/rpu2/scratch/code/Con-R/agedb-dir/checkpoint/agedb_resnet50ConR_4.0_w=1.0_adam_l1_0.00025_64_2025-09-19-18:36:40.853379/ckpt.best.pth.tar')
        # Con-R
        #checkpoint = torch.load('/home/rpu2/scratch/code/Con-R/agedb-dir/checkpoint/agedb_resnet50ConR_4.0_w=1.0_adam_l1_0.00025_64_2025-09-19-18:36:40.853379/ckpt.best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"===> Checkpoint '{args.resume}' loaded (epoch [{checkpoint['epoch']}]), testing...")
        # CR : /home/rpu2/scratch/code/last/pth
    
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
        # change  the loss from MAE to MSE
        loss_mse = torch.mean(torch.abs(y -  y_pred)**2)
        # LDS
        #loss_mse = torch.mean(loss_mse * w.expand_as(loss_mse))
        loss += loss_mse
        opt.zero_grad()
        loss.backward()
        opt.step()
    #mse_avg, l1_avg, loss_gmean = test(model,test_loader, train_labels, args)
    #print(f' Maj MAE {} Med MAE {} Few MAE {}')
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
    # we stop here because we want to record the Frobenius norm only
    # if we want to implement the post-hoc-train, start here and remove assert
    #

    #######
    #mse_avg, l1_avg, loss_gmean = test(model_regression, test_loader, train_labels, args)
    mse_avg, l1_avg, loss_gmean = test(model_regression, train_loader, train_labels, args)
    #
    cal_per_label_frobs_mae(model_regression, train_loader, test_loader, model_name='RankSim_SFT')
    #
    # just want to test the train MAR
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
    #
    #cal_per_label_frobs_mae(model_regression, train_loader, test_loader, model_name='RankSim_SFT')

    #torch.save(model, './MAE.pth')
    # this can be written for SDE-EDG
    #
    #
    # to do : calcualte the distance between the majority and minority