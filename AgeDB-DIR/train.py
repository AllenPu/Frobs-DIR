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

import os
os.environ["KMP_WARNINGS"] = "FALSE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)




def buil_model(args):
    return resnet18()




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
