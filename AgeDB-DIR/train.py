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


from utils import *

import os
os.environ["KMP_WARNINGS"] = "FALSE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)







def warm_up_one_epoch(model, train_loader, opt):
    for idx, (x, y) in enumerate(train_loader):
        x,y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y, y_pred)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model
