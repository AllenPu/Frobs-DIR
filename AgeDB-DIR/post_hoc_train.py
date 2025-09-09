import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import cal_per_label_Frob
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#####################################
def post_hoc_train_one_epoch(models, loaders, opts, train_labels, maj_shot):
    #
    model_regression, model_linear = models
    train_loader, val_loader = loaders
    opt_regression, opt_linear = opts
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
        print(f' shape of f {f.shape} shape of f_pred {f_pred.shape}')
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
        for y_ in torch.unique(y):
            idxs = (y == y_).nonzero(as_tuple=True)[0].unsqueeze(-1)
            #print(f'f pred is {z_pred_f_norm[idxs]} idxs {idxs} y {y} y_ {y_}')
            pred_frob = torch.mean(z_pred_f_norm[idxs].float())
            print('====', pred_frob)
            gt_frob = frob_norm_pred[y_.item()]
            #print(f'pred frob {pred_frob} gt forb {gt_frob}')
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
