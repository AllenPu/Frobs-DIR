import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import gmean
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AverageMeter(object):
    def __init__(self,  name = '', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    


def shot_metric(pred, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    # input of the pred & labels are all numpy.darray
    # train_labels is from csv , e.g. df['age']
    #
    preds = np.hstack(pred)
    labels = np.hstack(labels)
    #
    train_labels = np.array(train_labels).astype(int)
    #
    train_class_count, test_class_count = [], []
    #
    l1_per_class, l1_all_per_class = [], []
    #
    for l in np.unique(labels):
        train_class_count.append(len(
            train_labels[train_labels == l]))
        test_class_count.append(
            len(labels[labels == l]))
        l1_per_class.append(
            np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(
            np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
            #print(train_class_count[i])
            #print(l1_per_class[i])
            #print(l1_all_per_class[i])
        else:
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    #
    shot_dict = defaultdict(dict)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    #
    shot_dict['median']['l1'] = np.sum(
        median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    #
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict






def test(model, test_loader, train_labels, args):
    model.eval()
    #
    mse_pred = AverageMeter()
    acc_mae_gt = AverageMeter()
    acc_mae_pred = AverageMeter()
    # gmean
    criterion_gmean_gt = nn.L1Loss(reduction='none')
    criterion_gmean_pred = nn.L1Loss(reduction='none')
    gmean_loss_all_gt, gmean_loss_all_pred = [], [] 
    #
    pred_gt, pred, labels, = [], [], []
    #
    with torch.no_grad():
        for idx, (x, y, g) in enumerate(test_loader):
            bsz = x.shape[0]
            x, y, g = x.to(device), y.to(device), g.to(device)
        #
            labels.extend(y.data.cpu().numpy())
            y_output, _ = model(x)
            #
            #print(f' y shape is  {y_output.shape}')
            #
            y_chunk = torch.chunk(y_output, 2, dim=1)
            g_hat, y_pred = y_chunk[0], y_chunk[1]
            #
            g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            # newly added
            #
            y_hat = torch.gather(y_pred, dim=1, index=g_index)
            y_pred_gt = torch.gather(y_pred, dim=1, index=g.to(torch.int64))
            #
            mae_y = torch.mean(torch.abs(y_hat - y))
            mae_y_gt = torch.mean(torch.abs(y_pred_gt - y))
            mse_y_pred = F.mse_loss(y_hat, y)
            #
            pred.extend(y_hat.data.cpu().numpy())
            pred_gt.extend(y_pred_gt.data.cpu().numpy())
            #
            # gmean
            loss_all_gt = criterion_gmean_gt(y_pred_gt, y)
            loss_all_pred = criterion_gmean_pred(y_hat, y)
            gmean_loss_all_gt.extend(loss_all_gt.cpu().numpy())
            gmean_loss_all_pred.extend(loss_all_pred.cpu().numpy())
            #
            mse_pred.update(mse_y_pred.item(), bsz)
            #
            acc_mae_gt.update(mae_y_gt.item(), bsz)
            acc_mae_pred.update(mae_y.item(), bsz)
        #
        # gmean
        gmean_gt = gmean(np.hstack(gmean_loss_all_gt), axis=None).astype(float)
        gmean_pred = gmean(np.hstack(gmean_loss_all_pred), axis=None).astype(float)
        shot_pred = shot_metric(pred, labels, train_labels)
        shot_pred_gt = shot_metric(pred_gt, labels, train_labels)
    print(f' MSE is {mse_pred.avg}')

    return acc_mae_gt.avg, acc_mae_pred.avg, shot_pred, shot_pred_gt, gmean_gt, gmean_pred
        # np.hstack(group), np.h