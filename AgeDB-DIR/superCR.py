import torch
from collections import OrderedDict
from resnet import *
from train import load_datasets
import argparse
from utils import cal_per_label_mae, cal_per_label_Frob

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--dataset', type=str, default='agedb', choices=['agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='/home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data', help='data directory')
parser.add_argument('--batch_size', type=int, default=128)



class Encoder_regression_single(nn.Module):
    def __init__(self, name='resnet18', norm=False, weight_norm= False):
        super(Encoder_regression_single, self).__init__()
        backbone, dim_in = model_dict[name]
        self.encoder = backbone()
        self.norm = norm
        self.weight_norm = weight_norm
        self.regressor = nn.Linear(512,1)
        

    def forward(self, x):
        feat = self.encoder(x)
        if self.norm:
            feat = F.normalize(feat, dim=-1)
        pred = self.regressor(feat)
        return pred, feat


class RNC(nn.Module):
    def __init__(self, name='resnet18'):
        super(RNC, self).__init__()
        #model_fun, dim_in = model_dict[name]
        self.model = get_model(name)
    

    def forward(self, x):
        feat = self.model.encoder(x)
        pred = self.model.regressor(feat)
        return pred, feat



def get_model(model_name='last.pth', regressor_name='./regressor.pth',  norm=False,  weight_norm= False):
    #
    model = Encoder_regression_single(name='resnet18', norm=norm, weight_norm=weight_norm).to(device)
    # Encoder
    ckpt = torch.load(model_name)
    new_state_dict = OrderedDict()
    for k,v in ckpt['model'].items():
        key = k.replace('module.','')
        keys = key.replace('encoder.','')
        new_state_dict[keys] =  v
    model.encoder.load_state_dict(new_state_dict)  
    #
    regressor = torch.load(regressor_name)
    model.regressor.load_state_dict(regressor['state_dict'])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=1e-4)
    return model, optimizer


if __name__ == '__main__':
    paths = '/home/rpu2/scratch/code/rnc_agedb/'
    model_name = paths + 'last.pth' 
    regressor_name = paths + 'regressor.pth'
    model = get_model(model_name, regressor_name)
    print('model loaded')
    #######
    args = parser.parse_args()
    train_loader, val_loader, test_loader, train_labels, diff_shots = load_datasets(args)
    maj_shot, med_shot, few_shot = diff_shots
    #######
    per_label_MAE_train = cal_per_label_mae(model, train_loader)
    print('===============train key MAE============='+'\n')
    k_train = [k for k in per_label_MAE_train.keys()]
    print(k_train + '\n')
    v_train = [per_label_MAE_train[k] for k in per_label_MAE_train.keys()]
    print(v_train + '\n')
    print('===============train MAE============='+'\n')
    per_label_MAE_test = cal_per_label_mae(model, test_loader)
    print('===============test key MAE============='+'\n')
    k_test = [k for k in per_label_MAE_test.keys()]
    print(k_test + '\n')
    v_test = [per_label_MAE_test[k] for k in per_label_MAE_test.keys()]
    print(v_test + '\n')
    print('===============test MAE============='+'\n')
    #
    per_label_Frobs_train = cal_per_label_Frob(model, train_loader)
    per_label_Frobs_test = cal_per_label_Frob(model, test_loader)
    k_frobs_train = [k for k in per_label_Frobs_train.keys()]
    k_frobs_test = [k for k in per_label_Frobs_test.keys()]
    v_frobs_train = [per_label_Frobs_train[k] for k in per_label_Frobs_train.keys()]
    v_frobs_test = [per_label_Frobs_train[k] for k in per_label_Frobs_test.keys()]
    print('===============train frobs key============='+'\n')
    print(k_frobs_train + '\n')
    print('===============train frobs============='+'\n')
    print(v_frobs_train + '\n')
    print('===============test frobs key============='+'\n')
    print(k_frobs_test + '\n')
    print('===============test frobs============='+'\n')
    print(v_frobs_test + '\n')