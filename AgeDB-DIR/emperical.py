from resnet import *
import torch
from collections import OrderedDict
from train import load_datasets, cal_prototype
import argparse
import os




os.environ["KMP_WARNINGS"] = "FALSE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device is ", device)


parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--dataset', type=str, default='agedb', choices=['agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='/home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data', help='data directory')
parser.add_argument('--batch_size', type=int, default=128)

# load the pretrained model on contrastive learning
def load_model():
    model = Regression(name='resnet18')
    ckpt = torch.load('/home/rpu2/scratch/code/imbalanced-contrastive-ordinary-regression/agedb/models/last.pth')
    new_state_dict = OrderedDict()
    for k,v in ckpt['model'].items():
        key = k.replace('module.','')
        keys = key.replace('encoder.','')
        new_state_dict[keys] =  v
    model.encoder.load_state_dict(new_state_dict) 
    return model



if __name__ == '__main__':
    print('start')
    args = parser.parse_args()
    print('load hyper-param')
    train_loader, val_loader, test_loader, train_lables = load_datasets(args)
    print('data loaded')
    #model = load_model().to(device)
    model = Regression(name='resnet18')
    #
    model_name = 'MSE' + '.pth'
    #
    model = torch.load(model_name)
    #
    #model.load_state_dict(ckpt.state_dict)
    #
    print('model loaded')
    model.eval()
    proto, labels = cal_prototype(model, test_loader)
    # [feature]
    protos = torch.stack(proto, dim=0)
    distances = torch.norm(protos[1:] - protos[:-1], dim=1).tolist()
    print('======================')
    with open(f'dis_{model_name}.txt', 'a') as f:
        for e in distances:
            f.write(str(e) + '\n')
        f.close()
    with open(f'label_{model_name}.txt', 'a') as f:
        for e in labels:
            f.write(str(e) + '\n')
        f.close()
    
            
    
