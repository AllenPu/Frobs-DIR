from resnet import resnet50
import torch
from collections import OrderedDict
from train import load_datasets, cal_prototype
import argparse
import os




os.environ["KMP_WARNINGS"] = "FALSE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='agedb', choices=['agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./data', help='data directory')

# load the pretrained model on contrastive learning
def load_model():
    model = resnet50()
    ckpt = torch.load('last.pthmodel_name')
    new_state_dict = OrderedDict()
    for k,v in ckpt['model'].items():
        key = k.replace('module.','')
        keys = key.replace('encoder.','')
        new_state_dict[keys] =  v
    model.load_state_dict(new_state_dict) 
    return model



if __name__ == "main":
    args, unknown = parser.parse_known_args()
    train_loader, val_loader, test_loader, train_lables = load_datasets(args)
    model = load_model().to(device)
    model.eval()
    label_feat = {}
    protos = cal_prototype(model, test_loader)
    # [label, feature]
    proto_list = torch.stack(protos)
    distances = torch.norm(protos[1:] - protos[:-1], dim=1)
    print(distances.tolist())
    print(list(protos.keys()))
            
    
