from resnet import *
import torch
from collections import OrderedDict
from train import load_datasets, cal_prototype
import argparse
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



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


def draw_tsne(protos, model_name):
    tsne = TSNE(n_components=2, random_state=0)
    data_tsne = tsne.fit_transform(protos)
    plt.figure(figsize=(10,10))
    plt.scatter(data_tsne[:,0], data_tsne[:,1])
    plt.title(f' T-SNE in loss {model_name}')
    plt.savefig(f'./{model_name}.jpg')


if __name__ == '__main__':
    print('start')
    args = parser.parse_args()
    print('load hyper-param')
    train_loader, val_loader, test_loader, train_lables = load_datasets(args)
    print('data loaded')
    model = load_model().to(device)
    #
    model_loss = 'SuperCR'
    #
    #model_name = model_loss + '.pth'
    #
    #model = torch.load(model_name)
    #
    #model.load_state_dict(ckpt.state_dict)
    #
    print('model loaded')
    model.eval()
    proto, labels = cal_prototype(model, test_loader)
    # [feature]
    protos = torch.stack(proto, dim=0).tolist()
    #
    draw_tsne(protos, 'SuperCR')
    '''
    distances = torch.norm(protos[1:] - protos[:-1], dim=1).tolist()
    print('======================')
    with open(f'dis_{model_loss}.txt', 'a') as f:
        for e in distances:
            f.write(str(e) + '\n')
        f.close()
    with open(f'label_{model_loss}.txt', 'a') as f:
        for e in labels:
            f.write(str(e) + '\n')
        f.close()
    '''
    
            
    
