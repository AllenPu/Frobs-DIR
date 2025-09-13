import torch
from collections import OrderedDict
from resnet import *



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


class RNC(nn.module):
    def __init__(self, name='resnet18'):
        super(RNC, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.model = get_model(name)
    

    def forward(self, x):
        feat = self.model.encoder(x)
        pred = self.model.regressor(feat)
        return pred, feat



def get_model(model_name='last.pth', regressor_name='./regressor.pth',  norm=False,  weight_norm= False):
    model = Encoder_regression_single(name='resnet18', norm=norm, weight_norm=weight_norm)  
    # Encoder
    ckpt = torch.load(model_name)
    new_state_dict = OrderedDict()
    for k,v in ckpt['model'].items():
        key = k.replace('module.','')
        keys = key.replace('encoder.','')
        new_state_dict[keys] =  v
    model.encoder.load_state_dict(new_state_dict)  
    model.regressor.load_state_dict(regressor_name['state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=1e-4)
    return model, optimizer


if __name__ == '__main__':
    model_name = './last.pth'
    regressor_name = './regressor.pth'
    model = get_model()