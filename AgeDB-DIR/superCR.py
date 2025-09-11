import torch
from collections import OrderedDict
from resnet import *



class Encoder_regression_single(nn.Module):
    def __init__(self, name='resnet50', norm=False, weight_norm= False):
        super(Encoder_regression_single, self).__init__()
        backbone, dim_in = model_dict[name]
        self.encoder = backbone()
        self.norm = norm
        self.weight_norm = weight_norm
        if self.weight_norm:
            self.regressor = torch.nn.utils.weight_norm(nn.Linear(dim_in, 1), name='weight')
        else:
            self.regressor = nn.Sequential(nn.Linear(dim_in, 1))

    def forward(self, x):
        feat = self.encoder(x)
        if self.norm:
            feat = F.normalize(feat, dim=-1)
        pred = self.regressor(feat)
        return pred, feat
    



def get_model(norm=False,  weight_norm= False):
    model_name = './last.pth'
    model = Encoder_regression_single(name='resnet18', norm=norm, weight_norm=weight_norm)  
    ckpt = torch.load(model_name)
    new_state_dict = OrderedDict()
    for k,v in ckpt['model'].items():
        key = k.replace('module.','')
        keys = key.replace('encoder.','')
        new_state_dict[keys] =  v
    model.encoder.load_state_dict(new_state_dict)   
    optimizer = torch.optim.SGD(model.regressor.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    return model, optimizer