import sys
#from model import *
from resnet import *
import os
from collections import OrderedDict

def build_model_from(model_name):
    if model_name == 'RankSim':
        return '/home/rpu2/scratch/code/ranksim/agedb-dir/checkpoint/agedb_resnet18_reg100.0_il2.0_adam_l1_0.00025_256_2025-09-24-06:52:59.565226/ckpt.best.pth.tar'
    elif model_name == 'ConR':
        return '/home/rpu2/scratch/code/Con-R/agedb-dir/checkpoint/agedb_resnet50ConR_4.0_w=1.0_adam_l1_0.00025_64_2025-09-19-18:36:40.853379/ckpt.best.pth.tar'
    elif model_name == 'MSE':
        return '/home/rpu2/scratch/code/Frobs-DIR/AgeDB-DIR/trained_models/MSE.pth'
    elif model_name == 'MAE':
        return '/home/rpu2/scratch/code/Frobs-DIR/AgeDB-DIR/trained_models/MAE.pth'
    elif model_name == 'MSE_LDS':
        return '/home/rpu2/scratch/code/Frobs-DIR/AgeDB-DIR/trained_models/MSE_LDS.pth'
    elif model_name == 'MSE_LDS':
        return '/home/rpu2/scratch/code/Frobs-DIR/AgeDB-DIR/trained_models/MAE_LDS.pth'
    elif model_name == 'BMSE':
        return '/home/rpu2/scratch/code/Frobs-DIR/AgeDB-DIR/trained_models/bmse.pth'
    elif model_name == 'SuperCR':
        return '/home/rpu2/scratch/code/Frobs-DIR/AgeDB-DIR/trained_models/rnc'
    else:
        NotImplementedError


def build_models(model_name):
    model_path = build_model_from(model_name)
    if model_name in ['RankSim', 'ConR']:
        model = resnet18(fds=False, bucket_num=100, bucket_start=3,
                     start_update=0, start_smooth=1,
                     kernel='gaussian', ks=9, sigma=1, momentum=0.9,
                     return_features=True)
        # Ranksim
        checkpoint = torch.load(model_path)
        # ConR
        #checkpoint = torch.load('/home/rpu2/scratch/code/Con-R/agedb-dir/checkpoint/agedb_resnet50ConR_4.0_w=1.0_adam_l1_0.00025_64_2025-09-19-18:36:40.853379/ckpt.best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    elif model_name in ['MAE', 'MSE', 'MAE_LDS', 'MSE_LDS']:
        #
        model = torch.load(model_name)
    elif model_name in ['SuperCR']:
        #
        model = Regression()
        #
        model_path_encoder, model_path_regressor = os.path.join(model_path, 'last.pth'), os.path.join(model_path, 'regressor.pth')
        ckpt = torch.load(model_path_encoder)
        new_state_dict = OrderedDict()
        for k,v in ckpt['model'].items():
            key = k.replace('module.','')
            keys = key.replace('encoder.','')
            new_state_dict[keys] =  v
        model.encoder.load_state_dict(new_state_dict)
        # load regressor
        ckpt_regressor =  torch.load(model_path_regressor)                          
        regressor_state_dict = OrderedDict()
        for k,v in ckpt_regressor['state_dict'].items():
            k = '0.' + k
            regressor_state_dict[k] =  v
        model.regressor.load_state_dict(regressor_state_dict)
        return model