from resnet import *
from model_repo import *


model_name = 'RankSim'
model_path = build_model_from(model_name)
model = Regression()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['state_dict'], strict=False)