from resnet import *
from model_repo import *


model_name = 'MSE'
model_path = build_model_from(model_name)
model = Regression()
model = torch.load(model_name)