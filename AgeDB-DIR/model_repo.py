import sys

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
    else:
        NotImplementedError