import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from train import load_datasets, build_model
from test import test
import argparse
import torch.optim as optim
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma=10):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)

#criterion = BMCLoss(init_noise_sigma=10)
#optimizer.add_param_group({'params': criterion.noise_sigma, 'lr': sigma_lr, 'name': 'noise_sigma'})
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--dataset', type=str, default='agedb', choices=['agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='/home/rpu2/scratch/data/imbalanced-regression/agedb-dir/data', help='data directory')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--resume', action='store_true', help='whether use the ptrtrained model')
parser.add_argument('--model_name', type=str, default='B-MSE' )


if __name__ == '__main__':
    args = parser.parse_args()
    model = build_model(args).to(device)
    train_loader, val_loader, test_loader, train_labels, _ = load_datasets(args)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bmc = BMCLoss(init_noise_sigma=10)
    for e in tqdm(range(args.epoch)):
        for idx, (x, y, _) in enumerate(train_loader):
            loss = 0
            x,y,w = x.to(device), y.to(device), w.to(device)
            y_pred, _ = model(x)
            #loss_mse = torch.nn.functional.mse_loss(y, y_pred, reduction='none')
            #loss_mse = torch.mean(torch.abs(y -  y_pred))
            loss += bmc(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    mse_avg, l1_avg, loss_gmean = test(model,test_loader, train_labels, args)
    #
    torch.save(model, './pretrained_models/bmse.pth')
    