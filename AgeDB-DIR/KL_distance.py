import torch 


class KLDivergenceLossExplicit(nn.Module):
    def __init__(self, temperature=1.0, eps=1e-8):
        super(KLDivergenceLossExplicit, self).__init__()
        self.temperature = temperature
        self.eps = eps
    
    def forward(self, y_true, l_pred, distance_type='l2'):
        """
        Explicit implementation with double loops (for clarity)
        """
        batch_size = y_true.shape[0]
        
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(-1)
        if l_pred.dim() == 1:
            l_pred = l_pred.unsqueeze(-1)
        
        # Pre-compute all pairwise distances
        if distance_type == 'l2':
            d_y = torch.cdist(y_true, y_true, p=2)
            d_l = torch.cdist(l_pred, l_pred, p=2)
        elif distance_type == 'l1':
            d_y = torch.cdist(y_true, y_true, p=1)
            d_l = torch.cdist(l_pred, l_pred, p=1)
        
        # Compute normalization constants
        exp_neg_d_y = torch.exp(-d_y / self.temperature)
        exp_neg_d_l = torch.exp(-d_l / self.temperature)
        
        Z_y = torch.sum(exp_neg_d_y)  # Normalization for p_ij
        Z_l = torch.sum(exp_neg_d_l)  # Normalization for q_ij
        
        # Compute KL divergence
        kl_loss = 0.0
        for i in range(batch_size):
            for j in range(batch_size):
                p_ij = exp_neg_d_y[i, j] / Z_y
                q_ij = exp_neg_d_l[i, j] / Z_l
                
                # Add epsilon to prevent log(0)
                p_ij = p_ij + self.eps
                q_ij = q_ij + self.eps
                
                kl_loss += p_ij * torch.log(p_ij / q_ij)
        
        return kl_loss