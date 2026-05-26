import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "outputs"

def compute_dag_constraint(adj):
    adj_sq = adj * adj 
    return torch.trace(torch.matrix_exp(adj_sq)) - adj.shape[0]

class VariationalCausalLearner(nn.Module):
    def __init__(self, n_nodes, mask):
        super().__init__()
        self.n_nodes = n_nodes
        self.mu = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        self.log_sigma = nn.Parameter(torch.ones(n_nodes, n_nodes) * -4.0)
        self.register_buffer('mask', torch.from_numpy(mask).float())
        nn.init.uniform_(self.mu, -0.01, 0.01)

    def forward(self, x):
        W = (self.mu + torch.randn_like(self.mu) * torch.exp(self.log_sigma)) * self.mask
        W = W.fill_diagonal_(0)
        return torch.matmul(x, W), W

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df = pd.read_csv("debutanizer.csv", header=None)
        df_x = df.iloc[:, 0:7]
        mask = np.load(os.path.join(OUTPUT_DIR, "causal_mask.npy"))
    except Exception as e:
        exit(f"Data missing: {e}")
    
    X = torch.tensor(StandardScaler().fit_transform(df_x.values), dtype=torch.float32).to(device)
    model = VariationalCausalLearner(7, mask).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    rho, alpha, h_prev = 1.0, 0.0, float('inf')

    print("--- Learning Causal Structure (Step 2) ---")
    for ep in range(1, 501):
        model.train()
        opt.zero_grad()
        x_recon, W = model(X)
        
        loss_mse = F.mse_loss(x_recon, X)
        h_val = compute_dag_constraint(model.mu * model.mask)
        loss_dag = (alpha * h_val + 0.5 * rho * h_val**2) if ep > 100 else 0
        loss_l1 = 0.008 * torch.norm(model.mu, 1)
        loss = loss_mse + loss_dag + loss_l1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        
        if ep % 50 == 0:
            print(f"Ep {ep:03d} | MSE: {loss_mse:.4f} | DAG: {h_val:.6e} | Rho: {rho:.1f}")
            if ep > 100 and h_val.item() > 0.25 * h_prev:
                rho = min(rho * 1.2, 500.0) 
            alpha += rho * h_val.item()
            h_prev = h_val.item()

    W_final = (model.mu * model.mask).detach().cpu().numpy()
    W_final[np.abs(W_final) < 0.01] = 0
    np.save(os.path.join(OUTPUT_DIR, "adj_causal_static.npy"), W_final)
    print("Step 2 Complete: Static Causal Graph Saved (7x7).")
