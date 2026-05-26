import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import os

# --- 配置 ---
DELTA_RATIO = 0.5
SAMPLE_SIZE = 5000
PHY_THRESH = 0.4
MASK_THRESH = 0.4
OUTPUT_DIR = "outputs"

class NRSCalculator:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def calculate_soft_gamma(self, X, y):
        X_norm = self.scaler.fit_transform(X)
        y = y.flatten()
        X_sub, y_sub = X_norm, y
        print(f"[NRS] Calculating Soft Dependency on {len(X_sub)} samples...")
        
        std_y = np.std(y_sub)
        delta_y = std_y * DELTA_RATIO
        Ny_mask = (np.abs(y_sub[:, None] - y_sub[None, :]) <= delta_y)
        
        gamma_list = []
        n_feat = X_sub.shape[1]
        
        for i in range(n_feat):
            feat_col = X_sub[:, i].reshape(-1, 1)
            std_x = np.std(feat_col)
            delta_x = max(std_x * DELTA_RATIO, 1e-6)
            Nx_mask = (euclidean_distances(feat_col) <= delta_x)
            intersection = Nx_mask & Ny_mask
            purity = np.sum(intersection, axis=1) / (np.sum(Nx_mask, axis=1) + 1e-9)
            gamma_list.append(np.mean(purity))
            
        g = np.array(gamma_list)
        g_norm = (g - g.min()) / (g.max() - g.min() + 1e-9)
        print(f"    Gamma Range: [{g.min():.4f}, {g.max():.4f}] -> Normalized")
        return g_norm

def build_graphs(X, gamma):
    X_norm = MinMaxScaler().fit_transform(X).T
    n_nodes = X.shape[1]
    
    dist = euclidean_distances(X_norm)
    sigma = np.mean(dist)
    sim_mat = np.exp(- (dist**2) / (2 * sigma**2))
    
    adj_phy = np.zeros((n_nodes, n_nodes))
    mask = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j: 
                adj_phy[i, j] = 1.0
                continue
            sim = sim_mat[i, j]
            imp = (gamma[i] + gamma[j]) / 2
            w_phy = sim * (0.5 + imp)
            if sim > PHY_THRESH:
                adj_phy[i, j] = w_phy
            if sim * (1 + imp) > MASK_THRESH:
                mask[i, j] = 1.0
    return adj_phy, mask

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_csv("debutanizer.csv", header=None)
    df_x = df.iloc[:, 0:7]
    df_y = df.iloc[:, 7]
    
    nrs = NRSCalculator()
    gamma = nrs.calculate_soft_gamma(df_x.values, df_y.values)
    adj_phy, mask = build_graphs(df_x.values, gamma)
    
    np.save(os.path.join(OUTPUT_DIR, "adj_phy.npy"), adj_phy)
    np.save(os.path.join(OUTPUT_DIR, "causal_mask.npy"), mask)
    print(f"Saved: adj_phy shape {adj_phy.shape}, mask edges {np.sum(mask)}")
