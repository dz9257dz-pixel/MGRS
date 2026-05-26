import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyG_DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import os
import random

OUTPUT_DIR = "outputs"

class MultiGraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_phy': return self.num_nodes 
        return super().__inc__(key, value, *args, **kwargs)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DynamicGraphLearner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.wq = nn.Linear(dim, 64)
        self.wk = nn.Linear(dim, 64)
    def forward(self, h):
        Q = self.wq(h)
        K = self.wk(h)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / 8.0
        return torch.sigmoid(attn)

class DynamicCausalDualGCN(nn.Module):
    def __init__(self, num_nodes=7, hidden_dim=64, lstm_dim=32, dyn_weight=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.dyn_weight = dyn_weight
        self.lstm = nn.LSTM(1, lstm_dim, batch_first=True)
        self.dyn_learner = DynamicGraphLearner(lstm_dim)
        self.conv_phy = GCNConv(lstm_dim, hidden_dim)
        self.W_causal = nn.Linear(lstm_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim*2, 1)
        self.reg = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, static_adj, mask, edge_index_phy, batch):
        x_in = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x_in)
        h = h_n.squeeze(0) 
        bs = batch.max().item() + 1
        h_grouped = h.view(bs, self.num_nodes, -1) 
        
        adj_dyn = self.dyn_learner(h_grouped)
        adj_causal = (static_adj.unsqueeze(0) + self.dyn_weight * adj_dyn) * mask.unsqueeze(0)
        h_trans = self.W_causal(h_grouped)
        h_c = torch.bmm(adj_causal, h_trans)
        h_c = F.relu(h_c).view(-1, h_c.size(-1))
        
        h_p = F.relu(self.conv_phy(h, edge_index_phy))
        
        alpha = torch.sigmoid(self.gate(torch.cat([h_c, h_p], dim=1)))
        h_final = alpha * h_c + (1 - alpha) * h_p
        
        pool_final = global_mean_pool(h_final, batch)
        return self.reg(pool_final)


def adj_to_index(adj):
    src, dst = np.where(adj > 0.01)
    return torch.from_numpy(np.array([src, dst])).long()

def create_win(X, y, w=32):
    Xs, ys = [], []
    for i in range(w, len(X)):
        Xs.append(X[i-w:i].T)
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def evaluate(model, loader, ts_static, ts_mask, device):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            out = model(b.x, ts_static, ts_mask, b.edge_index_phy, b.batch)
            ps.append(out.cpu().numpy())
            ts.append(b.y.cpu().numpy())
    
    y_pred = np.concatenate(ps)
    y_true = np.concatenate(ts)
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return r2, rmse, mae, mape, mse, y_true, y_pred


def train_and_evaluate(dyn_weight, seed, model_template_cpu, device,
                       loader_tr, loader_val, loader_te,
                       ts_static, ts_mask, EPOCH=200):
    import copy
    set_seed(seed)
    
    model = copy.deepcopy(model_template_cpu).to(device)
    model.dyn_weight = dyn_weight
    set_seed(seed)
    
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.MSELoss()
    
    best_val_r2 = -100
    best_model_path = os.path.join(OUTPUT_DIR, f"best_w{dyn_weight:.1f}_s{seed}.pth")
    
    for ep in range(EPOCH):
        model.train()
        loss_sum = 0
        for b in loader_tr:
            b = b.to(device)
            opt.zero_grad()
            out = model(b.x, ts_static, ts_mask, b.edge_index_phy, b.batch)
            loss = crit(out, b.y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        
        model.eval()
        val_r2, _, _, _, _, _, _ = evaluate(model, loader_val, ts_static, ts_mask, device)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), best_model_path)
    
    model.load_state_dict(torch.load(best_model_path))
    test_r2, test_rmse, test_mae, test_mape, test_mse, _, _ = evaluate(
        model, loader_te, ts_static, ts_mask, device)
    
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
    
    return test_r2, test_rmse, test_mae, test_mape, test_mse


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    WEIGHT_VALUES = [round(i * 0.1, 1) for i in range(1, 10)]
    N_SEEDS = 5
    BASE_SEED = 42
    EPOCH = 200
    
    try:
        adj_phy = np.load(os.path.join(OUTPUT_DIR, "adj_phy.npy"))
        adj_static = np.load(os.path.join(OUTPUT_DIR, "adj_causal_static.npy"))
        mask = np.load(os.path.join(OUTPUT_DIR, "causal_mask.npy"))
        df = pd.read_csv("debutanizer.csv", header=None)
        X_raw = df.iloc[:, 0:7].values.astype(np.float32)
        y_raw = df.iloc[:, 7].values.astype(np.float32)
    except Exception as e:
        exit(f"Missing files or errors: {e}")
    
    ei_phy = adj_to_index(adj_phy).cpu()
    ts_static = torch.FloatTensor(adj_static).to(device)
    ts_mask = torch.FloatTensor(mask).to(device)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    X_seq, y_seq = create_win(X, y_raw, w=32)
    
    data_list = []
    for i in range(len(X_seq)):
        d = MultiGraphData(x=torch.tensor(X_seq[i]), y=torch.tensor(y_seq[i]).view(1,1),
                           edge_index_phy=ei_phy)
        data_list.append(d)
    
    set_seed(BASE_SEED)
    train_data, temp_data = train_test_split(data_list, test_size=0.3, random_state=BASE_SEED)
    val_data, test_data = train_test_split(temp_data, test_size=0.3, random_state=BASE_SEED)
    
    print(f"Data Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    loader_tr = PyG_DataLoader(train_data, batch_size=128, shuffle=False)
    loader_val = PyG_DataLoader(val_data, batch_size=128, shuffle=False)
    loader_te = PyG_DataLoader(test_data, batch_size=128, shuffle=False)
    
    model_template_cpu = DynamicCausalDualGCN(num_nodes=7, dyn_weight=0.0)
    
    all_results = {w: [] for w in WEIGHT_VALUES}
    total_jobs = len(WEIGHT_VALUES) * N_SEEDS
    job_count = 0
    
    print(f"\n{'='*70}")
    print(f"  灵敏度分析: {len(WEIGHT_VALUES)} 权重 x {N_SEEDS} 种子 = {total_jobs} 次实验")
    print(f"  每轮 {EPOCH} epoch, 总计 {total_jobs * EPOCH} epoch")
    print(f"{'='*70}")
    
    for w in WEIGHT_VALUES:
        for s in range(N_SEEDS):
            seed = BASE_SEED + s
            job_count += 1
            print(f"\n[{job_count}/{total_jobs}] dyn_weight={w:.1f}, seed={seed}")
            
            r2, rmse, mae, mape, mse = train_and_evaluate(
                w, seed, model_template_cpu, device,
                loader_tr, loader_val, loader_te,
                ts_static, ts_mask, EPOCH=EPOCH)
            
            all_results[w].append({'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'MSE': mse})
            print(f"  -> R2={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | MAPE={mape:.2f}%")
    
    print(f"\n\n{'='*95}")
    print(f"  灵敏度分析结果 (dyn_weight 对预测性能的影响)")
    print(f"  (每组 N={N_SEEDS} 次重复, 均值 +/- 标准差)")
    print(f"{'='*95}")
    print(f"  {'Weight':>8} | {'R2':>8} | {'RMSE':>8} | {'MAE':>8} | {'MAPE(%)':>8}")
    print(f"{'-'*60}")
    
    best_mean_r2 = -1e9
    best_w = None
    summary = []
    
    for w in WEIGHT_VALUES:
        r2s = np.array([r['R2'] for r in all_results[w]])
        rmses = np.array([r['RMSE'] for r in all_results[w]])
        maes = np.array([r['MAE'] for r in all_results[w]])
        mapes = np.array([r['MAPE'] for r in all_results[w]])
        
        m_r2, s_r2 = r2s.mean(), r2s.std()
        m_rmse, s_rmse = rmses.mean(), rmses.std()
        m_mae, s_mae = maes.mean(), maes.std()
        m_mape, s_mape = mapes.mean(), mapes.std()
        
        summary.append((w, m_r2, s_r2, m_rmse, s_rmse, m_mae, s_mae, m_mape, s_mape))
        print(f"  {w:>8.1f} | {m_r2:>8.4f}+-{s_r2:.4f} | {m_rmse:>8.4f}+-{s_rmse:.4f} | {m_mae:>8.4f}+-{s_mae:.4f} | {m_mape:>8.2f}+-{s_mape:.2f}")
        
        if m_r2 > best_mean_r2:
            best_mean_r2 = m_r2
            best_w = w
    
    print(f"{'-'*60}")
    print(f"  最优: weight={best_w:.1f}, 平均 R2={best_mean_r2:.4f}")
    
    # 保存 Excel
    excel_path = os.path.join(OUTPUT_DIR, "sensitivity_analysis_dynweight.xlsx")
    rows_raw = []
    for w in WEIGHT_VALUES:
        for s in range(N_SEEDS):
            rows_raw.append({
                'dyn_weight': w,
                'seed': BASE_SEED + s,
                'R2': all_results[w][s]['R2'],
                'RMSE': all_results[w][s]['RMSE'],
                'MAE': all_results[w][s]['MAE'],
                'MAPE': all_results[w][s]['MAPE'],
                'MSE': all_results[w][s]['MSE']
            })
    
    rows_summary = []
    for w, m_r2, s_r2, m_rmse, s_rmse, m_mae, s_mae, m_mape, s_mape in summary:
        rows_summary.append({
            'dyn_weight': w,
            'R2_mean': m_r2, 'R2_std': s_r2,
            'RMSE_mean': m_rmse, 'RMSE_std': s_rmse,
            'MAE_mean': m_mae, 'MAE_std': s_mae,
            'MAPE_mean': m_mape, 'MAPE_std': s_mape
        })
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        pd.DataFrame(rows_raw).to_excel(writer, sheet_name='Raw_Data', index=False)
        pd.DataFrame(rows_summary).to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\n  Excel saved to {excel_path}")
    print("  Done.")
