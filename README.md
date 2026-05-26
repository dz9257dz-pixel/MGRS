# MGRS-Net — Debutanizer 验证实验

## 三阶段流程

```
NR1.py  →  NR2.py  →  NR3.py
构图      因果学习    模型训练 + 灵敏度分析
```

## 运行方式

```bash
# Step 1: 构图（输出 adj_phy.npy, causal_mask.npy）
python NR1.py

# Step 2: 因果学习（输出 adj_causal_static.npy）
python NR2.py

# Step 3: 灵敏度分析（输出 Excel）
python NR3.py
```

## 输入数据

| 文件 | 说明 |
|------|------|
| `debutanizer.csv` | Debutanizer 过程数据，7维输入 + 1维输出 |

## 中间输出（`outputs/` 文件夹）

| 文件 | 来源 | 说明 |
|------|------|------|
| `adj_phy.npy` | NR1.py | 物理邻接矩阵（奖惩加权） |
| `causal_mask.npy` | NR1.py | 因果骨架掩码 |
| `adj_causal_static.npy` | NR2.py | 7×7 静态因果邻接矩阵 |
| `sensitivity_analysis_dynweight.xlsx` | NR3.py | 灵敏度分析结果（Raw_Data + Summary） |

## 所需环境

```
torch >= 2.0
torch_geometric
scikit-learn
numpy
pandas
openpyxl
```

## box plot

outputs/sensitivity_analysis_dynweight.png

