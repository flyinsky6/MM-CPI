import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from esm import pretrained
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import KFold
from itertools import product
from copy import deepcopy

# =========================
# 数据与特征函数（与你一致）
# =========================
def load_human_dataset(filepath):
    compounds, proteins, labels = [], [], []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                compounds.append(parts[0])  # 化合物SMILES
                proteins.append(parts[1])  # 蛋白质序列
                labels.append(int(parts[2]))  # 标签（0或1）
    return compounds, proteins, labels

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    edge_index = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    edge_index += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] for bond in mol.GetBonds()]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(atom_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

# 使用ESM模型将蛋白质序列转为嵌入（与你一致）
esm_model, alphabet = pretrained.esm2_t6_8M_UR50D()
esm_model.eval()
batch_converter = alphabet.get_batch_converter()

def sequence_to_embedding(sequence):
    sequence = sequence.upper().replace(" ", "")
    for char in sequence:
        if char not in alphabet.tok_to_idx:
            raise ValueError(f"Unsupported character '{char}' in protein sequence.")
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[6])
    token_embeddings = results["representations"][6]
    protein_embedding = token_embeddings.mean(1)
    return protein_embedding.squeeze(0)

def load_compound_features():
    ecfp = torch.tensor(np.load('MM-CPI/Feature/human化合物四种特征完整版/ECFP完整.npy'), dtype=torch.float32)
    extfp = torch.tensor(np.load('MM-CPI/Feature/human化合物四种特征完整版/Extfp完整.npy'), dtype=torch.float32)
    maccs = torch.tensor(np.load('MM-CPI/Feature/human化合物四种特征完整版/MACCS完整.npy'), dtype=torch.float32)
    mol2vec = torch.tensor(np.load('MM-CPI/Feature/human化合物四种特征完整版/mol2vec完整.npy'), dtype=torch.float32)
    return ecfp, extfp, maccs, mol2vec

def load_protein_features():
    aac = torch.tensor(pd.read_csv(
        'MM-CPI/Feature/human蛋白质三种特征完整版/human（AAC）完整.csv', header=None).values, dtype=torch.float32)
    ctdc = torch.tensor(pd.read_csv(
        'MM-CPI/Feature/human蛋白质三种特征完整版/human（CTDC）完整.csv', header=None).values, dtype=torch.float32)
    gaac = torch.tensor(pd.read_csv(
        'MM-CPI/Feature/human蛋白质三种特征完整版/human（GAAC）完整.csv', header=None).values, dtype=torch.float32)
    return aac, ctdc, gaac

# =========================
# Dataset 与模型（与你一致）
# =========================
class CompoundProteinDataset(Dataset):
    def __init__(self, compound_smiles, protein_sequences, labels, compound_features, protein_features):
        self.compound_smiles = compound_smiles
        self.protein_sequences = protein_sequences
        self.labels = labels
        self.compound_features = compound_features
        self.protein_features = protein_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        compound_graph = smiles_to_graph(self.compound_smiles[idx])
        protein_feat = sequence_to_embedding(self.protein_sequences[idx])

        # 拼接化合物特征
        compound_feat = torch.cat([f[idx].unsqueeze(0) for f in self.compound_features], dim=1)
        compound_feat = compound_feat.view(-1)

        # 拼接蛋白质特征
        protein_feat = torch.cat([protein_feat] + [f[idx] for f in self.protein_features], dim=0)
        protein_feat = protein_feat.view(-1)

        label = self.labels[idx]
        return compound_graph, compound_feat, protein_feat, torch.tensor(label, dtype=torch.float32)

class GNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, graph_embedding_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, graph_embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x

class InteractionPredictor(nn.Module):
    def __init__(self, graph_embedding_dim, compound_feat_dim, protein_dim, hidden_dim):
        super(InteractionPredictor, self).__init__()
        self.gnn = GNN(node_feature_dim=1, hidden_dim=64, graph_embedding_dim=graph_embedding_dim)
        self.fc1 = nn.Linear(graph_embedding_dim + compound_feat_dim + protein_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, compound_graph, compound_feat, protein_feat):
        graph_embedding = self.gnn(compound_graph)
        combined = torch.cat((graph_embedding, compound_feat, protein_feat), dim=1)
        x = self.relu(self.fc1(combined))
        output = torch.sigmoid(self.fc2(x))
        return output

# =========================
# 加载数据（路径与你一致）
# =========================
compounds, proteins, labels = load_human_dataset('MM-CPI/Dataset/human.txt')
ecfp, extfp, maccs, mol2vec = load_compound_features()
aac, ctdc, gaac = load_protein_features()

compound_features = [ecfp, extfp, maccs, mol2vec]
protein_features = [aac, ctdc, gaac]
dataset = CompoundProteinDataset(compounds, proteins, labels, compound_features, protein_features)

# =========================
# 80/20 划分（固定测试集）
# =========================
train_size_base = int(0.8 * len(dataset))
test_size = len(dataset) - train_size_base
train_base, test_dataset = random_split(dataset, [train_size_base, test_size])

# 公共维度与默认超参
graph_embedding_dim_default = 128
compound_feat_dim = ecfp.shape[1] + extfp.shape[1] + maccs.shape[1] + mol2vec.shape[1]
protein_dim = 320 + aac.shape[1] + ctdc.shape[1] + gaac.shape[1]
hidden_dim_default = 256
num_epochs = 50
log_file = "human.txt"

# =========================
# 训练循环（带早停，保持每个epoch三段日志格式不变）
# 监控指标：Val AUC；patience=20；恢复最佳权重
# 返回：该训练过程的最佳 Val AUC
# =========================
def train_val_test_epochs(model, train_loader, val_loader, test_loader,
                          optimizer, criterion_bce, num_epochs, log_fh,
                          patience=20, restore_best=True):
    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # ===== Train =====
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []
        for compound_graph, compound_feat, protein_feat, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(compound_graph, compound_feat, protein_feat)
            loss = criterion_bce(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().detach().numpy())

        train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        try:
            train_auc = roc_auc_score(all_labels, all_preds)
        except Exception:
            train_auc = 0.5
        train_precision = precision_score(all_labels, (np.array(all_preds) > 0.5).astype(int), zero_division=0)
        train_recall = recall_score(all_labels, (np.array(all_preds) > 0.5).astype(int), zero_division=0)

        log_fh.write(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}\n")
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")

        # ===== Val =====
        model.eval()
        all_labels, all_preds = [], []
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for compound_graph, compound_feat, protein_feat, labels in val_loader:
                outputs = model(compound_graph, compound_feat, protein_feat)
                loss = criterion_bce(outputs, labels.unsqueeze(1))
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted.squeeze() == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().detach().numpy())

        val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except Exception:
            val_auc = 0.5
        val_precision = precision_score(all_labels, (np.array(all_preds) > 0.5).astype(int), zero_division=0)
        val_recall = recall_score(all_labels, (np.array(all_preds) > 0.5).astype(int), zero_division=0)

        log_fh.write(
            f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}\n")
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        # ===== Test =====
        all_labels, all_preds = [], []
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for compound_graph, compound_feat, protein_feat, labels in test_loader:
                outputs = model(compound_graph, compound_feat, protein_feat)
                loss = criterion_bce(outputs, labels.unsqueeze(1))
                test_loss += loss.item()

                predicted = (outputs > 0.5).float()
                test_total += labels.size(0)
                test_correct += (predicted.squeeze() == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().detach().numpy())

        test_loss = test_loss / len(test_loader)
        test_accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        try:
            test_auc = roc_auc_score(all_labels, all_preds)
        except Exception:
            test_auc = 0.5
        test_precision = precision_score(all_labels, (np.array(all_preds) > 0.5).astype(int), zero_division=0)
        test_recall = recall_score(all_labels, (np.array(all_preds) > 0.5).astype(int), zero_division=0)

        log_fh.write(
            f"Epoch {epoch + 1}/{num_epochs} - Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}\n")
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

        # ===== Early Stopping on Val AUC =====
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            if restore_best:
                best_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                msg = f"[EarlyStopping] No improvement in Val AUC for {patience} epochs. Stop at epoch {epoch + 1}."
                print(msg); log_fh.write(msg + "\n")
                break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return best_val_auc

# =========================
# 网格搜索 + 5 折交叉验证（基于固定的 80% 训练基集合）
# =========================
param_grid = {
    "graph_embedding_dim": [128, 256],
    "hidden_dim": [256, 512],
    "lr": [1e-3, 5e-4]
}

criterion_bce = nn.BCELoss()
test_loader_fixed = DataLoader(test_dataset, batch_size=32, shuffle=False)

best_score = -1.0
best_params = None

with open(log_file, "a") as log:
    for graph_emb_dim, hidden_dim, lr in product(param_grid["graph_embedding_dim"],
                                                 param_grid["hidden_dim"],
                                                 param_grid["lr"]):
        print(f"\n=== GridSearch Start: graph_embedding_dim={graph_emb_dim}, hidden_dim={hidden_dim}, lr={lr} ===")
        log.write(f"\n=== GridSearch Start: graph_embedding_dim={graph_emb_dim}, hidden_dim={hidden_dim}, lr={lr} ===\n")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        base_indices = np.arange(len(train_base))
        fold_scores = []

        for fold_id, (tr_idx, va_idx) in enumerate(kf.split(base_indices), start=1):
            # 将 train_base 的相对索引映射回 dataset 的绝对索引
            train_abs_idx = [train_base.indices[i] for i in tr_idx]
            val_abs_idx   = [train_base.indices[i] for i in va_idx]

            train_subset = Subset(dataset, train_abs_idx)
            val_subset   = Subset(dataset, val_abs_idx)

            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader   = DataLoader(val_subset, batch_size=32, shuffle=False)

            model = InteractionPredictor(graph_emb_dim, compound_feat_dim, protein_dim, hidden_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            print(f"--- Fold {fold_id}/5 ---")
            log.write(f"--- Fold {fold_id}/5 ---\n")

            best_val_auc_fold = train_val_test_epochs(
                model, train_loader, val_loader, test_loader_fixed,
                optimizer, criterion_bce, num_epochs, log,
                patience=20, restore_best=True
            )
            fold_scores.append(best_val_auc_fold)

        mean_val_auc = float(np.mean(fold_scores)) if len(fold_scores) > 0 else -1.0
        print(f"=== GridSearch Result: graph_embedding_dim={graph_emb_dim}, hidden_dim={hidden_dim}, lr={lr} | 5-fold mean Val AUC={mean_val_auc:.4f} ===")
        log.write(f"=== GridSearch Result: graph_embedding_dim={graph_emb_dim}, hidden_dim={hidden_dim}, lr={lr} | 5-fold mean Val AUC={mean_val_auc:.4f} ===\n")

        if mean_val_auc > best_score:
            best_score = mean_val_auc
            best_params = {"graph_embedding_dim": graph_emb_dim, "hidden_dim": hidden_dim, "lr": lr}

print("\n>>> Best Params (by 5-fold mean Val AUC):", best_params, "Score:", round(best_score, 4))

# =========================
# 用最优超参按原流程再训练一次：
# 在 train_base 上再划分 1/8 为验证集
# =========================
train_size = len(train_base)
val_size = train_size // 8
train_size2 = train_size - val_size
final_train_subset, final_val_subset = random_split(train_base, [train_size2, val_size])

final_train_loader = DataLoader(final_train_subset, batch_size=32, shuffle=True)
final_val_loader   = DataLoader(final_val_subset, batch_size=32, shuffle=False)
final_test_loader  = test_loader_fixed

final_graph_emb_dim = best_params.get("graph_embedding_dim", graph_embedding_dim_default) if best_params else graph_embedding_dim_default
final_hidden_dim    = best_params.get("hidden_dim", hidden_dim_default) if best_params else hidden_dim_default
final_lr            = best_params.get("lr", 1e-3) if best_params else 1e-3

model = InteractionPredictor(final_graph_emb_dim, compound_feat_dim, protein_dim, final_hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=final_lr)
criterion_bce = nn.BCELoss()

with open(log_file, "a") as log:
    print("\n=== Final Training with Best Params ===")
    log.write("\n=== Final Training with Best Params ===\n")
    _ = train_val_test_epochs(
        model, final_train_loader, final_val_loader, final_test_loader,
        optimizer, criterion_bce, num_epochs, log,
        patience=20, restore_best=True
    )

# 保存模型
torch.save(model, "human.pth")