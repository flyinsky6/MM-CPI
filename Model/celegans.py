import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from esm import pretrained
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

# 加载human数据集
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


# 将SMILES转换为图结构
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


# 使用ESM模型将蛋白质序列转为嵌入
esm_model, alphabet = pretrained.esm2_t6_8M_UR50D()
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


# 加载化合物特征
def load_compound_features():
    ecfp = torch.tensor(np.load('D:/研究生/加特征数据/celegans化合物四种特征完整版/ECFP完整.npy'), dtype=torch.float32)
    extfp = torch.tensor(np.load('D:/研究生/加特征数据/celegans化合物四种特征完整版/Extfp完整.npy'), dtype=torch.float32)
    maccs = torch.tensor(np.load('D:/研究生/加特征数据/celegans化合物四种特征完整版/MACCS完整.npy'), dtype=torch.float32)
    mol2vec = torch.tensor(np.load('D:/研究生/加特征数据/celegans化合物四种特征完整版/mol2vec完整.npy'), dtype=torch.float32)
    return ecfp, extfp, maccs, mol2vec


# 加载蛋白质特征
def load_protein_features():
    aac = torch.tensor(pd.read_csv(
        'D:/研究生/加特征数据/celegans蛋白质三种特征完整版/celegans（AAC）完整.csv', header=None).values, dtype=torch.float32)
    ctdc = torch.tensor(pd.read_csv(
        'D:/研究生/加特征数据/celegans蛋白质三种特征完整版/celegans（CTDC）完整.csv', header=None).values, dtype=torch.float32)
    gaac = torch.tensor(pd.read_csv(
        'D:/研究生/加特征数据/celegans蛋白质三种特征完整版/celegans（GAAC）完整.csv', header=None).values, dtype=torch.float32)
    return aac, ctdc, gaac


# 定义Dataset类
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
        compound_feat = compound_feat.view(-1)  # 将三维特征展平为一维

        # 拼接蛋白质特征
        protein_feat = torch.cat([protein_feat] + [f[idx] for f in self.protein_features], dim=0)
        protein_feat = protein_feat.view(-1)  # 将三维特征展平为一维

        label = self.labels[idx]
        return compound_graph, compound_feat, protein_feat, torch.tensor(label, dtype=torch.float32)


# 定义GNN模型
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


# 定义预测模型
class InteractionPredictor(nn.Module):
    def __init__(self, graph_embedding_dim, compound_feat_dim, protein_dim, hidden_dim):
        super(InteractionPredictor, self).__init__()
        self.gnn = GNN(node_feature_dim=1, hidden_dim=64, graph_embedding_dim=graph_embedding_dim)
        self.fc1 = nn.Linear(graph_embedding_dim + compound_feat_dim + protein_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, compound_graph, compound_feat, protein_feat):
        # 获取图嵌入
        graph_embedding = self.gnn(compound_graph)

        # 拼接图嵌入、化合物特征和蛋白质特征
        combined = torch.cat((graph_embedding, compound_feat, protein_feat), dim=1)

        # 全连接层
        x = self.relu(self.fc1(combined))
        output = torch.sigmoid(self.fc2(x))
        return output


# 加载数据
compounds, proteins, labels = load_human_dataset('/dataset/celegans.txt')
ecfp, extfp, maccs, mol2vec = load_compound_features()
aac, ctdc, gaac = load_protein_features()

# 创建数据集
compound_features = [ecfp, extfp, maccs, mol2vec]
protein_features = [aac, ctdc, gaac]
dataset = CompoundProteinDataset(compounds, proteins, labels, compound_features, protein_features)

# 划分数据集
train_size = int(0.8 * len(dataset))  # 80%作为训练集
test_size = len(dataset) - train_size  # 剩余的作为测试集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 进一步拆分训练集为8份，其中7份作为训练集，1份作为验证集
train_size = len(train_dataset)
val_size = train_size // 8  # 每份占1/8
train_size = train_size - val_size  # 剩余部分为训练集

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 模型参数
graph_embedding_dim = 128
compound_feat_dim = ecfp.shape[1] + extfp.shape[1] + maccs.shape[1] + mol2vec.shape[1]
protein_dim = 320 + aac.shape[1] + ctdc.shape[1] + gaac.shape[1]
hidden_dim = 256

# 初始化模型
model = InteractionPredictor(graph_embedding_dim, compound_feat_dim, protein_dim, hidden_dim)
criterion_bce = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 日志文件路径
log_file = "celegans.txt"

# 训练模型
num_epochs = 50
with open(log_file, "w") as log:
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []

        # 训练过程
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

            # 计算AUC、ACC、Precision、Recall
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().detach().numpy())

        # 计算训练损失、准确率、AUC、Precision、Recall
        train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        train_auc = roc_auc_score(all_labels, all_preds)
        train_precision = precision_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        train_recall = recall_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        log.write(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}\n")
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")

        # 进行验证
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

                # 计算AUC、ACC、Precision、Recall
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().detach().numpy())

        # 计算验证损失、准确率、AUC、Precision、Recall
        val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        val_auc = roc_auc_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        val_recall = recall_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        log.write(
            f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}\n")
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        # 进行测试
        model.eval()
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

                # 计算AUC、ACC、Precision、Recall
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().detach().numpy())

        # 计算测试损失、准确率、AUC、Precision、Recall
        test_loss = test_loss / len(test_loader)
        test_accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        test_auc = roc_auc_score(all_labels, all_preds)
        test_precision = precision_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        test_recall = recall_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        log.write(
            f"Epoch {epoch + 1}/{num_epochs} - Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}\n")
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

# 保存模型
torch.save(model, "celegans.pth")