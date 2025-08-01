import torch
import torch.nn.functional as F
from fontTools.varLib.interpolatableHelpers import transform_from_stats
from torch.onnx.symbolic_opset9 import tensor
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected, to_networkx
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, SAGEConv, Node2Vec
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression


#set seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

#files loading
def load_graph(file_path):
    edge_list = pd.read_csv(file_path, sep=' ', header=None)
    edge_index = torch.tensor(edge_list.values.T, dtype=torch.long)

#convert to undirected and self-loops removing
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

#networkX graph creating for visualization
    G = nx.Graph()
    G.add_edges_from(edge_index.T.tolist())

    return edge_index, G

edge_index, nx_graph = load_graph('/Users/cynthia/PycharmProjects/graphMining/data/facebook/0.edges')
print(f"Number of nodes: {nx_graph.number_of_nodes()}")
print(f"Number of edges: {nx_graph.number_of_edges()}")

#node feature generation && edge splitting
def generate_node_features(G):
    num_nodes = G.number_of_nodes()
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, node_mapping)
#creating feature matrix degree and random vector
    degrees = np.array([val for (node, val) in G.degree()])
    degrees = degrees.reshape(-1, 1)
    random_vectors = np.random.rand(num_nodes, 32) #32dim
    features = np.hstack((degrees, random_vectors))
#normalize features
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    x = torch.tensor(features, dtype=torch.float)
    return x, node_mapping

#applying node relabeling and get features
x, node_mapping = generate_node_features(nx_graph)

#coverting edge_index using new node mapping
def remap_edge_index(edge_index, node_mapping):
    mapped_edges = []
    for src, dst in edge_index.T.tolist():
        if src in node_mapping and dst in node_mapping:
            mapped_edges.append([node_mapping[src], node_mapping[dst]])
    return torch.tensor(mapped_edges, dtype=torch.long).T

edge_index = remap_edge_index(edge_index, node_mapping)
edge_index = to_undirected(edge_index)

#PyG Data object creating
data = Data(x=x, edge_index=edge_index)

#spliyying into train/val/test with negative samples
transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True,
    split_labels=True
)
train_data, val_data, test_data = transform(data)

print(f"Train Pos Edges: {train_data.pos_edge_label_index.shape[1]}")
print(f"Validation Pos Edge: {val_data.pos_edge_label_index.shape[1]}")
print(f"Test Pos Edges: {test_data.pos_edge_label_index.shape[1]}")

#GCN Encoder
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

#Dot product decoder for link prediction
def decode(z, edge_label_index):
    src = z[edge_label_index[0]]
    dst = z[edge_label_index[1]]
    return (src * dst).sum(dim=1)

#Evaluation metrics
def evaluate(embeddings, edge_label_index, edge_label):
    preds = decode(embeddings, edge_label_index)
    preds = preds.detach().cpu()
    labels = edge_label.detach().cpu()

    probs = torch.sigmoid(preds)
    preds_binary = (probs > 0.5).float()

    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds_binary)
    precision = precision_score(labels, preds_binary)
    recall = recall_score(labels, preds_binary)
    ap = average_precision_score(labels, probs)

    return {
        'AUC': auc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'PR-AUC': ap
    }

#Generic training loop
def train_gnn_model(model, data, epochs=100, lr=0.01, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)

        edge_label_index = torch.cat([
            data.pos_edge_label_index,
            data.neg_edge_label_index
        ], dim=1)
        edge_label = torch.cat([
            torch.ones(data.pos_edge_label_index.shape[1]),
            torch.zeros(data.neg_edge_label_index.shape[1])
        ]).to(data.x.device)

        pred = decode(z, edge_label_index)
        loss = loss_fn(pred, edge_label)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    return model, z

#Evaluation wrapper
def evaluate_model(model, data, stage="Validation"):
    model.eval()
    z = model(data.x, data.edge_index)

#create 2 manually
    edge_label_index = torch.cat([
        data.pos_edge_label_index,
        data.neg_edge_label_index
    ], dim=1)

    edge_label = torch.cat([
        torch.ones(data.pos_edge_label_index.shape[1]),
        torch.zeros(data.neg_edge_label_index.shape[1])
    ]).to(data.x.device)

    metrics = evaluate(z, edge_label_index, edge_label)

    print(f"\n {stage} Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics

#Train and evaluate GCN
print("\n Training GCN...")
gcn_model = GCNEncoder(in_channels=train_data.num_node_features, hidden_channels=64)
gcn_model, z_train = train_gnn_model(gcn_model, train_data, epochs=100)

#Validation & Test performance
gcn_val_metrics = evaluate_model(gcn_model, val_data, stage="Validation")
gcn_test_metrics = evaluate_model(gcn_model, test_data, stage="Test")

### GraphSAGE Encoder
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

#Link Prediction Decoder (Dot Product)
def decode(z, edge_label_index):
    src = z[edge_label_index[0]]
    dst = z[edge_label_index[1]]
    return (src * dst).sum(dim=1)

#Evaluation Function
def evaluate(embeddings, edge_label_index, edge_label):
    preds = decode(embeddings, edge_label_index)
    probs = torch.sigmoid(preds).cpu().detach().numpy()
    labels = edge_label.cpu().detach().numpy()
    binary_preds = (probs > 0.5).astype(int)

    return {
        'AUC': roc_auc_score(labels, probs),
        'F1': f1_score(labels, binary_preds),
        'Precision': precision_score(labels, binary_preds),
        'Recall': recall_score(labels, binary_preds),
        'PR-AUC': average_precision_score(labels, probs)
    }

#Generic GNN Training
def train_gnn_model(model, data, epochs=100, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        edge_label_index = torch.cat([data.pos_edge_label_index, data.neg_edge_label_index], dim=1)
        edge_label = torch.cat([
            torch.ones(data.pos_edge_label_index.size(1)),
            torch.zeros(data.neg_edge_label_index.size(1))
        ]).to(device)

        preds = decode(z, edge_label_index)
        loss = loss_fn(preds, edge_label)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    return model, z.detach()

#Node2Vec Embedding + Logistic Regression
def train_node2vec_link_predictor(data, embedding_dim=64, walk_length=20, context_size=10, walks_per_node=10, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node2vec = Node2Vec(
        data.edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
        context_size=context_size, walks_per_node=walks_per_node, num_negative_samples=1,
        sparse=True
    ).to(device)

    loader = node2vec.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

    node2vec.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"[Node2Vec] Epoch {epoch} | Loss: {total_loss:.4f}")

    embeddings = node2vec.embedding.weight.detach().cpu()
    return embeddings

def evaluate_node2vec(embeddings, data):
    edge_label_index = torch.cat([data.pos_edge_label_index, data.neg_edge_label_index], dim=1)
    edge_label = torch.cat([
        torch.ones(data.pos_edge_label_index.shape[1]),
        torch.zeros(data.neg_edge_label_index.shape[1])
    ])

    src = embeddings[edge_label_index[0]]
    dst = embeddings[edge_label_index[1]]
    edge_features = (src * dst).numpy()
    clf = LogisticRegression().fit(edge_features, edge_label.numpy())
    preds = clf.predict_proba(edge_features)[:, 1]
    binary_preds = (preds > 0.5).astype(int)

    return {
        'AUC': roc_auc_score(edge_label, preds),
        'F1': f1_score(edge_label, binary_preds),
        'Precision': precision_score(edge_label, binary_preds),
        'Recall': recall_score(edge_label, binary_preds),
        'PR-AUC': average_precision_score(edge_label, preds)
    }

#PCA and t-SNE Visualization
def visualize_embeddings(embeddings, method="pca", title="Embeddings"):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("Unsupported method")

    reduced = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=8, alpha=0.7, c="skyblue")
    plt.title(f"{method.upper()} Projection of Node Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Link Prediction Visualization
def visualize_predicted_links(data, z, threshold=0.5, method='GCN'):
    """Visualize predicted positive links on top of original graph"""
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)
    # Decode predictions
    edge_label_index = torch.cat([data.pos_edge_label_index, data.neg_edge_label_index], dim=1)
    preds = torch.sigmoid((z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1))
    pred_labels = (preds > threshold).cpu().numpy()
    predicted_edges = edge_label_index[:, pred_labels == 1].T.tolist()

    # Draw original graph
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, node_size=20, edge_color='lightgray', node_color='skyblue', alpha=0.6)

    # Draw predicted links in red
    nx.draw_networkx_edges(G, pos,
                           edgelist=[(int(u), int(v)) for u, v in predicted_edges],
                           edge_color='red', width=1.2, alpha=0.8)

    plt.title(f"Predicted Links ({method})")
    plt.axis('off')
    plt.show()

#Embedding Visualization
def visualize_embeddings(z, method="t-SNE", model_name="GCN"):
    """Reduce embedding dimensions to 2D and visualize them"""
    z_np = z.detach().cpu().numpy()
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
    elif method.lower() == "t-sne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        raise ValueError("Choose either 'pca' or 't-sne'")
    z_2d = reducer.fit_transform(z_np)

    plt.figure(figsize=(8, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=20, c='steelblue', alpha=0.7)
    plt.title(f"{model_name} Node Embeddings ({method.upper()})")
    plt.axis('off')
    plt.show()

# Visualize predicted links from GCN
visualize_predicted_links(test_data, z_train, method="GCN")

# Visualize embeddings
visualize_embeddings(z_train, method="pca", model_name="GCN")
visualize_embeddings(z_train, method="t-sne", model_name="GCN")

#Evaluate all models on test data

# Evaluate GCN
print("\nEvaluating GCN on Test Data")
evaluate_model(gcn_model, test_data, stage="Test (GCN)")

# Evaluate GraphSAGE
print("\nTraining and Evaluating GraphSAGE...")
sage_model = GraphSAGE(in_channels=train_data.num_node_features, hidden_channels=64)
sage_model, z_sage = train_gnn_model(sage_model, train_data, epochs=100)
evaluate_model(sage_model, test_data, stage="Test (GraphSAGE)")

# Evaluate Node2Vec
print("\nTraining and Evaluating Node2Vec...")
node2vec_embeddings = train_node2vec_link_predictor(train_data)
node2vec_metrics = evaluate_node2vec(node2vec_embeddings, test_data)
print("\nNode2Vec Test Metrics:")
for k, v in node2vec_metrics.items():
    print(f"{k}: {v:.4f}")

#Link Prediction Visualizations

# 1. Visualize GCN Predicted Links
print("\n Visualizing GCN Predicted Links")
visualize_predicted_links(test_data, z_train, method="GCN")
visualize_embeddings(z_train, method="pca", model_name="GCN")
visualize_embeddings(z_train, method="t-sne", model_name="GCN")

#2. Visualize GraphSAGE Predicted Links
print("\n Visualizing GraphSAGE Predicted Links")
visualize_predicted_links(test_data, z_sage, method="GraphSAGE")
visualize_embeddings(z_sage, method="pca", model_name="GraphSAGE")
visualize_embeddings(z_sage, method="t-sne", model_name="GraphSAGE")

#3. Visualize Node2Vec Predicted Links
print("\n Visualizing Node2Vec Predicted Links")
# Since Node2Vec returns only embeddings (not a model), use directly
visualize_predicted_links(test_data, node2vec_embeddings, method="Node2Vec")
visualize_embeddings(node2vec_embeddings, method="pca", model_name="Node2Vec")
visualize_embeddings(node2vec_embeddings, method="t-sne", model_name="Node2Vec")