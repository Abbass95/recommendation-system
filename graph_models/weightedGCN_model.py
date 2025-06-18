import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd

def build_weighted_edge_index(df, user2idx, item2idx, num_users):
    """
    Construct edge_index and edge_weight tensors from DataFrame with Quantity as weight.
    """
    edge_index_list = []
    edge_weights = []

    for _, row in df.iterrows():
        if pd.notna(row['CustomerID']) and row['StockCode'] in item2idx:
            user_idx = user2idx[row['CustomerID']]
            item_idx = item2idx[row['StockCode']]

            # Add edge: user -> item (shift item index by num_users)
            edge_index_list.append([user_idx, num_users + item_idx])

            # Use Quantity as edge weight (can be any other numeric measure)
            edge_weights.append(row['Quantity'])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    # Normalize or scale edge weights if desired
    edge_weight = torch.log1p(edge_weight)  # Log-scale to reduce skew

    return edge_index, edge_weight

class WeightedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

def train_weighted_gcn(model, data_x, edge_index, edge_weight, user_item_edges, num_users, num_items,
                       epochs=400, lr=1e-3, batch_size=1024):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = model(data_x, edge_index, edge_weight)
        user_emb = embeddings[:num_users]
        item_emb = embeddings[num_users:]

        idx = torch.randint(0, user_item_edges.size(0), (batch_size,))
        user_batch = user_item_edges[idx][:, 0]
        pos_item_batch = user_item_edges[idx][:, 1] - num_users  # Adjust item indices

        neg_item_batch = torch.randint(0, num_items, (batch_size,))

        u_emb = user_emb[user_batch]
        pos_emb = item_emb[pos_item_batch]
        neg_emb = item_emb[neg_item_batch]

        loss = bpr_loss(u_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

def recommend_top_k_from_embeddings(user_emb, item_emb, idx2item, user_ids, k=3):
    scores = torch.matmul(user_emb, item_emb.T)  # [num_users x num_items]
    _, topk_indices = torch.topk(scores, k=k, dim=1)

    recommendations = [
        [idx2item[idx.item()] for idx in row]
        for row in topk_indices
    ]

    return pd.DataFrame({'CustomerID': user_ids, 'Recommendations': recommendations})
