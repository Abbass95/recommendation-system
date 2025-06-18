import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd

class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)
    diff = torch.clamp(pos_scores - neg_scores, min=-20, max=20)
    loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
    return loss

def sample_training_batch(user_item_edges, num_items, batch_size=1024):
    idx = torch.randint(0, user_item_edges.size(0), (batch_size,))
    user_batch = user_item_edges[idx][:, 0]
    pos_item_batch = user_item_edges[idx][:, 1]
    neg_item_batch = torch.randint(0, num_items, (batch_size,))
    return user_batch, pos_item_batch, neg_item_batch

def train(model, data, user_item_edges, num_users, num_items, epochs=400, lr=1e-4, batch_size=1024):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = model(data.x, data.edge_index)
        user_emb = embeddings[:num_users]
        item_emb = embeddings[num_users:]

        user_batch, pos_item_batch, neg_item_batch = sample_training_batch(user_item_edges, num_items, batch_size)

        # Adjust positive items to zero-based indexing for items (items start after users)
        pos_item_batch = pos_item_batch - num_users

        u_emb = user_emb[user_batch]
        pos_emb = item_emb[pos_item_batch]
        neg_emb = item_emb[neg_item_batch]

        loss = bpr_loss(u_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

def get_embeddings(model, data, num_users, num_items):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    user_embeddings = embeddings[:num_users]
    item_embeddings = embeddings[num_users:]
    return user_embeddings, item_embeddings

def recommend_top_k(user_embeddings, item_embeddings, idx2item, user_ids, k=3):
    """
    Compute top-k recommendations for each user.

    Returns a DataFrame with CustomerID and Recommendations (list of StockCodes).
    """
    scores = torch.matmul(user_embeddings, item_embeddings.T)  # [num_users, num_items]
    _, topk_indices = torch.topk(scores, k=k, dim=1)

    recommended_items = [
        [idx2item[idx.item()] for idx in user_topk]
        for user_topk in topk_indices
    ]

    recommendation_df = pd.DataFrame({
        'CustomerID': user_ids,
        'Recommendations': recommended_items
    })

    return recommendation_df
