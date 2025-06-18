import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pandas as pd

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        """
        Initialize the GAT model.

        Args:
            in_channels (int): Number of input features per node.
            hidden_channels (int): Number of hidden units per head in the first GAT layer.
            out_channels (int): Number of output features per node.
            heads (int): Number of attention heads in the first GAT layer.
        """
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT model.

        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, in_channels].
            edge_index (Tensor): Edge indices in COO format with shape [2, num_edges].

        Returns:
            Tensor: Node embeddings with shape [num_nodes, out_channels].
        """
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x


def train_gat_model(model, data, user_item_edges, num_users, num_items, epochs=250, batch_size=1024, lr=0.01, weight_decay=1e-4, device=None):
    """
    Train the GAT model using Bayesian Personalized Ranking (BPR) loss.

    Args:
        model (torch.nn.Module): The GAT model instance.
        data (Data): PyG data object containing features and edge_index.
        user_item_edges (Tensor): Tensor of user→item edges, shape [num_edges, 2].
        num_users (int): Number of users.
        num_items (int): Number of items.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for sampling edges.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        device (torch.device): Device to run training on.

    Returns:
        torch.nn.Module: Trained GAT model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = model(data.x.to(device), data.edge_index.to(device))
        user_emb = embeddings[:num_users]
        item_emb = embeddings[num_users:]

        # Sample batch indices from user-item edges
        idx = torch.randint(0, user_item_edges.size(0), (batch_size,))
        user_ids_batch = user_item_edges[idx][:, 0]
        pos_item_ids_batch = user_item_edges[idx][:, 1] - num_users
        neg_item_ids_batch = torch.randint(0, num_items, (len(user_ids_batch),), device=device)

        u_emb = user_emb[user_ids_batch]
        pos_emb = item_emb[pos_item_ids_batch]
        neg_emb = item_emb[neg_item_ids_batch]

        loss = bpr_loss(u_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model


def evaluate_gat_model(model, data, num_users, num_items, user_ids, idx2item, device=None, top_k=3):
    """
    Evaluate the trained GAT model and generate top-k recommendations.

    Args:
        model (torch.nn.Module): Trained GAT model.
        data (Data): PyG data object containing features and edge_index.
        num_users (int): Number of users.
        num_items (int): Number of items.
        user_ids (list): List or array of user IDs aligned with model indices.
        idx2item (dict): Mapping from item indices to item IDs.
        device (torch.device): Device for computation.
        top_k (int): Number of top recommendations per user.

    Returns:
        pd.DataFrame: DataFrame with columns 'CustomerID' and 'Recommendations'.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x.to(device), data.edge_index.to(device))
        user_emb = embeddings[:num_users].cpu()
        item_emb = embeddings[num_users:].cpu()

    scores = torch.matmul(user_emb, item_emb.T)
    _, topk_indices = torch.topk(scores, k=top_k, dim=1)

    recommended_items = [
        [idx2item[idx.item()] for idx in item_indices]
        for item_indices in topk_indices
    ]

    recommendation_df = pd.DataFrame({
        'CustomerID': user_ids,
        'Recommendations': recommended_items
    })

    return recommendation_df


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """
    Bayesian Personalized Ranking (BPR) loss function.

    Args:
        user_emb (Tensor): User embeddings, shape [batch_size, embedding_dim].
        pos_item_emb (Tensor): Positive item embeddings, shape [batch_size, embedding_dim].
        neg_item_emb (Tensor): Negative item embeddings, shape [batch_size, embedding_dim].

    Returns:
        Tensor: Scalar loss value.
    """
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)
    diff = torch.clamp(pos_scores - neg_scores, min=-20, max=20)
    loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
    return loss
