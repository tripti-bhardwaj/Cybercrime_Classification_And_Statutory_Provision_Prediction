import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
import warnings
warnings.filterwarnings("ignore")


def pool_tweet_embeddings(word_node_embs, tweet_node_indices):
    reps = []
    for idxs in tweet_node_indices:
        if len(idxs) == 0:
            reps.append(
                torch.zeros(
                    word_node_embs.size(1),
                    device=word_node_embs.device
                )
            )
        else:
            reps.append(word_node_embs[idxs].mean(dim=0))
    return torch.stack(reps)


class TweetGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class WordGNN(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.conv1 = GCNConv(in_dim, out_dim)
        self.conv2 = GCNConv(out_dim, out_dim)

    def forward(self, data, tweet_node_indices=None):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        if tweet_node_indices is None:
            return x

        tweet_reps = []
        for idxs in tweet_node_indices:
            if len(idxs) == 0:
                tweet_reps.append(
                    torch.zeros(x.size(1), device=x.device)
                )
            else:
                tweet_reps.append(x[idxs].mean(dim=0))

        return torch.stack(tweet_reps)


class CategoryClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)