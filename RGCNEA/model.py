import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from torch.cuda.amp import autocast, GradScaler

torch.cuda.empty_cache()


class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RGCN, self).__init__()
        self.embedding = torch.nn.Embedding(num_entities, embedding_dim)
        self.conv1 = RGCNConv(embedding_dim, embedding_dim, num_relations, num_bases=30)
        self.conv2 = RGCNConv(embedding_dim, embedding_dim, num_relations, num_bases=30)

    def forward(self, edge_index, edge_type):
        x = self.embedding.weight
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x


def train_rgcn(data, num_entities, num_relations, embedding_dim=128, num_epochs=50, batch_size=5000,
               accumulation_steps=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = RGCN(num_entities, num_relations, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scaler = GradScaler()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for i in range(0, data.num_edges, batch_size):
            edge_index = data.edge_index[:, i:i + batch_size].to(device)
            edge_type = data.edge_type[i:i + batch_size].to(device)

            with autocast():
                out = model(edge_index, edge_type)
                loss = F.mse_loss(out, model.embedding.weight)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i // batch_size + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}')

        torch.cuda.empty_cache()

    return model





