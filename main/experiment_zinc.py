import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# --- Load ZINC ---
def load_dataset():
    dataset = ZINC('./data/ZINC')
    return GeoDataLoader(dataset, batch_size=64, shuffle=True), dataset.num_features

# --- Evaluation Metrics ---

def safe_smoothness(x):
    if x.dtype != torch.float:
        x = x.float()
    if x.shape[1] < 2:
        return float('nan')
    return torch.mean((x[:, 1:] - x[:, :-1]) ** 2).item()

def evaluate(x_real, x_gen):
    mse = F.mse_loss(x_gen, x_real).item()
    diversity = torch.std(x_gen).item()
    smoothness = safe_smoothness(x_gen)
    return round(mse, 4), round(diversity, 4), round(smoothness, 4)


# --- VAE ---
class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, in_dim))

    def forward(self, x):
        x = x.float()
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# --- GraphVAE ---
class GraphVAE(nn.Module):
    def __init__(self, node_dim, latent_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_dim, 64)
        self.conv2 = GCNConv(64, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, node_dim))

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        z = self.conv2(x, edge_index, edge_weight=edge_weight)
        z_pool = global_mean_pool(z, batch)
        x_hat = self.decoder(z_pool)
        return x_hat, z_pool

# --- BioGraphVAE ---
class BioGraphVAE(nn.Module):
    def __init__(self, node_dim, latent_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_dim, 64)
        self.conv2 = GCNConv(64, latent_dim)
        self.funnel = nn.Parameter(torch.ones(latent_dim))  # learned funnel
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, node_dim))

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        z = self.conv2(x, edge_index, edge_weight=edge_weight)
        z_pool = global_mean_pool(z, batch)
        z_funnel = z_pool * self.funnel
        x_hat = self.decoder(z_funnel)
        return x_hat, z_funnel

# --- GAN ---
class Generator(nn.Module):
    def __init__(self, latent_dim=64, out_dim=11):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, out_dim))

    def forward(self, z): return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, in_dim=11):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x): return self.model(x)

# --- Training Loop (VAE, GraphVAE, BioGraphVAE) ---
def run_experiment(loader, in_dim, model_name='VAE'):
    if model_name == 'VAE':
        model = VAE(in_dim).cuda()
    elif model_name == 'Graph':
        model = GraphVAE(in_dim).cuda()
    elif model_name == 'BioGraph':
        model = BioGraphVAE(in_dim).cuda()
    else:
        raise ValueError("Unknown model")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    real_all, gen_all, latent_all = [], [], []

    for batch in loader:
        batch = batch.to('cuda')

        if model_name == 'VAE':
            x = batch.x.float()
            x_hat, z = model(x)
            x_real = x
        else:
            x = batch.x.float()
            x_hat, z = model(batch)
            x_real = global_mean_pool(x, batch.batch)

        optimizer.zero_grad()
        loss = F.mse_loss(x_hat, x_real)
        loss.backward()
        optimizer.step()

        real_all.append(x_real.detach().cpu())
        gen_all.append(x_hat.detach().cpu())
        latent_all.append(z.detach().cpu())

    x_real = torch.cat(real_all, dim=0)
    x_gen = torch.cat(gen_all, dim=0)
    z_latents = torch.cat(latent_all, dim=0)
    min_len = min(x_real.size(0), x_gen.size(0), z_latents.size(0))

    metrics = evaluate(x_real[:min_len], x_gen[:min_len])
    return metrics, x_real[:min_len], x_gen[:min_len], z_latents[:min_len]




# --- GAN Training ---
def train_gan(loader, in_dim):
    gen = Generator(out_dim=in_dim).cuda()
    disc = Discriminator(in_dim=in_dim).cuda()
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    real_all, gen_all = [], []

    for batch in loader:
        batch = batch.to('cuda')
        x_real = batch.x.float()  # ✅ FIXED: Extract features only

        B = x_real.size(0)
        z = torch.randn(B, 64).cuda()
        x_fake = gen(z)

        # Train Discriminator
        opt_d.zero_grad()
        loss_d_real = criterion(disc(x_real), torch.ones(B, 1).cuda())
        loss_d_fake = criterion(disc(x_fake.detach()), torch.zeros(B, 1).cuda())
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        opt_d.step()

        # Train Generator
        opt_g.zero_grad()
        loss_g = criterion(disc(x_fake), torch.ones(B, 1).cuda())
        loss_g.backward()
        opt_g.step()

        real_all.append(x_real.detach().cpu())
        gen_all.append(x_fake.detach().cpu())

    real_all = torch.cat(real_all, dim=0)
    gen_all = torch.cat(gen_all, dim=0)
    min_len = min(real_all.size(0), gen_all.size(0))
    return evaluate(real_all[:min_len], gen_all[:min_len])



# --- Run Benchmark on ZINC ---
def benchmark_zinc():
    print("\n🔬 Benchmarking on ZINC dataset")
    loader, in_dim = load_dataset()

    vae_metrics = run_experiment(loader, in_dim, model_name='VAE')
    graph_metrics = run_experiment(loader, in_dim, model_name='Graph')
    gan_metrics = train_gan(loader, in_dim)
    bio_metrics = run_experiment(loader, in_dim, model_name='BioGraph')

    print("\n📊 Final Metrics on ZINC:")
    print(f"{'Model':<15} {'MSE':<10} {'Diversity':<12} {'Smoothness'}")
    print("-" * 45)
    print(f"{'VAE':<15} {vae_metrics[0]:<10} {vae_metrics[1]:<12} {vae_metrics[2]}")
    print(f"{'GAN':<15} {gan_metrics[0]:<10} {gan_metrics[1]:<12} {gan_metrics[2]}")
    print(f"{'GraphVAE':<15} {graph_metrics[0]:<10} {graph_metrics[1]:<12} {graph_metrics[2]}")
    print(f"{'BioGraphVAE':<15} {bio_metrics[0]:<10} {bio_metrics[1]:<12} {bio_metrics[2]}")

# ✅ RUN IT
def main():
  benchmark_zinc()

if __name__ == '__main__':
  main()
