import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9, ZINC
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np

# --- Loaders ---
def load_dataset(name='QM9'):
    dataset = QM9('./data/QM9') if name == 'QM9' else ZINC('./data/ZINC')
    return GeoDataLoader(dataset, batch_size=64, shuffle=True), dataset.num_features

# --- Metrics ---
def evaluate(x_real, x_gen):
    mse = F.mse_loss(x_gen, x_real).item()
    diversity = torch.std(x_gen).item()
    smoothness = torch.mean((x_gen[:, 1:] - x_gen[:, :-1])**2).item()
    return round(mse, 4), round(diversity, 4), round(smoothness, 4)

# --- VAE ---
class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, in_dim))

    def forward(self, x):
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        z = self.conv2(x, edge_index)
        z_pool = global_mean_pool(z, batch)
        x_hat = self.decoder(z_pool)
        return x_hat, z_pool

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

#-------BIOGRAPHVAE--------
class BioGraphVAE(nn.Module):
    def __init__(self, node_dim, latent_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_dim, 64)
        self.conv2 = GCNConv(64, latent_dim)
        self.funnel = nn.Parameter(torch.ones(latent_dim))  # learned funnel attention
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, node_dim))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        z = self.conv2(x, edge_index)
        z_pool = global_mean_pool(z, batch)  # shape [B, latent_dim]

        # Apply funneling (reaction center inspired)
        z_funnel = z_pool * self.funnel  # element-wise attention modulation
        x_hat = self.decoder(z_funnel)
        return x_hat, z_funnel


# --- Run Experiment (VAE / GraphVAE) ---
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

    real_all, gen_all, z_all = [], [], []

    for batch in loader:
        batch = batch.to('cuda')
        if model_name == 'VAE':
            x = batch.x.float()
            x_hat, z = model(x)
        else:
            x = global_mean_pool(batch.x.float(), batch.batch)
            x_hat, z = model(batch)

        loss = F.mse_loss(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        real_all.append(x.detach().cpu())
        gen_all.append(x_hat.detach().cpu())
        z_all.append(z.detach().cpu())

    x_real = torch.cat(real_all, dim=0)
    x_gen = torch.cat(gen_all, dim=0)
    z_latent = torch.cat(z_all, dim=0)

    metrics = evaluate(x_real[:len(x_gen)], x_gen)
    return metrics, x_real, x_gen, z_latent



# --- GAN Training ---
def train_gan(loader, in_dim):
    gen = Generator(out_dim=in_dim).cuda()
    disc = Discriminator(in_dim=in_dim).cuda()
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    real_all, gen_all, z_all = [], [], []

    for batch in loader:
        batch = batch.to('cuda')
        x_real = batch.x.float()
        B = x_real.size(0)
        z = torch.randn(B, 64).cuda()
        x_fake = gen(z)

        # Discriminator
        opt_d.zero_grad()
        loss_d = criterion(disc(x_real), torch.ones(B, 1).cuda()) + \
                 criterion(disc(x_fake.detach()), torch.zeros(B, 1).cuda())
        loss_d.backward()
        opt_d.step()

        # Generator
        opt_g.zero_grad()
        loss_g = criterion(disc(x_fake), torch.ones(B, 1).cuda())
        loss_g.backward()
        opt_g.step()

        real_all.append(x_real.detach().cpu())
        gen_all.append(x_fake.detach().cpu())
        z_all.append(z.detach().cpu())

    x_real = torch.cat(real_all, dim=0)
    x_fake = torch.cat(gen_all, dim=0)
    z_noise = torch.cat(z_all, dim=0)

    metrics = evaluate(x_real[:len(x_fake)], x_fake)
    return metrics, x_real, x_fake, z_noise


def benchmark_all(dataset_name='QM9'):
    print(f"\n🔬 Benchmarking on {dataset_name} dataset")
    loader, in_dim = load_dataset(dataset_name)

    vae_metrics, x_vae, xhat_vae, z_vae = run_experiment(loader, in_dim, model_name='VAE')
    graph_metrics, x_graph, xhat_graph, z_graph = run_experiment(loader, in_dim, model_name='Graph')
    gan_metrics, x_gan, xhat_gan, z_gan = train_gan(loader, in_dim)
    bio_metrics, x_bio, xhat_bio, z_bio = run_experiment(loader, in_dim, model_name='BioGraph')

    print("\n📊 Final Metrics:")
    print(f"{'Model':<15} {'MSE':<10} {'Diversity':<12} {'Smoothness'}")
    print("-" * 45)
    print(f"{'VAE':<15} {vae_metrics[0]:<10} {vae_metrics[1]:<12} {vae_metrics[2]}")
    print(f"{'GAN':<15} {gan_metrics[0]:<10} {gan_metrics[1]:<12} {gan_metrics[2]}")
    print(f"{'GraphVAE':<15} {graph_metrics[0]:<10} {graph_metrics[1]:<12} {graph_metrics[2]}")
    print(f"{'BioGraphVAE':<15} {bio_metrics[0]:<10} {bio_metrics[1]:<12} {bio_metrics[2]}")


# --- Run Benchmarks ---
def main()
  benchmark_all('QM9')

if __name_ = "__main__":
  main()
  
