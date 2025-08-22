import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns


def plot_recon_scatter_plotly(x_real, x_gen, filename):
    x_real_flat = x_real.flatten().numpy()
    x_gen_flat = x_gen.flatten().numpy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_real_flat,
        y=x_gen_flat,
        mode='markers',
        marker=dict(size=5, opacity=0.4, color='rgba(99, 110, 250, 0.7)')
    ))

    min_val = min(x_real_flat.min(), x_gen_flat.min())
    max_val = max(x_real_flat.max(), x_gen_flat.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(dash='dash', color='red'),
        showlegend=False
    ))

    fig.update_layout(
        height=500,
        width=500,
        margin=dict(l=40, r=40, t=20, b=40),
        template='simple_white',
        xaxis=dict(title="Ground Truth", titlefont=dict(size=14), tickfont=dict(size=12), showgrid=False),
        yaxis=dict(title="Generated", titlefont=dict(size=14), tickfont=dict(size=12), showgrid=False),
        showlegend=False
    )

    fig.write_image(filename)


def plot_histogram_plotly(real, gen, filename, bins=50):
    real_vals = real.flatten().numpy()
    gen_vals = gen.flatten().numpy()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=real_vals,
        name='Real',
        opacity=0.5,
        nbinsx=bins,
        marker_color='rgba(99, 110, 250, 0.7)'
    ))

    fig.add_trace(go.Histogram(
        x=gen_vals,
        name='Generated',
        opacity=0.5,
        nbinsx=bins,
        marker_color='rgba(239, 85, 59, 0.7)'
    ))

    fig.update_layout(
        barmode='overlay',
        height=400,
        width=600,
        margin=dict(l=40, r=40, t=20, b=40),
        template='simple_white',
        xaxis=dict(title="Feature Value", titlefont=dict(size=14), tickfont=dict(size=12), showgrid=False),
        yaxis=dict(title="Count", titlefont=dict(size=14), tickfont=dict(size=12), showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=11))
    )

    fig.write_image(filename)

import matplotlib.pyplot as plt

def plot_recon_scatter_matplotlib(x_real, x_gen, filename):
    plt.figure(figsize=(5, 5))
    plt.scatter(x_real.flatten(), x_gen.flatten(), alpha=0.4, s=10, color='cornflowerblue')
    min_val = min(x_real.min(), x_gen.min())
    max_val = max(x_real.max(), x_gen.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Ground Truth", fontsize=13)
    plt.ylabel("Generated", fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_histogram_matplotlib(x_real, x_gen, filename, bins=50):
    plt.figure(figsize=(6, 4))
    plt.hist(x_real.flatten(), bins=bins, alpha=0.5, label='Real', color='cornflowerblue')
    plt.hist(x_gen.flatten(), bins=bins, alpha=0.5, label='Generated', color='tomato')
    plt.xlabel("Feature Value", fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()



def plot_grid_recon_and_hist(
    x_real_list,
    x_gen_list,
    model_names,
    dataset_name,
    filename,
    bins=50
):
    num_models = len(model_names)
    fig, axs = plt.subplots(num_models, 2, figsize=(10, 4 * num_models))

    for i, (x_real, x_gen, name) in enumerate(zip(x_real_list, x_gen_list, model_names)):
        x_real_flat = x_real.flatten()
        x_gen_flat = x_gen.flatten()

        # --- Scatter Plot ---
        ax_scatter = axs[i, 0] if num_models > 1 else axs[0]
        ax_scatter.scatter(x_real_flat, x_gen_flat, alpha=0.4, s=10, color='cornflowerblue')
        min_val = min(x_real_flat.min(), x_gen_flat.min())
        max_val = max(x_real_flat.max(), x_gen_flat.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax_scatter.set_xlabel("Ground Truth", fontsize=13)
        ax_scatter.set_ylabel("Generated", fontsize=13)
        ax_scatter.set_title(f"{dataset_name} - {name}", fontsize=14)
        ax_scatter.grid(False)

        # --- Histogram ---
        ax_hist = axs[i, 1] if num_models > 1 else axs[1]
        all_vals = np.concatenate([x_real_flat, x_gen_flat])
        range_vals = (all_vals.min(), all_vals.max())
        ax_hist.hist(x_real_flat, bins=bins, range=range_vals, alpha=0.5, label='Real', color='cornflowerblue')
        ax_hist.hist(x_gen_flat, bins=bins, range=range_vals, alpha=0.5, label='Generated', color='tomato')
        ax_hist.set_xlabel("Feature Value", fontsize=13)
        ax_hist.set_ylabel("Count", fontsize=13)
        ax_hist.legend(fontsize=11)
        ax_hist.grid(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()




def plot_biograph_pca_side_by_side(z_qm9, z_zinc, n_clusters=6, filename="biograph_pca_side_by_side.png"):
    # Convert to numpy
    z_qm9_np = z_qm9.cpu().numpy() if isinstance(z_qm9, torch.Tensor) else z_qm9
    z_zinc_np = z_zinc.cpu().numpy() if isinstance(z_zinc, torch.Tensor) else z_zinc

    # PCA
    pca = PCA(n_components=2)
    qm9_2d = pca.fit_transform(z_qm9_np)
    zinc_2d = pca.fit_transform(z_zinc_np)

    # Clustering
    k_qm9 = KMeans(n_clusters=n_clusters, random_state=0).fit(z_qm9_np)
    k_zinc = KMeans(n_clusters=n_clusters, random_state=0).fit(z_zinc_np)

    # Combine into DataFrame
    import pandas as pd
    df_qm9 = pd.DataFrame(qm9_2d, columns=["PC1", "PC2"])
    df_qm9["Cluster"] = k_qm9.labels_
    df_qm9["Dataset"] = "QM9"

    df_zinc = pd.DataFrame(zinc_2d, columns=["PC1", "PC2"])
    df_zinc["Cluster"] = k_zinc.labels_
    df_zinc["Dataset"] = "ZINC"

    df_all = pd.concat([df_qm9, df_zinc], axis=0, ignore_index=True)

    # Plot
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

    sns.scatterplot(data=df_qm9, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=10, ax=axs[0], legend=False)
    axs[0].set_xlabel("PC1", fontsize=12)
    axs[0].set_ylabel("PC2", fontsize=12)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("QM9", fontsize=13)
    axs[0].grid(False)

    sns.scatterplot(data=df_zinc, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=10, ax=axs[1], legend=False)
    axs[1].set_xlabel("PC1", fontsize=12)
    axs[1].set_ylabel("PC2", fontsize=12)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title("ZINC", fontsize=13)
    axs[1].grid(False)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

plot_biograph_pca_side_by_side(z_qm9_bio, z_zinc_bio)
import torch
import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# --- Utility: Subsample and convert to numpy ---
def preprocess_latent(z_tensor, max_points=2000, seed=42):
    np.random.seed(seed)
    z_np = z_tensor.cpu().numpy()
    if z_np.shape[0] > max_points:
        indices = np.random.choice(z_np.shape[0], max_points, replace=False)
        z_np = z_np[indices]
    return z_np

# --- Plotting function (shared for both t-SNE and UMAP) ---
def plot_latent_projection(z_qm9, z_zinc, method="TSNE", filename="latent_projection.png"):
    z_q = preprocess_latent(z_qm9)
    z_z = preprocess_latent(z_zinc)
    combined = np.vstack([z_q, z_z])
    labels = np.array(['QM9'] * len(z_q) + ['ZINC'] * len(z_z))

    if method == "TSNE":
        reducer = TSNE(n_components=2, perplexity=30, random_state=0)
    elif method == "UMAP":
        reducer = umap.UMAP(n_components=2, random_state=0)
    else:
        raise ValueError("method must be 'TSNE' or 'UMAP'")

    z_embedded = reducer.fit_transform(combined)

    # Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=z_embedded[:, 0], y=z_embedded[:, 1], hue=labels,
                    palette="Set2", s=10, alpha=0.7, edgecolor=None)
    plt.xlabel("Component 1", fontsize=13)
    plt.ylabel("Component 2", fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(title="", loc="upper right", fontsize=11)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# === QM9 ===
plot_grid_recon_and_hist(
    [x_qm9_real_v, x_qm9_real_gv, x_qm9_real],
    [x_qm9_gen_v, x_qm9_gen_gv, x_qm9_gen],
    ['VAE', 'GraphVAE', 'BioGraphVAE'],
    'QM9',
    filename="qm9_recon_grid.png"
)

# === ZINC ===
plot_grid_recon_and_hist(
    [x_zinc_real_v, x_zinc_real_gv, x_zinc_real],
    [x_zinc_gen_v, x_zinc_gen_gv, x_zinc_gen],
    ['VAE', 'GraphVAE', 'BioGraphVAE'],
    'ZINC',
    filename="zinc_recon_grid.png"
)
plot_recon_scatter_matplotlib(x_qm9_real, x_qm9_gen, "qm9_bio_scatter.png")
plot_histogram_matplotlib(x_qm9_real, x_qm9_gen, "qm9_bio_hist.png")
plot_latent_projection(z_qm9_bio, z_zinc_bio, method="TSNE", filename="tsne_biograph_qm9_zinc.png")
plot_latent_projection(z_qm9_bio, z_zinc_bio, method="UMAP", filename="umap_biograph_qm9_zinc.png")
