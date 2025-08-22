from torch_geometric.datasets import QM9, ZINC
from torch_geometric.loader import DataLoader as GeoDataLoader

def load_qm9_dataset(root='./data/QM9'):
    dataset = QM9(root)
    loader = GeoDataLoader(dataset, batch_size=32, shuffle=True)
    return loader

def load_zinc_dataset(root='./data/ZINC'):
    dataset = ZINC(root)
    loader = GeoDataLoader(dataset, batch_size=32, shuffle=True)
    return loader
