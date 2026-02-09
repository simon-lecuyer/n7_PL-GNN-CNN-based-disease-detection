"""
PyTorch Datasets pour charger les données preprocessées.

Classes:
    - DiseaseDetectionDataset: Dataset de base pour CNN ou GNN
    - TemporalDiseaseDataset: Dataset pour séquences temporelles
    
Usage:
    from utils.datasets import DiseaseDetectionDataset
    from torch.utils.data import DataLoader
    
    # Dataset CNN
    dataset = DiseaseDetectionDataset(
        "data/processed/processed_XXX/datasets/cnn/train.json",
        format="cnn"
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Dataset GNN
    dataset = DiseaseDetectionDataset(
        "data/processed/processed_XXX/datasets/gnn/train.json",
        format="gnn"
    )
"""

import json
from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle


class DiseaseDetectionDataset(Dataset):
    """
    Dataset PyTorch pour la détection de maladies (CNN ou GNN).
    
    Args:
        split_file: Chemin vers train.json, val.json ou test.json
        format: 'cnn' ou 'gnn'
        transform: Transformations PyTorch à appliquer (pour CNN)
        target_transform: Transformations pour la cible
        return_metadata: Retourner aussi les métadonnées
        
    Returns:
        Pour CNN: (image_tensor, metadata) si return_metadata else image_tensor
        Pour GNN: (graph_dict, metadata) si return_metadata else graph_dict
    """
    
    def __init__(
        self,
        split_file: str,
        format: Literal["cnn", "gnn"] = "cnn",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        return_metadata: bool = False
    ):
        self.split_file = Path(split_file)
        self.format = format
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = return_metadata
        
        # Charger le split
        if self.split_file.suffix == ".pkl":
            with open(self.split_file, "rb") as f:
                split_data = pickle.load(f)
        else:
            with open(self.split_file, "r") as f:
                split_data = json.load(f)
        
        self.samples = split_data["samples"]
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        file_path = sample["file"]
        
        # Charger les données
        data = np.load(file_path, allow_pickle=True).item()
        
        if self.format == "cnn":
            # Format CNN: (H, W) -> (1, H, W)
            img = data["data"]
            img = torch.from_numpy(img).unsqueeze(0)  # Ajouter dimension channel
            
            if self.transform:
                img = self.transform(img)
            
            if self.return_metadata:
                metadata = {
                    "sim_id": sample["sim_id"],
                    "timestep": sample["timestep"],
                    "infection_level": sample["infection_level"],
                    "original_shape": data.get("original_shape"),
                    "crop_bbox": data.get("crop_bbox")
                }
                return img, metadata
            else:
                return img
        
        else:  # GNN
            # Format GNN: dict avec nodes, edges, features
            graph = {
                "nodes": torch.from_numpy(data["nodes"]),
                "edges": torch.from_numpy(data["edges"]),
                "node_features": torch.from_numpy(data["node_features"]),
                "shape": data["shape"]
            }
            
            if self.return_metadata:
                metadata = {
                    "sim_id": sample["sim_id"],
                    "timestep": sample["timestep"],
                    "infection_level": sample["infection_level"]
                }
                return graph, metadata
            else:
                return graph


class TemporalDiseaseDataset(Dataset):
    """
    Dataset PyTorch pour séquences temporelles.
    
    Args:
        split_file: Chemin vers train_seqN.json, val_seqN.json ou test_seqN.json
        format: 'cnn' ou 'gnn'
        transform: Transformations PyTorch
        return_metadata: Retourner les métadonnées
        
    Returns:
        (sequence, target, metadata) où:
        - sequence: list de tensors ou dicts (longueur = sequence_length)
        - target: tensor ou dict (prochaine étape)
        - metadata: dict avec infos de séquence
    """
    
    def __init__(
        self,
        split_file: str,
        format: Literal["cnn", "gnn"] = "cnn",
        transform: Optional[callable] = None,
        return_metadata: bool = True
    ):
        self.split_file = Path(split_file)
        self.format = format
        self.transform = transform
        self.return_metadata = return_metadata
        
        # Charger le split
        if self.split_file.suffix == ".pkl":
            with open(self.split_file, "rb") as f:
                split_data = pickle.load(f)
        else:
            with open(self.split_file, "r") as f:
                split_data = json.load(f)
        
        self.sequences = split_data["samples"]
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def _load_sample(self, sample_dict):
        """Charge un échantillon individuel."""
        file_path = sample_dict["file"]
        data = np.load(file_path, allow_pickle=True).item()
        
        if self.format == "cnn":
            img = data["data"]
            img = torch.from_numpy(img).unsqueeze(0)
            if self.transform:
                img = self.transform(img)
            return img
        else:  # GNN
            return {
                "nodes": torch.from_numpy(data["nodes"]),
                "edges": torch.from_numpy(data["edges"]),
                "node_features": torch.from_numpy(data["node_features"]),
                "shape": data["shape"]
            }
    
    def __getitem__(self, idx: int) -> Tuple:
        seq_data = self.sequences[idx]
        
        # Charger la séquence
        sequence = [self._load_sample(s) for s in seq_data["sequence"]]
        
        # Charger la cible
        target = self._load_sample(seq_data["target"])
        
        if self.return_metadata:
            metadata = {
                "sim_id": seq_data["sim_id"],
                "start_timestep": seq_data["start_timestep"],
                "sequence_length": len(sequence)
            }
            return sequence, target, metadata
        else:
            return sequence, target


def collate_fn_cnn(batch):
    """
    Fonction de collate pour CNN batch.
    
    Args:
        batch: Liste de (image_tensor,) ou (image_tensor, metadata)
        
    Returns:
        Batch stacked
    """
    if isinstance(batch[0], tuple):
        # Avec métadonnées
        images = torch.stack([item[0] for item in batch])
        metadata = [item[1] for item in batch]
        return images, metadata
    else:
        # Sans métadonnées
        return torch.stack(batch)


def collate_fn_gnn(batch):
    """
    Fonction de collate pour GNN batch.
    
    Note: Pour PyTorch Geometric, utiliser plutôt torch_geometric.data.Batch
    
    Args:
        batch: Liste de (graph_dict,) ou (graph_dict, metadata)
        
    Returns:
        Batch de graphes (ou utiliser PyTorch Geometric Batch)
    """
    if isinstance(batch[0], tuple):
        # Avec métadonnées
        graphs = [item[0] for item in batch]
        metadata = [item[1] for item in batch]
        return graphs, metadata
    else:
        return batch


def collate_fn_temporal_cnn(batch):
    """
    Fonction de collate pour séquences temporelles CNN.
    
    Args:
        batch: Liste de (sequence, target, metadata)
        
    Returns:
        (sequences_batch, targets_batch, metadata_list)
    """
    sequences = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    metadata = [item[2] for item in batch] if len(batch[0]) > 2 else None
    
    # Stack séquences: (batch, seq_len, channels, H, W)
    sequences_batch = torch.stack([torch.stack(seq) for seq in sequences])
    targets_batch = torch.stack(targets)
    
    if metadata:
        return sequences_batch, targets_batch, metadata
    else:
        return sequences_batch, targets_batch


def get_dataloader(
    split_file: str,
    format: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[callable] = None,
    is_temporal: bool = False,
    **kwargs
):
    """
    Fonction helper pour créer un DataLoader configuré.
    
    Args:
        split_file: Chemin vers le fichier de split
        format: 'cnn' ou 'gnn'
        batch_size: Taille du batch
        shuffle: Mélanger les données
        num_workers: Nombre de workers
        transform: Transformations
        is_temporal: Utiliser dataset temporel
        **kwargs: Arguments supplémentaires pour Dataset
        
    Returns:
        DataLoader configuré
    """
    from torch.utils.data import DataLoader
    
    if is_temporal:
        dataset = TemporalDiseaseDataset(
            split_file,
            format=format,
            transform=transform,
            **kwargs
        )
        collate_fn = collate_fn_temporal_cnn if format == "cnn" else None
    else:
        dataset = DiseaseDetectionDataset(
            split_file,
            format=format,
            transform=transform,
            **kwargs
        )
        collate_fn = collate_fn_cnn if format == "cnn" else collate_fn_gnn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


# Exemple d'utilisation
if __name__ == "__main__":
    # Test CNN
    print("Test CNN Dataset:")
    try:
        dataset = DiseaseDetectionDataset(
            "data/processed/processed_20260204_173051/datasets/cnn/train.json",
            format="cnn",
            return_metadata=True
        )
        print(f"  Samples: {len(dataset)}")
        
        img, meta = dataset[0]
        print(f"  Image shape: {img.shape}")
        print(f"  Metadata: {meta}")
    except Exception as e:
        print(f"  Erreur: {e}")
    
    # Test GNN
    print("\nTest GNN Dataset:")
    try:
        dataset = DiseaseDetectionDataset(
            "data/processed/processed_20260204_173051/datasets/gnn/train.json",
            format="gnn",
            return_metadata=True
        )
        print(f"  Samples: {len(dataset)}")
        
        graph, meta = dataset[0]
        print(f"  Nodes: {graph['nodes'].shape}")
        print(f"  Edges: {graph['edges'].shape}")
        print(f"  Features: {graph['node_features'].shape}")
        print(f"  Metadata: {meta}")
    except Exception as e:
        print(f"  Erreur: {e}")
