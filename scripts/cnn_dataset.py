import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class CNNDataset(Dataset):
    """Charge les données CNN (images 64x64 .npy avec infection_level)"""
    
    def __init__(self, json_file, return_metadata=False):
        """
        Args:
            json_file: chemin vers le JSON contenant les samples
            return_metadata: si True, retourne (x, metadata) sinon (x, y)
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.samples = data['samples']
        self.return_metadata = return_metadata

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Charger le fichier .npy (contient un dict avec 'data', 'original_shape', etc.)
        npy_path = sample['file']  # Chemin relatif depuis la racine du projet
        npy_data = np.load(npy_path, allow_pickle=True).item()
        x = npy_data['data'].astype(np.float32)  # (64, 64)
        
        # Ajouter une dimension de channel: (1, 64, 64)
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x)
        
        if self.return_metadata:
            metadata = {
                'infection_level': sample['infection_level'],
                'sim_id': sample['sim_id'],
                'timestep': sample['timestep'],
                'file': sample['file']
            }
            return x, metadata
        else:
            target_npy = np.load(target['file'], allow_pickle=True).item()
            y = torch.from_numpy(target_npy['data'].astype(np.float32))  # (64, 64)
            return x, y


class TemporalCNNDataset(Dataset):
    """
    Charge les données CNN avec séquences temporelles.
    Input: séquence de L frames consécutives (B, L, 64, 64)
    Output: scalaire (infection_level au temps t+1)
    """
    
    def __init__(self, json_file, sequence_length=5, require_consecutive=True):
        """
        Args:
            json_file: chemin vers le JSON contenant les samples
            sequence_length: nombre de frames consécutives à charger (L)
            require_consecutive: si True, impose des timesteps consécutifs
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.samples = data['samples']
        self.sequence_length = sequence_length
        self.require_consecutive = require_consecutive
        
        # Organiser les samples par simulation et timestep
        self.sequences = self._build_sequences()
    
    def _build_sequences(self):
        """
        Crée les séquences temporelles en groupant par simulation
        """
        # Grouper par sim_id
        by_sim = defaultdict(list)
        for sample in self.samples:
            by_sim[sample['sim_id']].append(sample)
        
        # Trier par timestep et créer les séquences
        sequences = []
        for sim_id, samples in by_sim.items():
            # Trier par timestep
            samples.sort(key=lambda x: x['timestep'])
            
            # Créer les séquences de longueur sequence_length
            # Target = frame suivante (t+1), donc on a besoin d'un pas de plus
            for i in range(len(samples) - self.sequence_length):
                seq = samples[i:i + self.sequence_length]
                target = samples[i + self.sequence_length]

                if self.require_consecutive:
                    seq_steps = [s['timestep'] for s in seq]
                    target_step = target['timestep']
                    is_consecutive = all(
                        seq_steps[j + 1] == seq_steps[j] + 1
                        for j in range(len(seq_steps) - 1)
                    ) and (target_step == seq_steps[-1] + 1)
                    if not is_consecutive:
                        continue

                sequences.append((seq, target))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]  # (liste de L samples, sample target)
        
        # Charger toutes les frames de la séquence
        frames = []
        for sample in seq:
            npy_path = sample['file']
            npy_data = np.load(npy_path, allow_pickle=True).item()
            frame = npy_data['data'].astype(np.float32)  # (64, 64)
            frames.append(frame)
        
        # Stack: (L, 64, 64)
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames)
        
        # Target: infection_level du frame suivant (t+1)
        y = torch.tensor(target['infection_level'], dtype=torch.float32)
        
        return frames, y


def cnn_collate_fn(batch):
    """
    Collate function pour batches avec metadata.
    Si le batch contient (x, metadata), retourne:
        - Un tensor x concaténé
        - Une liste de metadata dicts
    """
    xs = []
    metas = []
    
    for item in batch:
        if len(item) == 2:  # (x, metadata) ou (x, y)
            x, meta = item
            xs.append(x)
            metas.append(meta)
        else:
            xs.append(item)
    
    x_batch = torch.stack(xs)
    
    # Si meta est un dict (return_metadata=True), garder l'ordre
    if isinstance(metas[0], dict):
        return x_batch, metas
    else:
        # Sinon c'est des targets, stack them
        return x_batch, torch.stack(metas)