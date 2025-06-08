import numpy as np
import torch
from torch.utils.data import Dataset

class NPZSequenceDataset(Dataset):
    def __init__(self, file_list, window_size=5, stride=1, device='cpu'):
        self.samples = []
        self.window_size = window_size
        self.stride = stride
        self.device = device

        for file_path in file_list:
            data = np.load(file_path)
            embeddings = data['embeddings']
            labels = data['labels']

            T = len(labels)
            for i in range(0, T - window_size + 1, stride):
                emb_seq = embeddings[i:i+window_size]
                label = labels[i + window_size - 1]
                self.samples.append((emb_seq, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emb_seq, label = self.samples[idx]
        emb_tensor = torch.tensor(emb_seq, dtype=torch.float32).to(self.device)
        label_tensor = torch.tensor(label, dtype=torch.long).to(self.device)
        return emb_tensor, label_tensor

def load_fold_file_list(fold_index):
    with open(f"fold_{fold_index}.txt") as f:
        val_files = [line.strip() for line in f.readlines()]
    
    train_files = []
    for i in range(5):
        if i != fold_index:
            with open(f"fold_{i}.txt") as f:
                train_files.extend([line.strip() for line in f.readlines()])
    
    return train_files, val_files
