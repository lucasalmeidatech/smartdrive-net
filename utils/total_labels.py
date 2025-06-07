import numpy as np
import os

# Caminho para a pasta dos npz
dataset_path = 'clipData/dataset'

# Lista para guardar todos os labels
all_labels = []

# Percorre todos os arquivos npz
for file in os.listdir(dataset_path):
    if file.endswith('.npz') and 'Dashboard' in file:
        filepath = os.path.join(dataset_path, file)
        data = np.load(filepath)
        all_labels.append(data['labels'])

# Junta todos os labels
all_labels = np.hstack(all_labels)

# Salva
np.savez(os.path.join(dataset_path, 'total_labels.npz'), labels=all_labels)

print(f"Arquivo total_labels.npz criado com {all_labels.shape[0]} labels.")
