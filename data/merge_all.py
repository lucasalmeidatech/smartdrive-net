import numpy as np
import os

# Caminho para a pasta onde estão os .npz
dataset_path = 'clipData/dataset'

# Inicializa listas para embeddings e labels
all_embeddings = []
all_labels = []

# Percorre todos arquivos .npz
for file in os.listdir(dataset_path):
    if file.endswith('.npz'):
        data = np.load(os.path.join(dataset_path, file))
        all_embeddings.append(data['embeddings'])
        all_labels.append(data['labels'])

# Concatena tudo em um único array
all_embeddings = np.vstack(all_embeddings)
all_labels = np.hstack(all_labels)

# Salva o novo arquivo .npz
np.savez(os.path.join(dataset_path, 'total_dashboard_data.npz'), embeddings=all_embeddings, labels=all_labels)

print(f"total_dashboard_data.npz salvo com sucesso! {all_embeddings.shape[0]} amostras.")
