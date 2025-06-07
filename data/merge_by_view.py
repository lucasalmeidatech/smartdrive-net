import os
import numpy as np

# Pasta onde estão todos os .npz
dataset_path = 'clipData/dataset'

# Listas para cada vista
dashboard_embeddings = []
dashboard_labels = []

rearview_embeddings = []
rearview_labels = []

rightmirror_embeddings = []
rightmirror_labels = []

# Processa todos os arquivos .npz
for file in os.listdir(dataset_path):
    if file.endswith('.npz'):
        filepath = os.path.join(dataset_path, file)
        data = np.load(filepath)
        
        if 'Dashboard' in file:
            dashboard_embeddings.append(data['embeddings'])
            dashboard_labels.append(data['labels'])
        elif 'Rearview' in file:
            rearview_embeddings.append(data['embeddings'])
            rearview_labels.append(data['labels'])
        elif 'RightMirror' in file:
            rightmirror_embeddings.append(data['embeddings'])
            rightmirror_labels.append(data['labels'])

# Junta os dados de cada câmera
if dashboard_embeddings:
    dashboard_embeddings = np.vstack(dashboard_embeddings)
    dashboard_labels = np.hstack(dashboard_labels)
    np.savez(os.path.join(dataset_path, 'total_dashboard_data.npz'), embeddings=dashboard_embeddings, labels=dashboard_labels)
    print(f"Salvo total_dashboard_data.npz com {dashboard_embeddings.shape[0]} amostras.")
else:
    print("Nenhum dado de Dashboard encontrado.")

if rearview_embeddings:
    rearview_embeddings = np.vstack(rearview_embeddings)
    rearview_labels = np.hstack(rearview_labels)
    np.savez(os.path.join(dataset_path, 'total_rearview_data.npz'), embeddings=rearview_embeddings, labels=rearview_labels)
    print(f"Salvo total_rearview_data.npz com {rearview_embeddings.shape[0]} amostras.")
else:
    print("Nenhum dado de Rearview encontrado.")

if rightmirror_embeddings:
    rightmirror_embeddings = np.vstack(rightmirror_embeddings)
    rightmirror_labels = np.hstack(rightmirror_labels)
    np.savez(os.path.join(dataset_path, 'total_rightmirror_data.npz'), embeddings=rightmirror_embeddings, labels=rightmirror_labels)
    print(f"Salvo total_rightmirror_data.npz com {rightmirror_embeddings.shape[0]} amostras.")
else:
    print("Nenhum dado de RightMirror encontrado.")
