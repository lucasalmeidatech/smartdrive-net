import os
import numpy as np

if __name__ == '__main__':


    # Caminho agora para o treino e validação
    dataset_path = 'clipData/dataset'

    # Listas para armazenar embeddings e labels
    dashboard_embeddings = []
    dashboard_labels = []

    rearview_embeddings = []
    rearview_labels = []

    side_embeddings = []
    side_labels = []

    # Processar todos os arquivos
    for file in os.listdir(dataset_path):
        if file.endswith('.npz'):
            filepath = os.path.join(dataset_path, file)
            data = np.load(filepath)

            if 'Dashboard' in file:
                dashboard_embeddings.append(data['embeddings'])
                dashboard_labels.append(data['labels'])

            elif 'Rear_view' in file or 'Rearview' in file:
                rearview_embeddings.append(data['embeddings'])
                rearview_labels.append(data['labels'])

            elif 'Right_side_window' in file or 'RightMirror' in file or 'Side' in file:
                side_embeddings.append(data['embeddings'])
                side_labels.append(data['labels'])

    # Salvar embeddings combinados
    if dashboard_embeddings:
        dashboard_embeddings = np.vstack(dashboard_embeddings)
        dashboard_labels = np.hstack(dashboard_labels)
        np.savez(os.path.join(dataset_path, 'total_dashboard_data.npz'), embeddings=dashboard_embeddings, labels=dashboard_labels)
        print(f"Salvo total_dashboard_data.npz com {dashboard_embeddings.shape[0]} amostras.")

    if rearview_embeddings:
        rearview_embeddings = np.vstack(rearview_embeddings)
        rearview_labels = np.hstack(rearview_labels)
        np.savez(os.path.join(dataset_path, 'total_rearview_data.npz'), embeddings=rearview_embeddings, labels=rearview_labels)
        print(f"Salvo total_rearview_data.npz com {rearview_embeddings.shape[0]} amostras.")

    if side_embeddings:
        side_embeddings = np.vstack(side_embeddings)
        side_labels = np.hstack(side_labels)
        np.savez(os.path.join(dataset_path, 'total_side_data.npz'), embeddings=side_embeddings, labels=side_labels)
        print(f"Salvo total_side_data.npz com {side_embeddings.shape[0]} amostras.")

    # Salvar o arquivo total_labels.npz
    all_labels = np.hstack(dashboard_labels)  # Pegamos dos dashboards como referência
    np.savez(os.path.join(dataset_path, 'total_labels.npz'), labels=all_labels)
    print(f"Salvo total_labels.npz com {all_labels.shape[0]} labels.")

