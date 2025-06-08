import os
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedKFold
import pandas as pd

base_path = r'C:\Users\USER\smartdrive-net\data\embeddings'
folders = [f'clip_embeddingsA1_{i}' for i in range(1, 8)]

file_paths = []
user_ids = []
main_labels = []

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.npz'):
            filepath = os.path.join(folder_path, filename)
            try:
                data = np.load(filepath)
                labels = data['labels']
                most_common_label = Counter(labels.tolist()).most_common(1)[0][0]
                user_id = int(filename.split('_')[filename.split('_').index('id') + 1])

                file_paths.append(filepath)
                user_ids.append(user_id)
                main_labels.append(most_common_label)
            except Exception as e:
                print(f"Erro ao processar {filepath}: {e}")

df = pd.DataFrame({'file_path': file_paths, 'user_id': user_ids, 'label': main_labels})

# Agrupar por user_id para criar uma amostra por motorista
grouped = df.groupby('user_id')
user_id_list = []
user_major_label = []

for user_id, group in grouped:
    major_label = group['label'].value_counts().idxmax()
    user_id_list.append(user_id)
    user_major_label.append(major_label)

# Stratified 5-Fold com base no label majoritário de cada motorista
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = defaultdict(list)

for fold_idx, (_, test_idx) in enumerate(skf.split(user_id_list, user_major_label)):
    for i in test_idx:
        uid = user_id_list[i]
        user_files = df[df['user_id'] == uid]['file_path'].tolist()
        folds[fold_idx].extend(user_files)

# Salvar os folds (opcional)
for fold, paths in folds.items():
    with open(f'fold_{fold}.txt', 'w') as f:
        for path in paths:
            f.write(f"{path}\n")