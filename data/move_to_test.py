import os
import random
import shutil

# Diretórios
dataset_dir = 'clipData/dataset'
test_dir = 'clipData/trainTest'

# Lista de todos os arquivos .npz no dataset
all_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]

# Quantos arquivos mover para teste (20%)
num_test_files = int(0.2 * len(all_files))

# Escolher aleatoriamente os arquivos
test_files = random.sample(all_files, num_test_files)

# Mover os arquivos
for file in test_files:
    src = os.path.join(dataset_dir, file)
    dst = os.path.join(test_dir, file)
    shutil.move(src, dst)

print(f"Movidos {num_test_files} arquivos para {test_dir}")