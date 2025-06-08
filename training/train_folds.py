import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset_tcn import NPZSequenceDataset, load_fold_file_list

# ======== Configurações ========
window_size = 5
stride = 1
batch_size = 32
epochs = 10
learning_rate = 0.001
embedding_dim = 768
num_classes = 17
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== Modelo TCN ========
class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        from torch.nn.utils import weight_norm
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                             stride=1, padding=(kernel_size-1)*dilation_size,
                                             dilation=dilation_size)),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        y = self.network(x)
        y = y[:, :, -1]  # último time step
        out = self.linear(y)
        return out

# ======== Função principal por fold ========
def train_model(fold_index):
    print(f"\n=== Treinando Fold {fold_index} ===")

    train_files, val_files = load_fold_file_list(fold_index=fold_index)
    train_dataset = NPZSequenceDataset(train_files, window_size, stride, device=device)
    val_dataset = NPZSequenceDataset(val_files, window_size, stride, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TCNModel(input_size=embedding_dim, output_size=num_classes, num_channels=[64, 64, 64])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    metrics = {
        'epoch': [],
        'val_loss': [],
        'val_acc': [],
        'f1_macro': [],
        'auroc': [],
        'brier': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.2f}%")

        # ======== Validação ========
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X)
                probs = torch.softmax(outputs, dim=1)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(probs.data, 1)

                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

                all_labels.extend(y.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        f1_macro = f1_score(all_labels, all_preds, average='macro')

        try:
            auroc = roc_auc_score(np.eye(num_classes)[all_labels], all_probs, average='macro', multi_class='ovr')
        except ValueError:
            auroc = float('nan')

        true_one_hot = np.eye(num_classes)[all_labels]
        brier = np.mean(np.sum((np.array(all_probs) - true_one_hot) ** 2, axis=1))

        print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        print(f"F1 Macro: {f1_macro:.4f} | AUROC: {auroc:.4f} | Brier Score: {brier:.4f}\n")

        metrics['epoch'].append(epoch + 1)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['f1_macro'].append(f1_macro)
        metrics['auroc'].append(auroc)
        metrics['brier'].append(brier)

    # ======== Exportar CSV ========
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(f"metrics_fold_{fold_index}.csv", index=False)

    # ======== Plotar gráfico ========
    plt.figure(figsize=(10, 6))
    plt.plot(df_metrics['epoch'], df_metrics['val_acc'], label='Val Acc')
    plt.plot(df_metrics['epoch'], df_metrics['f1_macro'], label='F1 Macro')
    plt.plot(df_metrics['epoch'], df_metrics['auroc'], label='AUROC')
    plt.plot(df_metrics['epoch'], df_metrics['brier'], label='Brier Score')
    plt.xlabel("Época")
    plt.ylabel("Valor")
    plt.title(f"Métricas Fold {fold_index}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"metrics_fold_{fold_index}.png")
    plt.close()

# ======== Rodar todos os folds ========
if __name__ == "__main__":
    for fold_index in range(5):
        train_model(fold_index)
