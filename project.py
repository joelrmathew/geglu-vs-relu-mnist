import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import random


# ------------------ Model Definitions ------------------ #
class FFN_GeGLU(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.W_in = nn.Parameter(torch.empty(dim_in, dim_hidden))
        self.W_gate = nn.Parameter(torch.empty(dim_in, dim_hidden))
        self.W_out = nn.Parameter(torch.empty(dim_hidden, dim_hidden))
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_gate)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x):
        x_proj = torch.einsum('BD,DF->BF', x, self.W_in)
        gate = torch.einsum('BD,DF->BF', x, self.W_gate)
        gated = F.gelu(x_proj) * gate
        return torch.einsum('BF,FD->BD', gated, self.W_out)


class FFN_ReLU(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.W1 = nn.Linear(dim_in, dim_hidden)
        self.W2 = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))


# ------------------ Lightning Module ------------------ #
class MNISTClassifier(pl.LightningModule):
    def __init__(self, model_cls, dim_hidden, lr):
        super().__init__()
        self.model = model_cls(28 * 28, dim_hidden)
        self.output = nn.Linear(dim_hidden, 10)
        self.lr = lr
        self.accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.model(x)
        logits = self.output(features)
        return logits

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        val_acc = self.accuracy(logits, y)
        self.log("val_acc", val_acc, prog_bar=True)

    def test_step(self, batch, _):
        x, y = batch
        logits = self(x)
        test_acc = self.accuracy(logits, y)
        self.log("test_acc", test_acc, prog_bar=True)
        return test_acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ------------------ Data ------------------ #
def load_data(batch_size):
    transform = transforms.ToTensor()
    dataset = MNIST(root=".", train=True, download=True, transform=transform)
    testset = MNIST(root=".", train=False, download=True, transform=transform)
    trainset, valset = random_split(dataset, [55000, 5000])
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True),
        DataLoader(valset, batch_size=batch_size),
        DataLoader(testset, batch_size=batch_size)
    )


# ------------------ Bootstrap CI ------------------ #
def bootstrap_ci(data, num_samples=1000, alpha=0.05):
    means = []
    n = len(data)
    for _ in range(num_samples):
        sample = np.random.choice(data, n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(data), lower, upper


# ------------------ Experiment Loop ------------------ #
def run_experiments(model_cls, model_name, k_list, hidden_dims, batch_sizes, lrs):
    all_results = {}
    for k in k_list:
        print(f"\n\nRunning {model_name} with k={k} trials")
        results = []
        for h in hidden_dims:
            trial_accs = []
            for _ in range(k):
                bs = random.choice(batch_sizes)
                lr = random.choice(lrs)
                train_loader, val_loader, test_loader = load_data(bs)
                model = MNISTClassifier(model_cls, h, lr)
                trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False, logger=False)
                trainer.fit(model, train_loader, val_loader)
                test_result = trainer.test(model, test_loader, verbose=False)
                trial_accs.append(test_result[0]['test_acc'])
            avg, lo, hi = bootstrap_ci(trial_accs)
            results.append((h, avg, lo, hi))
        all_results[k] = results
    return all_results


# ------------------ Plotting ------------------ #
def plot_results(results_dict, title):
    for k, res in results_dict.items():
        x = [r[0] for r in res]
        y = [r[1] for r in res]
        yerr = [(r[1] - r[2], r[3] - r[1]) for r in res]
        yerr = np.array(yerr).T
        plt.errorbar(x, y, yerr=yerr, label=f"k={k}", capsize=5, fmt='-o')

    plt.title(title)
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


# ------------------ Main ------------------ #
k_list = [2, 4, 8]
hidden_dims = [2, 4, 8, 16]
batch_sizes = [8, 64]
lrs = [1e-1, 1e-2, 1e-3, 1e-4]

results_relu = run_experiments(FFN_ReLU, "FFN_ReLU", k_list, hidden_dims, batch_sizes, lrs)
results_geglu = run_experiments(FFN_GeGLU, "FFN_GeGLU", k_list, hidden_dims, batch_sizes, lrs)

plot_results(results_relu, "FFN_ReLU: MNIST Test Accuracy vs Hidden Dim")
plot_results(results_geglu, "FFN_GeGLU: MNIST Test Accuracy vs Hidden Dim")
