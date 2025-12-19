import scanpy as sc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import scaled_dot_product_attention
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
#import anndata
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import yaml

torch.backends.cudnn.benchmark = True

# """# Data Loading"""

n_classes = 20

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

n_genes = cfg["n_genes"]
train_batch_size = cfg["train_batch_size"]
load_weights = cfg["load_weights"]
n_epochs = cfg["n_epochs"]

save_by_epoch = cfg["save_by_epoch"]
save_frequency = cfg["save_frequency"]

weights_name = cfg["weights_name"]
cm_name = cfg["cm_name"]
attn_name = cfg["attn_name"]
seurat_expression_data = sc.read_h5ad(cfg["seurat_expression_data"])

X = seurat_expression_data.X
if hasattr(X, "toarray"):  # Convert sparse matrix to dense if necessary
    X = X.toarray()
X = np.nan_to_num(X)  # Replace NaNs with 0

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

X = normalize(X, axis=1, norm='l2')  # L2 normalization along rows

y = seurat_expression_data.obs["Type"].to_numpy()
X, y = shuffle(X, y, random_state=0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

print(max(y_encoded), min(y_encoded))

print(np.unique(y, return_counts=True))

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_labels(self):
        return self.labels

train_batch_size = 1024

train_num = int(len(y_encoded) * 0.75)
val_num = int(len(y_encoded) * 0.1)
test_num = int(len(y_encoded) * 0.15)
train_ds = CustomDataset(X[:train_num], y_encoded[:train_num])
val_ds = CustomDataset(X[train_num:train_num + val_num], y_encoded[train_num:train_num + val_num])
test_ds = CustomDataset(X[train_num + val_num:], y_encoded[train_num + val_num:])

train_loader = DataLoader(
    train_ds,
    batch_size=train_batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=train_batch_size,
    shuffle=True
)
test_loader = DataLoader(
    test_ds,
    batch_size=train_batch_size,
    shuffle=True
)

"""# Muon"""

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

class MultipleScheduler(object):
    def __init__(self, *op):
        self.schedulers = op

    def step(self):
        for op in self.schedulers:
            op.step()

"""# Training and Testing Functions"""

def test():
    model.eval()
    total_loss = 0
    cnt = 0
    correct = 0
    total_cnt = 0
    with torch.no_grad():
        for features, labels in tqdm(test_loader):
            labels = labels.type(torch.LongTensor)
            features, labels = features.to(device), labels.to(device)

            outputs, _ = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            cnt += 1
            correct += (np.sum((outputs.argmax(-1) == labels).detach().cpu().numpy()))
            total_cnt += len(labels)

    print(f"Loss: {total_loss / cnt:.4f} Percentage Correct: {correct * 100 / total_cnt:.4f}%")

def validate():
    model.eval()
    total_loss = 0
    cnt = 0
    correct = 0
    total_cnt = 0
    with torch.no_grad():
        for features, labels in tqdm(val_loader):
            labels = labels.type(torch.LongTensor)
            features, labels = features.to(device), labels.to(device)

            outputs, _ = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            cnt += 1
            correct += (np.sum((outputs.argmax(-1) == labels).detach().cpu().numpy()))
            total_cnt += len(labels)

    print(f"Loss: {total_loss / cnt:.4f} Percentage Correct: {correct * 100 / total_cnt:.4f}%")

"""# Model Definition"""

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.sqrt_dim = torch.tensor(hidden_dim**0.5, dtype=torch.bfloat16)
        self.Query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Value = nn.Linear(hidden_dim, hidden_dim, bias=False)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, x):
        # x: [batch size, seq len, hidden dim]
        query = self.Query(x)
        # query: [batch size, seq len, hidden dim]
        key = self.Key(x)
        # key: [batch size, hidden dim, seq len]
        value = self.Value(x)
        # value: [batch size, seq len, hidden dim]

        attn_weights = F.softmax(torch.bmm(query, key.permute(0, 2, 1)) / self.sqrt_dim, dim=-1)
        # attn_weights: [batch size, seq len, seq len]

        attn_output = torch.bmm(attn_weights, value)
        # attn_output: [batch size, seq len, hidden dim]

        return attn_output, attn_weights

class TransformerPretrain(nn.Module):
    def __init__(self, hidden_dim, seq_len, num_classes, ffn_dim=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.W = nn.Parameter(torch.randn(seq_len, 1, hidden_dim))
        self.b = nn.Parameter(torch.zeros(seq_len, hidden_dim))

        # self attention
        self.attn_1 = SelfAttention(hidden_dim)
        self.ffn_1 = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
            # nn.LayerNorm([hidden_dim]), # If architecture is different, remove this line
        )

        self.reconstruct_head = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, x):
        # x: [batch size, seq len]
        x = x.unsqueeze(-1)
        x = torch.matmul(x.unsqueeze(-2), self.W)  # -> [batch, seq_len, 1, out_dim]
        x = x.squeeze(-2) + self.b            # -> [batch, seq_len, out_dim]
        x = F.gelu(x)

        # x = self.dropout(x)

        # Attention Stuff Below
        attn_x_1, attn_1_weights = self.attn_1(x)
        # attn_x_1: [batch size, seq len, hidden dim]
        attn_x_1 = F.gelu(self.ffn_1(attn_x_1))
        x = attn_x_1

        x = x.unsqueeze(1)
        # x: [batch size, seq len, hidden dim, 1]
        x = self.reconstruct_head(x)
        # x: [batch size, seq len, hidden, 1]
        return x.squeeze().float()

class TransformerClassifier(nn.Module):
    def __init__(self, hidden_dim, seq_len, num_classes, ffn_dim=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        self.W = nn.Parameter(torch.randn(seq_len, 1, hidden_dim))
        self.b = nn.Parameter(torch.zeros(seq_len, hidden_dim))

        # self attention
        self.attn_1 = SelfAttention(hidden_dim)
        self.ffn_1 = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
            # nn.LayerNorm([hidden_dim]), # If architecture is different, remove this line
        )

        # to pool and obtain final output
        self.conv_pool = nn.Conv2d(1, 1, kernel_size=(seq_len, 1))
        # final fc
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.25)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, x):
        # x: [batch size, seq len]
        x = x.unsqueeze(-1)
        x = torch.matmul(x.unsqueeze(-2), self.W)  # -> [batch, seq_len, 1, out_dim]
        x = x.squeeze(-2) + self.b            # -> [batch, seq_len, out_dim]
        x = F.gelu(x)

        # Attention Stuff Below
        attn_x_1, attn_1_weights = self.attn_1(x)
        # attn_x_1: [batch size, seq len, hidden dim]
        attn_x_1 = F.gelu(self.ffn_1(attn_x_1))
        x = attn_x_1

        x = x.unsqueeze(1)
        # x: [batch size, seq len, hidden dim]
        x = F.gelu(self.conv_pool(x))
        x = x.squeeze()
        # x: [batch size, 128]
        return self.fc(x).float(), attn_1_weights

"""# Model Pretraining

Pretrain with reconstruction objective?
"""

pretraining_model = TransformerPretrain(64, n_genes, n_classes, ffn_dim=16).to(device)

"""# Full Model Training"""

model = TransformerClassifier(64, n_genes, n_classes, ffn_dim=16).to(device)
load_weights = False

# Run if loading pre-trained model weights
# if load_weights:
#     try:
#         model.load_state_dict(torch.load("WEIGHTS_NAME.pt", weights_only=True))
#         print("Model weights loaded successfully.")
#     except FileNotFoundError as e:
#         print("No existing weights found. Training from scratch")
#     except RuntimeError as e:
#         state_dict = torch.load("WEIGHTS_NAME.pt", weights_only=True)
#         model.W.weight = state_dict['W']
#         model.b.weight = state_dict['b']
#         model.attn_1.Query.weight.data = state_dict['attn_1.Query.weight']
#         model.attn_1.Key.weight.data = state_dict['attn_1.Key.weight']
#         model.attn_1.Value.weight.data = state_dict['attn_1.Value.weight']
#         print("Model architecture is different. Loaded partial weights")
# else:
#     print("Selected to train from scratch")

sum(p.numel() for p in model.parameters())

if device == "cuda":
    class_weights = torch.tensor(1 / np.unique(y_encoded, return_counts=True)[1], dtype=torch.float32).cuda()
elif device == "cpu":
    class_weights = torch.tensor(1 / np.unique(y_encoded, return_counts=True)[1], dtype=torch.float32)

class_weights = class_weights ** 0.5
class_weights[2] = class_weights[2] * 2.0
class_weights[3] = class_weights[3] * 1.5
class_weights[4] = class_weights[4] * 1.5
class_weights[15] = class_weights[15] * 1.1
class_weights[16] = class_weights[16] * 1.2
class_weights = class_weights / sum(class_weights)
class_weights

from torch.optim.lr_scheduler import *
from torch.optim import *
criterion = nn.CrossEntropyLoss(label_smoothing=0.00, weight=class_weights)

hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2]

param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=1e-3, weight_decay=0, momentum=0.95, nesterov=True),
]
muon_optimizer = SingleDeviceMuon(param_groups)
muon_scheduler = ExponentialLR(muon_optimizer, gamma=0.999)

param_groups = [
    dict(params=hidden_gains_biases, lr=5e-5, weight_decay=0),
]
adamw_optimizer = AdamW(param_groups)
adamw_scheduler = ExponentialLR(adamw_optimizer, gamma=0.999)

optimizer = MultipleOptimizer(muon_optimizer, adamw_optimizer)
scheduler = MultipleScheduler(muon_scheduler, adamw_scheduler)

print("Training the Transformer model...")
# Run training iteration
import time
start = time.perf_counter()
for epoch in range(n_epochs):  # Number of epochs
    model.train()
    total_loss = 0
    cnt = 0
    correct = 0
    total_cnt = 0
    for features, labels in (pbar := tqdm(train_loader)):
        labels = labels.type(torch.LongTensor)
        features, labels = features.to(device), labels.to(device)

        outputs, attn_weights = model(features)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()  # Better memory management

        total_loss += loss.item()
        cnt += 1
        correct += (np.sum((outputs.argmax(-1) == labels).detach().cpu().numpy()))
        total_cnt += len(labels)
        pbar.set_description(f"Epoch {epoch + 1}, Loss: {total_loss / cnt:.4f}")

    print(f"Epoch {epoch + 1} Percentage Correct: {correct * 100 / total_cnt:.4f}%")
    test()
    scheduler.step()

    # --- Save model every save_frequency epochs ---
    if save_by_epoch and (epoch + 1) % save_frequency == 0:
        save_path = f"{weights_name}_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Saved model weights to {save_path}")

end = time.perf_counter()
print("Ending timer...")
print("Raw time usage:")
print(end - start)
runtime = round((end - start), 10)
mins = int(runtime // 60)
secs = runtime % 60
if runtime < 60:
    print(f"Runtime: {secs:.10f} seconds")
else:
    print(f"Runtime: {mins} minutes {secs:.10f} seconds")

    # --- Save model every 10 epochs ---
    # if (epoch + 1) % 10 == 0:
    #     save_path = f"final_trained_ssthensl2_CosMx850_200epochs_l2_epoch_{epoch + 1}.pt"
    #     torch.save(model.state_dict(), save_path)
    #     print(f"Saved model weights to {save_path}")

# test()

torch.save(model.state_dict(), weights_name)
print("Model weights saved successfully!")

# """# Evaluation"""

from sklearn.metrics import confusion_matrix

def calc_confusion_matrix():
    model.eval()
    total_loss = 0
    cnt = 0
    correct = 0
    total_cnt = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for features, labels in tqdm(test_loader):
            labels = labels.type(torch.LongTensor)
            features, labels = features.to(device), labels.to(device)

            outputs, _ = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            cnt += 1
            # print(outputs.argmax(-1) == labels)
            correct += (np.sum((outputs.argmax(-1) == labels).detach().cpu().numpy()))
            preds = outputs.argmax(-1).detach().cpu().tolist()
            actuals = labels.detach().cpu().tolist()
            for i in preds:
                y_pred.append(i)
            for i in actuals:
                y_true.append(i)
            total_cnt += len(labels)
            # if cnt >= 2000:
            #     break
    print(f"Loss: {total_loss / cnt:.4f} Percentage Correct: {correct * 100 / total_cnt:.4f}%")
    return confusion_matrix(y_true, y_pred, normalize='true')

conf_mat = calc_confusion_matrix()
conf_mat.shape

import seaborn as sns

sns.set_theme(rc={'figure.figsize':(10, 8)})
sns.heatmap(
    conf_mat.round(2),
    cmap='coolwarm',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    annot=True,
    # fmt='d'
)

np.save(cm_name, conf_mat)

large_loader = DataLoader(
    test_ds,
    # sampler=ImbalancedDatasetSampler(test_ds),
    batch_size=train_batch_size, #16384
    shuffle=True
)

with torch.no_grad():
    for features, labels in tqdm(large_loader):
        labels = labels.type(torch.LongTensor)
        features, labels = features.to(device), labels.to(device)
        break

def compute_attn_weights(features):
    with torch.no_grad():
        x = features.unsqueeze(-1)

        x = torch.matmul(x.unsqueeze(-2), model.W)  # -> [batch, seq_len, 1, out_dim]
        x = x.squeeze(-2) + model.b            # -> [batch, seq_len, out_dim]
        x = F.gelu(x)

        query = model.attn_1.Query(x)

        key = model.attn_1.Key(x)

        value = model.attn_1.Value(x)

        attn_weights = F.softmax(torch.bmm(query, key.permute(0, 2, 1)) / (model.hidden_dim**0.5), dim=-1)
    return attn_weights.detach().cpu().numpy().astype(np.float16)

detached_attn_weights = np.zeros((features.shape[0], n_genes, n_genes)).astype(np.float16)
for i in tqdm(range(0, len(features), 1024)):
    detached_attn_weights[i:i+1024] = compute_attn_weights(features[i:i+1024])
detached_attn_weights = np.array(detached_attn_weights)
detached_attn_weights.shape

gene_names = seurat_expression_data.var.index.tolist()

def get_cell_by_type(labels, cellname):
    idx = label_encoder.transform([cellname])
    return np.where(labels.cpu().numpy() == idx)

def get_top_attn_genes(attn_weights, top_k=10):
    flattened = attn_weights.ravel()
    idx = list(np.argpartition(flattened, -top_k)[-top_k:])
    gene_pairs = []
    for id in idx:
        gene_1 = id // len(gene_names)
        gene_2 = id % len(gene_names)
        gene_pairs.append((gene_names[gene_1], gene_names[gene_2]))
    return gene_pairs

attn_by_celltype = []
for celltypename in np.unique(y):
    print(celltypename)
    all_attn = detached_attn_weights[get_cell_by_type(labels, celltypename)].mean(axis=0)
    attn_by_celltype.append(all_attn)

celltype_avg_attn = np.array(attn_by_celltype)

np.save(attn_name, celltype_avg_attn)

